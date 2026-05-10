"""AEC bench: play TTS speech through the speaker, capture the mic loop,
push the mic through AECProcessor, and report per-second RMS dBFS for the
far reference, the near (mic) signal, and the AEC output.

The interesting number is ``suppression = near - out``: positive means
AEC reduced the energy of whatever bled from speaker into mic.  No human
voice should be present during the run — that way the near signal is a
pure room loop and the AEC output ideally approaches the noise floor.

Critically, this bench drives audio through a **duplex** PortAudio
stream (``sd.RawStream``), not separate input + output streams.  Two
independent streams in PipeWire/ALSA each have their own clock — they
drift relative to each other by ~20 ms/sec, which makes any sample-aligned
AEC algorithm fail.  A duplex stream invokes one callback for both
directions on the output device's clock, so mic and far frames are
naturally time-aligned and the AEC's adaptive filter can converge.

    OPENAI_API_KEY=...  uv run python aec_bench.py
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import struct
import sys
import threading
import time
from collections import defaultdict

import aiohttp
import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
import webrtcvad  # type: ignore[import-untyped]

from aec import WebRtcAECProcessor


async def fetch_tts(text: str, *, voice: str, sample_rate_target: int) -> bytes:
    """OpenAI TTS → 24 kHz PCM16 mono. Resamples to *sample_rate_target*
    AEC3 only supports {16, 32, 48} kHz, so the bench resamples this
    24 kHz response to whatever ``--rate`` selects before driving the
    speaker."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set — needed for TTS")
    url = "https://api.openai.com/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "pcm",  # 24 kHz, signed 16-bit, little-endian, mono
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise SystemExit(f"TTS HTTP {resp.status}: {body[:200]}")
            return await resp.read()


def _energy(pcm: bytes) -> tuple[float, int]:
    """Return ``(sum_of_squares, n_samples)`` for PCM16 mono."""
    if not pcm:
        return 0.0, 0
    n = len(pcm) // 2
    if n == 0:
        return 0.0, 0
    samples = struct.unpack_from(f"<{n}h", pcm)
    return float(sum(s * s for s in samples)), n


def _to_dbfs(sumsq: float, n: float) -> float:
    if n == 0 or sumsq <= 0.0:
        return -120.0
    mean_sumsq = sumsq / n
    return 10.0 * math.log10(mean_sumsq / (32768.0**2))


async def main_async(args: argparse.Namespace) -> int:
    # OpenAI TTS pcm response is fixed at 24 kHz.  ``--rate`` controls the
    # rate we actually drive the audio device at — match the device's
    # native rate (e.g. 48000 Hz on most laptops, 44100 Hz on PipeWire's
    # default ``pipewire`` virtual device) to bypass PortAudio's
    # internal resampler, which is the dominant source of input/output
    # clock drift on non-pro audio stacks.
    sr_tts = 24000
    sr = args.rate
    text = args.text or (
        "The quick brown fox jumps over the lazy dog. "
        "How razorback-jumping frogs can level six piqued gymnasts. "
        "Pack my box with five dozen liquor jugs."
    )
    print(f"[bench] fetching TTS ({args.voice}, {len(text)} chars) …", flush=True)
    tts_24k = await fetch_tts(text, voice=args.voice, sample_rate_target=sr_tts)
    if sr != sr_tts:
        from aec import _linear_resample_pcm16

        tts_mono = _linear_resample_pcm16(tts_24k, sr_tts, sr)
    else:
        tts_mono = tts_24k
    if args.channels == 2:
        # Interleave each mono sample twice → stereo with identical L/R
        # so the device hears the same content on both channels.  Without
        # this, asking PortAudio for 2-channel output on a mono PCM
        # buffer ends up playing on the first channel only on some
        # backends (e.g. PipeWire's stereo Echo-Cancel Sink).
        mono = np.frombuffer(tts_mono, dtype="<i2")
        stereo = np.repeat(mono, 2)
        tts = stereo.tobytes()
    else:
        tts = tts_mono
    bytes_per_frame_total = 2 * args.channels  # int16 * channels
    duration_s = len(tts) / bytes_per_frame_total / sr
    print(
        f"[bench] got {len(tts)} bytes / {duration_s:.2f} s of speech at {sr} Hz × {args.channels} ch",
        flush=True,
    )

    aec: WebRtcAECProcessor | None = None
    if not args.bypass_aec:
        aec = WebRtcAECProcessor(
            sample_rate=sr,
            output_rate=sr,
            latency_hint_ms=args.latency_ms,
        )

    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"far": [0.0, 0], "near": [0.0, 0], "out": [0.0, 0]}
    )
    buckets_lock = threading.Lock()
    # Full recordings of near and out so we can replay them through
    # webrtcvad after the bench and compare voice-trigger rates.  The
    # callback runs at PortAudio's pace; appending bytes is O(1) and the
    # GIL is released while sounddevice writes the buffers.
    near_recording = bytearray()
    out_recording = bytearray()

    def _resolve(spec: str | None) -> int | str | None:
        if spec is None:
            return None
        try:
            return int(spec)
        except ValueError:
            return spec

    if args.input_device or args.output_device:
        device: object = (
            _resolve(args.input_device) if args.input_device else _resolve(args.device),
            _resolve(args.output_device) if args.output_device else _resolve(args.device),
        )
    else:
        device = _resolve(args.device)
    aec_mode = "OFF (mic→out passthrough)" if aec is None else "webrtc-aec3 (AEC3 + NS + HPF + TS)"
    print(
        f"[bench] sr={sr} latency_ms={args.latency_ms} device={device!r} aec={aec_mode}",
        flush=True,
    )
    print("[bench] DO NOT speak during the run — the bench needs a clean loop.", flush=True)

    # Duplex callback state:
    # - ``playback_buffer``: TTS bytes still to play (consumed left → right)
    # - ``t0``: monotonic clock at first callback, for per-second bucketing
    playback_buffer = bytearray(tts)
    state: dict[str, float] = {"t0": 0.0}
    done_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    deadline_s = duration_s + args.tail_s

    def add(sec: int, key: str, pcm: bytes) -> None:
        sumsq, n = _energy(pcm)
        with buckets_lock:
            bucket = buckets[sec][key]
            bucket[0] += sumsq
            bucket[1] += n

    ch = args.channels

    def callback(
        indata: memoryview,
        outdata: memoryview,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if state["t0"] == 0.0:
            state["t0"] = time.monotonic()
        sec = int(time.monotonic() - state["t0"])

        # 1) Drive the speaker: copy the next ``frames * 2 * ch`` bytes
        #    from playback_buffer into outdata, pad with silence when
        #    drained.  The buffer is already laid out as interleaved
        #    PCM16 with ``ch`` channels.
        wanted = frames * 2 * ch
        n = min(wanted, len(playback_buffer))
        if n:
            outdata[:n] = bytes(playback_buffer[:n])
            del playback_buffer[:n]
        if n < wanted:
            outdata[n:wanted] = b"\x00" * (wanted - n)

        # Downmix to mono for AEC and VAD (averages all channels per frame).
        if ch == 1:
            far_pcm = bytes(outdata[:wanted])
            near_pcm = bytes(indata)
        else:
            far_pcm = (
                np.frombuffer(bytes(outdata[:wanted]), dtype="<i2")
                .reshape(-1, ch)
                .mean(axis=1)
                .astype("<i2")
                .tobytes()
            )
            near_pcm = np.frombuffer(bytes(indata), dtype="<i2").reshape(-1, ch).mean(axis=1).astype("<i2").tobytes()

        if aec is not None:
            aec.feed_speaker(far_pcm)
            cleaned = aec.process_mic(near_pcm)
        else:
            cleaned = near_pcm
        add(sec, "far", far_pcm)
        add(sec, "near", near_pcm)
        add(sec, "out", cleaned)
        near_recording.extend(near_pcm)
        out_recording.extend(cleaned)

        # 3) Schedule shutdown when both the playback buffer is drained
        #    and the post-roll has elapsed.
        if not playback_buffer and (time.monotonic() - state["t0"]) >= deadline_s:
            loop.call_soon_threadsafe(done_event.set)

    stream = sd.RawStream(
        samplerate=sr,
        channels=(ch, ch),
        dtype="int16",
        device=device,
        callback=callback,
    )
    with stream:
        await done_event.wait()

    # Report
    print()
    print(f"{'sec':>4}  {'far':>7}  {'near':>7}  {'out':>7}  {'supp':>8}")
    print(f"{'---':>4}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>8}")
    for sec in sorted(buckets):
        b = buckets[sec]
        far = _to_dbfs(*b["far"])
        near = _to_dbfs(*b["near"])
        out = _to_dbfs(*b["out"])
        supp = near - out
        print(f"{sec:>4}  {far:>+7.1f}  {near:>+7.1f}  {out:>+7.1f}  {supp:>+7.1f} dB")

    # Aggregate, weighted by *near* energy so quiet seconds don't dominate.
    near_total_sumsq = sum(b["near"][0] for b in buckets.values())
    out_total_sumsq = sum(b["out"][0] for b in buckets.values())
    n_total = sum(b["near"][1] for b in buckets.values())
    near_db = _to_dbfs(near_total_sumsq, n_total)
    out_db = _to_dbfs(out_total_sumsq, n_total)
    print()
    print(f"weighted near = {near_db:+.2f} dBFS")
    print(f"weighted out  = {out_db:+.2f} dBFS")
    print(f"avg suppression = {near_db - out_db:+.2f} dB")
    if aec is not None:
        print(
            f"far drift drops = {aec.far_overflow_bytes} bytes "
            f"({aec.far_overflow_bytes / 2 / sr * 1000:.1f} ms of audio)"
        )

    # VAD analysis — webrtcvad supports {8000, 16000, 32000, 48000} Hz.
    # Run it on whichever of these is the closest match to ``sr`` so we
    # don't have to introduce a high-quality resampler just for VAD.
    if sr in (8000, 16000, 32000, 48000):
        vad_sr = sr
        near_for_vad = bytes(near_recording)
        out_for_vad = bytes(out_recording)
    else:
        # Decimate to 16 kHz via the same crude linear resampler.  Fine
        # for VAD's ~spectral-energy-only decision; would be wrong for
        # ASR but VAD doesn't care about pitch quality.
        from aec import _linear_resample_pcm16

        vad_sr = 16000
        near_for_vad = _linear_resample_pcm16(bytes(near_recording), sr, vad_sr)
        out_for_vad = _linear_resample_pcm16(bytes(out_recording), sr, vad_sr)

    frame_ms = 30
    bytes_per_vad_frame = vad_sr * frame_ms // 1000 * 2

    def vad_rate(pcm: bytes) -> tuple[float, int, int]:
        # Fresh Vad instance per pass — webrtcvad's smoothing keeps
        # internal state across ``is_speech`` calls, so reusing one
        # instance across two full recordings makes the second pass
        # depend on the first.
        vad = webrtcvad.Vad(args.vad_aggressiveness)
        positive = 0
        total = 0
        for i in range(0, len(pcm) - bytes_per_vad_frame + 1, bytes_per_vad_frame):
            frame = pcm[i : i + bytes_per_vad_frame]
            if vad.is_speech(frame, vad_sr):
                positive += 1
            total += 1
        return (positive / total if total else 0.0, positive, total)

    if args.bypass_aec and bytes(near_recording) != bytes(out_recording):
        print(
            f"[bench] WARN bypass-aec: near≠out (lens {len(near_recording)} vs {len(out_recording)})",
            flush=True,
        )
    near_rate, near_pos, near_total = vad_rate(near_for_vad)
    out_rate, out_pos, out_total = vad_rate(out_for_vad)

    print()
    print(f"webrtcvad (aggressiveness={args.vad_aggressiveness}, sr={vad_sr}, frame={frame_ms}ms):")
    print(f"  near triggered on {near_rate * 100:5.1f}% of frames ({near_pos}/{near_total})")
    print(f"  out  triggered on {out_rate * 100:5.1f}% of frames ({out_pos}/{out_total})")
    if near_rate > 0:
        print(f"  AEC reduced VAD trigger rate by {(1 - out_rate / near_rate) * 100:5.1f}%")
    if out_rate > 0.10:
        print(
            "  ⚠ AEC out still triggers VAD on >10% of frames — server VAD will "
            "almost certainly self-interrupt the model.  Echo path is too loud, "
            "or the AEC's residual is above the VAD's gating threshold."
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rate",
        type=int,
        default=48000,
        help="Sample rate for the audio device + AEC. Match device native rate to skip PortAudio resampler.",
    )
    ap.add_argument("--text", default=None, help="Text to synthesize. Default: pangram run.")
    ap.add_argument("--voice", default="alloy")
    ap.add_argument(
        "--device",
        default="pipewire",
        help=(
            "Duplex device name/index for both mic and speaker. Must support "
            "both directions (or use the default ``pipewire`` virtual device "
            "which bridges to whatever is the user's default in/out)."
        ),
    )
    ap.add_argument(
        "--input-device",
        default=None,
        help="Override input (mic) device — useful for routing through PipeWire's echocancel_source.",
    )
    ap.add_argument(
        "--output-device",
        default=None,
        help="Override output (speaker) device — pair with --input-device for split routing.",
    )
    ap.add_argument(
        "--bypass-aec",
        action="store_true",
        help=(
            "Skip the in-process AEC.  Pair with "
            "``--input-device echocancel_source --output-device echocancel_sink`` "
            "to measure PipeWire's webrtc-aec3 in isolation."
        ),
    )
    ap.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "Audio channels. PipeWire's Echo-Cancel sink/source are stereo; set --channels 2 to drive both correctly."
        ),
    )
    ap.add_argument(
        "--latency-ms",
        type=int,
        default=80,
        help=(
            "Initial mic↔speaker round-trip hint (ms) for AEC3's delay "
            "estimator.  Adapts on its own; the hint just speeds up the "
            "first second of convergence."
        ),
    )
    ap.add_argument(
        "--tail-s",
        type=float,
        default=1.5,
        help="Seconds of mic capture after the TTS finishes (catches trailing echo).",
    )
    ap.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help=(
            "WebRTC VAD aggressiveness 0..3 (higher = stricter / less false "
            "positives, but also more false negatives on quiet speech). "
            "Server-side VAD on OpenAI/Gemini operates at roughly mode 2."
        ),
    )
    args = ap.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
