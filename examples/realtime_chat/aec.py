"""In-process acoustic echo cancellation backed by webrtc-audio-processing-1.

Wraps the native ``webrtc_apm`` pybind11 extension (built from
``webrtc_apm.cpp``) into a streaming :class:`WebRtcAECProcessor` with
``feed_speaker`` / ``process_mic`` / ``levels_dbfs`` so chat.py can
drive AEC3 without caring about its 10-ms-frame quirks.

Optional ``output_rate`` lets the cleaned mic come back at a different
rate than the device captures at — useful when the model wants 16 kHz
mic audio (Gemini Live) while the device runs at 48 kHz natively
because AEC3 only supports {16, 32, 48} kHz.
"""

from __future__ import annotations

import math
import struct
import threading
from collections import deque

import numpy as np


def _linear_resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler for PCM16 mono.

    Used in two places: the AEC output path (when the caller wants
    cleaned mic at a different rate than capture) and chat.py's
    speaker proxy (resampling the model's 24 kHz audio up to the
    AEC3-forced 48 kHz device rate before playback).  Linear interp
    introduces some HF aliasing but is fine for VAD/ASR-grade audio
    and avoids pulling in a real DSP-resampler dep.
    """
    if src_rate == dst_rate or not pcm:
        return pcm
    n = len(pcm) // 2
    src = struct.unpack(f"<{n}h", pcm)
    out_n = (n * dst_rate) // src_rate
    if out_n == 0:
        return b""
    out: list[int] = []
    ratio = src_rate / dst_rate
    for i in range(out_n):
        pos = i * ratio
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < n:
            v = src[idx] * (1.0 - frac) + src[idx + 1] * frac
        else:
            v = float(src[-1])
        out.append(max(-32768, min(32767, int(v))))
    return struct.pack(f"<{out_n}h", *out)


def _rms_dbfs_pcm16(pcm: bytes) -> float:
    """Fast RMS-in-dBFS for PCM16 mono via numpy.

    Returns ``-120.0`` for empty / digital-silent inputs (avoids
    ``log10(0)`` and undefined-on-some-platforms behaviour).
    """
    if not pcm:
        return -120.0
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return -120.0
    mean_sq = float(np.mean(samples.astype(np.float32) ** 2))
    if mean_sq <= 1.0:
        return -120.0
    return 10.0 * math.log10(mean_sq / (32768.0**2))


def _avg_dbfs(samples: deque[float]) -> float:
    if not samples:
        return -120.0
    return sum(samples) / len(samples)


class WebRtcAECProcessor:
    """Streaming AEC3 wrapper around the native ``webrtc_apm`` extension.

    The underlying webrtc-audio-processing-1 library exposes a fixed
    10 ms frame size; this class buffers caller chunks into full
    frames before handing them to AEC3, then concatenates the cleaned
    output and (optionally) resamples to ``output_rate``.

    AEC3 has its own delay estimator and runs nonlinear residual
    suppression — no ``filter_ms`` / ``latency_ms`` knobs to tune from
    the outside.  We still cap the far buffer (``max_far_lag_ms``) so
    accumulated drift between speaker callbacks and mic chunks doesn't
    push the reference past AEC3's internal search window (~250 ms).
    With sample-aligned duplex audio (one PortAudio callback for both
    directions) drift is zero and the cap never triggers.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        output_rate: int | None = None,
        max_far_lag_ms: int = 250,
        latency_hint_ms: int = 80,
    ) -> None:
        # Lazy-import the native extension so this module stays
        # importable on machines where ``webrtc_apm`` hasn't been
        # built yet — the import error then surfaces only when AEC
        # is actually requested.
        import webrtc_apm  # type: ignore[import-not-found]

        self.sample_rate = sample_rate
        self.output_rate = output_rate or sample_rate
        # webrtc-audio-processing fixes the frame at 10 ms.
        self.frame_size = sample_rate // 100
        self._bytes_per_frame = self.frame_size * 2
        self._apm = webrtc_apm.WebRtcAEC(sample_rate, sample_rate, 1, 1)
        self._apm.set_stream_delay_ms(latency_hint_ms)

        self._mic_pending = bytearray()
        self._far_buffer = bytearray()
        self._max_far_bytes = sample_rate * max_far_lag_ms // 1000 * 2
        self._lock = threading.Lock()

        self._level_window = max(1, 1000 // 10)  # last 1 s of 10 ms frames
        self._level_far: deque[float] = deque(maxlen=self._level_window)
        self._level_near: deque[float] = deque(maxlen=self._level_window)
        self._level_out: deque[float] = deque(maxlen=self._level_window)
        self.far_overflow_bytes: int = 0

    def feed_speaker(self, pcm: bytes) -> None:
        """Stream the playback reference into AEC3 (10 ms frames at a time)."""
        if not pcm:
            return
        with self._lock:
            self._far_buffer.extend(pcm)
            extra = len(self._far_buffer) - self._max_far_bytes
            if extra > 0:
                del self._far_buffer[:extra]
                self.far_overflow_bytes += extra
            while len(self._far_buffer) >= self._bytes_per_frame:
                frame = bytes(self._far_buffer[: self._bytes_per_frame])
                del self._far_buffer[: self._bytes_per_frame]
                self._apm.process_reverse_stream(frame)
                self._level_far.append(_rms_dbfs_pcm16(frame))

    def process_mic(self, pcm: bytes) -> bytes:
        """Run AEC3 against ``pcm``; returns cleaned mic at ``output_rate``."""
        self._mic_pending.extend(pcm)
        out = bytearray()
        while len(self._mic_pending) >= self._bytes_per_frame:
            mic_frame = bytes(self._mic_pending[: self._bytes_per_frame])
            del self._mic_pending[: self._bytes_per_frame]
            cleaned = self._apm.process_stream(mic_frame)
            self._level_near.append(_rms_dbfs_pcm16(mic_frame))
            self._level_out.append(_rms_dbfs_pcm16(cleaned))
            out.extend(cleaned)
        if self.output_rate != self.sample_rate:
            return _linear_resample_pcm16(bytes(out), self.sample_rate, self.output_rate)
        return bytes(out)

    def levels_dbfs(self) -> tuple[float, float, float]:
        """Rolling-1 s ``(far, near, out)`` RMS levels in dBFS."""
        return (
            _avg_dbfs(self._level_far),
            _avg_dbfs(self._level_near),
            _avg_dbfs(self._level_out),
        )
