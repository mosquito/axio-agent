"""Sample-aligned duplex audio: one PortAudio stream, one callback, one clock.

``Microphone`` and ``Speaker`` open independent ``sd.RawInputStream`` and
``sd.RawOutputStream`` instances — each on its own PortAudio host clock.
On most consumer audio stacks (PipeWire/PulseAudio over ALSA) those two
clocks drift by tens of ppm relative to each other, accumulating
~10–25 ms/sec of skew between captured mic samples and played speaker
samples.  Echo cancellers — Speex linear AEC, AEC3, anything — depend on
the relative timing of the far-end reference (speaker output) and the
near-end signal (mic input).  Drift past the algorithm's filter / search
window destroys cancellation quality after only a few minutes of
continuous use.

``DuplexAudio`` opens **one** ``sd.RawStream`` with both directions
hooked into a single callback.  Mic and speaker share the same
PortAudio clock; relative timing is sample-exact for the lifetime of
the stream.  This is the same architecture PipeWire's
``module-echo-cancel`` uses internally and is what lets in-process
AEC3 reach production-grade suppression without an external graph.

Usage mirrors the ``Microphone`` / ``Speaker`` pair so callers can
swap with minimal change::

    async with DuplexAudio(sample_rate=48000) as duplex:
        async def consume_mic() -> None:
            async for chunk in duplex.mic_chunks():
                await agent.send(chunk)

        async def play_audio_output() -> None:
            async for ev in agent.events():
                if isinstance(ev, AudioOutputDelta):
                    await duplex.feed_speaker(ev.data)
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from axio.blocks import AudioBlock


@dataclass
class DuplexAudio:
    """One sd.RawStream, both directions, sample-aligned by PortAudio.

    The callback runs on PortAudio's audio thread for every block of
    ``chunk_ms`` worth of frames; it (a) pulls the next chunk of
    speaker bytes out of an internal buffer into ``outdata`` and (b)
    forwards ``indata`` to a thread-safe queue that the asyncio side
    drains via :meth:`mic_chunks`.

    ``channels`` is how many channels we open the **device** with.  On
    a stereo-only device (e.g. PipeWire's ``Echo-Cancel Sink/Source``,
    or most laptop default outputs) you must set ``channels=2`` to
    drive both speakers correctly.  The ``feed_speaker`` /
    ``mic_chunks`` API itself is mono — ``feed_speaker`` upmixes by
    duplicating samples across channels and ``mic_chunks`` downmixes by
    averaging.  Set ``mono_io=False`` to skip the conversions and
    pass interleaved PCM through unchanged.
    """

    sample_rate: int = 48000
    chunk_ms: int = 20
    channels: int = 1
    device: int | str | tuple[int | str | None, int | str | None] | None = None
    mono_io: bool = True
    queue_maxsize: int = 100
    playback_tap: Callable[[bytes], None] | None = None
    """Optional sync callback fired from the audio thread with the *mono*
    bytes we just handed to the speaker (silence-padded when the buffer
    underruns).  Useful as a far-end reference for an external AEC, or
    for level meters that need real playback timing rather than enqueue
    timing.  Receives the same bytes the mono ``feed_speaker`` API
    accepts, so it's symmetric with ``mic_chunks``."""

    _stream: sd.RawStream | None = field(default=None, init=False, repr=False)
    _mic_queue: asyncio.Queue[bytes] | None = field(default=None, init=False, repr=False)
    _spk_buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        self._mic_queue = asyncio.Queue(maxsize=self.queue_maxsize)
        loop = self._loop
        queue = self._mic_queue
        ch = self.channels
        bytes_per_device_sample = 2 * ch

        def callback(
            indata: Any,
            outdata: Any,
            frames: int,
            time_info: Any,
            status: sd.CallbackFlags,
        ) -> None:
            wanted = frames * bytes_per_device_sample
            with self._lock:
                # ``_spk_buffer`` is laid out for the device (multi-channel
                # if ch > 1, see ``feed_speaker``).  Copy what we have, pad
                # with silence on underflow so the device never starves.
                available = len(self._spk_buffer)
                n = min(wanted, available)
                if n:
                    outdata[:n] = bytes(self._spk_buffer[:n])
                    del self._spk_buffer[:n]
            if n < wanted:
                outdata[n:wanted] = b"\x00" * (wanted - n)

            mono_far: bytes
            mono_near: bytes
            if self.mono_io and ch > 1:
                mono_far = (
                    np.frombuffer(bytes(outdata[:wanted]), dtype="<i2")
                    .reshape(-1, ch)
                    .mean(axis=1)
                    .astype("<i2")
                    .tobytes()
                )
                mono_near = (
                    np.frombuffer(bytes(indata), dtype="<i2").reshape(-1, ch).mean(axis=1).astype("<i2").tobytes()
                )
            else:
                mono_far = bytes(outdata[:wanted])
                mono_near = bytes(indata)

            # Tap fires before the mic-queue push so the consumer can
            # use it as a perfectly time-aligned far reference.  Cheap
            # ops only — this is the audio thread.
            if self.playback_tap is not None:
                try:
                    self.playback_tap(mono_far)
                except Exception:
                    # Don't propagate — a buggy tap shouldn't kill audio.
                    pass

            loop.call_soon_threadsafe(_put_or_drop, queue, mono_near)

        chunk_frames = max(1, self.sample_rate * self.chunk_ms // 1000)
        self._stream = sd.RawStream(
            samplerate=self.sample_rate,
            channels=(ch, ch),
            dtype="int16",
            blocksize=chunk_frames,
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._mic_queue is not None:
            # Same evict-then-sentinel dance Microphone uses: don't deadlock
            # close on a slow consumer.
            try:
                self._mic_queue.put_nowait(b"")
            except asyncio.QueueFull:
                self._mic_queue.get_nowait()
                self._mic_queue.put_nowait(b"")

    # ── mic side ─────────────────────────────────────────────────────

    def mic_chunks(self) -> AsyncIterator[AudioBlock]:
        """Async-iterate captured mic data as :class:`AudioBlock` chunks.

        Chunk size is whatever PortAudio hands the callback (typically
        ``chunk_ms`` worth of frames).  Yields mono PCM16 when
        ``mono_io`` is True, otherwise interleaved as opened.
        """
        return _MicChunks(self)

    # ── speaker side ─────────────────────────────────────────────────

    async def feed_speaker(self, pcm: bytes) -> None:
        """Append PCM16 to the speaker buffer.

        Input is mono when ``mono_io`` is True (samples are duplicated
        across channels for the device), otherwise must be in the
        device's exact channel layout.
        """
        if not pcm:
            return
        if self.mono_io and self.channels > 1:
            mono = np.frombuffer(pcm, dtype="<i2")
            interleaved = np.repeat(mono, self.channels).tobytes()
        else:
            interleaved = pcm
        with self._lock:
            self._spk_buffer.extend(interleaved)

    async def stop_speaker(self) -> None:
        """Drop everything queued for playback (use on user interruption)."""
        with self._lock:
            self._spk_buffer.clear()

    def speaker_pending_bytes(self) -> int:
        """Mono-equivalent bytes still waiting to be played."""
        with self._lock:
            n = len(self._spk_buffer)
        if self.mono_io and self.channels > 1:
            return n // self.channels
        return n

    # ── Microphone / Speaker compatibility views ─────────────────────
    # Existing realtime examples are built around the ``Microphone`` and
    # ``Speaker`` classes; ``duplex.mic`` / ``duplex.speaker`` expose the
    # same surface (async-iter on the mic, ``feed`` / ``stop`` /
    # ``pending_bytes`` on the speaker) so callers can swap to duplex
    # by replacing two ``async with`` arms with one.

    @property
    def mic(self) -> _MicView:
        return _MicView(self)

    @property
    def speaker(self) -> _SpeakerView:
        return _SpeakerView(self)


class _MicView:
    """Quacks like :class:`Microphone` — async-iterable of AudioBlocks."""

    __slots__ = ("_duplex",)

    def __init__(self, duplex: DuplexAudio) -> None:
        self._duplex = duplex

    def __aiter__(self) -> _MicChunks:
        return _MicChunks(self._duplex)


class _SpeakerView:
    """Quacks like :class:`Speaker` — feed/stop/pending_bytes."""

    __slots__ = ("_duplex",)

    def __init__(self, duplex: DuplexAudio) -> None:
        self._duplex = duplex

    async def feed(self, pcm: bytes) -> None:
        await self._duplex.feed_speaker(pcm)

    async def stop(self) -> None:
        await self._duplex.stop_speaker()

    def pending_bytes(self) -> int:
        return self._duplex.speaker_pending_bytes()


class _MicChunks:
    """Lightweight iterator wrapper so callers don't accidentally exhaust
    the duplex object itself."""

    __slots__ = ("_duplex",)

    def __init__(self, duplex: DuplexAudio) -> None:
        self._duplex = duplex

    def __aiter__(self) -> _MicChunks:
        return self

    async def __anext__(self) -> AudioBlock:
        if self._duplex._mic_queue is None:
            raise RuntimeError("DuplexAudio not started — use 'async with DuplexAudio() as d:'.")
        chunk = await self._duplex._mic_queue.get()
        if not chunk:
            raise StopAsyncIteration
        return AudioBlock(media_type="audio/pcm", data=chunk)


def _put_or_drop(queue: asyncio.Queue[bytes], data: bytes) -> None:
    try:
        queue.put_nowait(data)
    except asyncio.QueueFull:
        pass
