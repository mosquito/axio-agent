"""Speaker playback: consumes PCM16 chunks and plays them through the default output device."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import sounddevice as sd  # type: ignore[import-untyped]


@dataclass
class Speaker:
    """Async-friendly wrapper around a sounddevice ``RawOutputStream``.

    Holds an internal byte buffer; ``feed()`` appends, the audio callback
    drains as the device asks for samples.  ``stop()`` clears the buffer
    (use it to honour user interruptions — drops everything still queued
    so the assistant goes silent immediately).

    Usage::

        async with Speaker() as spk:
            async for ev in agent.events():
                if isinstance(ev, AudioOutputDelta):
                    await spk.feed(ev.data)
    """

    sample_rate: int = 24000
    device: int | str | None = None
    playback_tap: Callable[[bytes], None] | None = None
    """Optional callback invoked from the audio thread with each chunk that
    is actually being played.  Useful as the far-end reference for an echo
    canceller — the timing here matches what hits the speaker driver, not
    when the application called :meth:`feed`."""

    _stream: sd.RawOutputStream | None = field(default=None, init=False, repr=False)
    _buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        wanted = frames * 2  # int16 mono = 2 bytes per frame
        with self._lock:
            available = len(self._buffer)
            n = min(wanted, available)
            if n:
                outdata[:n] = bytes(self._buffer[:n])
                del self._buffer[:n]
        if n < wanted:
            # Pad with silence so the device never starves.
            outdata[n:wanted] = b"\x00" * (wanted - n)
        # Notify the tap with exactly what hit the device this tick — including
        # any silence padding — so the consumer's clock matches real playback.
        if self.playback_tap is not None:
            self.playback_tap(bytes(outdata[:wanted]))

    async def __aenter__(self) -> Self:
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._callback,
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

    async def feed(self, pcm: bytes) -> None:
        """Append PCM16 mono bytes to the playback buffer."""
        with self._lock:
            self._buffer.extend(pcm)

    async def stop(self) -> None:
        """Drop everything queued for playback (user interrupted)."""
        with self._lock:
            self._buffer.clear()

    def pending_bytes(self) -> int:
        """Bytes still waiting to be played — useful for back-pressure decisions."""
        with self._lock:
            return len(self._buffer)
