"""Microphone capture: yields :class:`AudioBlock` chunks from the system microphone."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import sounddevice as sd  # type: ignore[import-untyped]
from axio.blocks import AudioBlock


@dataclass
class Microphone:
    """Async-iterable wrapper around a sounddevice ``RawInputStream``.

    Captures PCM16 mono audio at ``sample_rate`` and yields
    :class:`AudioBlock` chunks of approximately ``chunk_ms`` milliseconds
    each.  The defaults match the OpenAI Realtime API (24 kHz PCM16).

    Usage::

        async with Microphone() as mic:
            async for chunk in mic:
                await agent.send(chunk)
    """

    sample_rate: int = 24000
    chunk_ms: int = 50
    device: int | str | None = None
    queue_maxsize: int = 100  # cap to avoid unbounded growth on slow consumers
    _stream: sd.RawInputStream | None = field(default=None, init=False, repr=False)
    _queue: asyncio.Queue[bytes] | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)
        chunk_frames = max(1, self.sample_rate * self.chunk_ms // 1000)
        loop = self._loop
        queue = self._queue

        def callback(indata: Any, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
            data = bytes(indata)
            loop.call_soon_threadsafe(_put_or_drop, queue, data)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
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
        if self._queue is not None:
            # Stream is now stopped, so the callback won't push more.  Push the
            # sentinel non-blockingly and evict the oldest chunk if the queue
            # is full — a blocking ``put`` here would deadlock close when the
            # consumer has already stopped draining.
            try:
                self._queue.put_nowait(b"")
            except asyncio.QueueFull:
                self._queue.get_nowait()
                self._queue.put_nowait(b"")

    def __aiter__(self) -> AsyncIterator[AudioBlock]:
        return self

    async def __anext__(self) -> AudioBlock:
        if self._queue is None:
            raise RuntimeError("Microphone not started — use 'async with Microphone() as mic:'.")
        chunk = await self._queue.get()
        if not chunk:
            raise StopAsyncIteration
        return AudioBlock(media_type="audio/pcm", data=chunk)


def _put_or_drop(queue: asyncio.Queue[bytes], data: bytes) -> None:
    """Push ``data`` into ``queue``; silently drop if full to keep the audio
    callback non-blocking under back-pressure."""
    try:
        queue.put_nowait(data)
    except asyncio.QueueFull:
        pass
