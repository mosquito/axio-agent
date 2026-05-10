"""Tests for Microphone helpers that don't require an actual audio device."""

from __future__ import annotations

import asyncio

import pytest

from axio_audio.microphone import Microphone, _put_or_drop


async def test_put_or_drop_pushes_when_space_available() -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)
    _put_or_drop(queue, b"a")
    _put_or_drop(queue, b"b")
    assert queue.qsize() == 2
    assert await queue.get() == b"a"
    assert await queue.get() == b"b"


async def test_put_or_drop_silently_drops_when_full() -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    _put_or_drop(queue, b"a")
    _put_or_drop(queue, b"dropped")  # full — must not raise
    assert queue.qsize() == 1
    assert await queue.get() == b"a"


async def test_close_does_not_hang_when_queue_full_and_no_consumer() -> None:
    """Regression: ``__aexit__`` used to ``await put(sentinel)`` which deadlocks
    if the consumer has already stopped draining and the queue is at capacity.
    Closing must complete promptly and leave the sentinel in the queue."""
    mic = Microphone(queue_maxsize=2)
    mic._queue = asyncio.Queue(maxsize=2)
    mic._queue.put_nowait(b"chunk1")
    mic._queue.put_nowait(b"chunk2")  # queue is full, no one is draining

    await asyncio.wait_for(mic.__aexit__(None, None, None), timeout=1.0)

    drained: list[bytes] = []
    while not mic._queue.empty():
        drained.append(mic._queue.get_nowait())
    assert b"" in drained, "sentinel must be present after close"


async def test_anext_raises_before_aenter() -> None:
    mic = Microphone()
    with pytest.raises(RuntimeError):
        await mic.__anext__()
