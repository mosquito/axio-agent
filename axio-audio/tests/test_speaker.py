"""Tests for Speaker buffer semantics — exercise the callback directly without
actually opening an audio device."""

from __future__ import annotations

import pytest

from axio_audio.speaker import Speaker


@pytest.fixture
def speaker() -> Speaker:
    return Speaker()


def test_callback_drains_buffer(speaker: Speaker) -> None:
    pcm = bytes(range(20))  # 10 int16 frames
    speaker._buffer.extend(pcm)
    out = bytearray(20)
    speaker._callback(out, frames=10, time_info=None, status=0)
    assert bytes(out) == pcm
    assert len(speaker._buffer) == 0


def test_callback_pads_with_silence_when_underflowing(speaker: Speaker) -> None:
    speaker._buffer.extend(b"\x01\x02\x03\x04")  # 2 frames of audio
    out = bytearray(20)  # caller wants 10 frames = 20 bytes
    speaker._callback(out, frames=10, time_info=None, status=0)
    assert bytes(out[:4]) == b"\x01\x02\x03\x04"
    assert bytes(out[4:]) == b"\x00" * 16


def test_callback_only_consumes_requested_frames(speaker: Speaker) -> None:
    speaker._buffer.extend(bytes(range(40)))
    out = bytearray(10)
    speaker._callback(out, frames=5, time_info=None, status=0)
    assert bytes(out) == bytes(range(10))
    assert len(speaker._buffer) == 30  # remainder still queued


async def test_feed_appends_bytes(speaker: Speaker) -> None:
    await speaker.feed(b"\xaa\xbb")
    await speaker.feed(b"\xcc\xdd")
    assert bytes(speaker._buffer) == b"\xaa\xbb\xcc\xdd"


async def test_stop_clears_buffer(speaker: Speaker) -> None:
    await speaker.feed(b"\xff" * 100)
    assert speaker.pending_bytes() == 100
    await speaker.stop()
    assert speaker.pending_bytes() == 0
