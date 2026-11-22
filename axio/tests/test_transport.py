"""Tests for transport protocols - structural protocol verification."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from axio.events import StreamEvent
from axio.messages import Message
from axio.tool import Tool
from axio.transport import (
    AudioGenTransport,
    CompletionTransport,
    DummySearchTransport,
    ImageGenTransport,
    SearchTransport,
    STTTransport,
    TTSTransport,
    VideoGenTransport,
)


class _MockCompletion:
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError

    stream.__doc__ = "Mock"


class _MockImageGen:
    async def generate_images(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        return [b""]


class _MockVideoGen:
    async def generate_videos(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        return [b""]


class _MockAudioGen:
    async def generate_audios(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        return [b""]


class _MockTTS:
    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        raise NotImplementedError


class _MockSTT:
    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str:
        return ""


class _MockSearch:
    async def search(self, query: str, *, model: str | None = None) -> str:
        return ""


class TestProtocolConformance:
    def test_completion_transport(self) -> None:
        assert isinstance(_MockCompletion(), CompletionTransport)

    def test_image_gen_transport(self) -> None:
        assert isinstance(_MockImageGen(), ImageGenTransport)

    def test_video_gen_transport(self) -> None:
        assert isinstance(_MockVideoGen(), VideoGenTransport)

    def test_audio_gen_transport(self) -> None:
        assert isinstance(_MockAudioGen(), AudioGenTransport)

    def test_tts_transport(self) -> None:
        assert isinstance(_MockTTS(), TTSTransport)

    def test_stt_transport(self) -> None:
        assert isinstance(_MockSTT(), STTTransport)

    def test_search_transport(self) -> None:
        assert isinstance(_MockSearch(), SearchTransport)

    async def test_dummy_search_transport_fails(self) -> None:
        import pytest

        with pytest.raises(RuntimeError, match="DummyCompletionTransport is a placeholder"):
            await DummySearchTransport().search("test")

    def test_image_and_video_distinct(self) -> None:
        """Image-only mock must not satisfy VideoGen, and vice versa."""
        assert not isinstance(_MockImageGen(), VideoGenTransport)
        assert not isinstance(_MockVideoGen(), ImageGenTransport)
