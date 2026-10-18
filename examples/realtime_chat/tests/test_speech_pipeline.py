"""Composition requires only the text protocol; never couples account tokens to speech."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock

import aiohttp
import pytest
from axio.testing import StubTransport, make_text_response
from axio_transport_openai import OpenAISTTTransport, OpenAITTSTransport

from speech_pipeline import respond


async def test_explicit_text_and_audio_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "separate-api-key")
    called: list[str] = []
    transcribe = AsyncMock(return_value="question")
    monkeypatch.setattr(OpenAISTTTransport, "transcribe", transcribe)

    async def synthesize(self: OpenAITTSTransport, text: str, *, voice: str | None = None) -> AsyncGenerator[bytes]:
        assert self.api_key == "separate-api-key"
        called.append(text)
        yield b"speech"

    monkeypatch.setattr(OpenAITTSTransport, "synthesize", synthesize)
    # Any subscription text transport fits the same protocol. No subscription
    # credentials or audio endpoints are requested by the composition layer.
    text = StubTransport([make_text_response("answer")])
    output = await respond(b"WAV", text, cast(aiohttp.ClientSession, object()))
    assert output == b"speech" and called == ["answer"]
    transcribe.assert_awaited_once_with(b"WAV")
