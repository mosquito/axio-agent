"""Standalone API-key speech transports; never use ChatGPT subscription credentials."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import aiohttp
from axio.exceptions import StreamError
from axio.transport import STTTransport, TTSTransport


@dataclass(slots=True, kw_only=True)
class _AudioAPI:
    env_vars: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""), repr=False)
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    timeout_seconds: float = 120.0

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[aiohttp.ClientSession]:
        if not self.api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for paid API audio; ChatGPT subscriptions are not API keys")
        if self.session is not None:
            if self.session.closed:
                raise RuntimeError("The supplied HTTP session is closed")
            yield self.session
        else:
            async with aiohttp.ClientSession() as client:
                yield client

    @staticmethod
    def _check(response: aiohttp.ClientResponse) -> None:
        # Do not echo provider bodies: they may contain user audio/text or credentials.
        if response.status != 200:
            raise StreamError(f"OpenAI audio HTTP {response.status}")


_INPUT_TYPES = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mpga": "mpga",
    "audio/m4a": "m4a",
    "audio/mp4": "m4a",
    "video/mp4": "mp4",
    "audio/webm": "webm",
    "video/webm": "webm",
}


@dataclass(slots=True, kw_only=True)
class OpenAISTTTransport(_AudioAPI, STTTransport):
    """Transcribe an encoded audio file via ``/audio/transcriptions`` (not raw PCM)."""

    model: str = "gpt-4o-mini-transcribe"
    language: str | None = None
    prompt: str | None = None

    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str:
        mime = media_type.partition(";")[0].strip().lower()
        if mime not in _INPUT_TYPES:
            raise ValueError(f"Unsupported transcription media type: {media_type!r}; wrap raw PCM in WAV first")
        if not audio or len(audio) > 25_000_000:
            raise ValueError("Audio must be nonempty and at most 25 MB; split longer files explicitly")
        form = aiohttp.FormData()
        form.add_field("model", self.model)
        form.add_field("response_format", "json")
        form.add_field("file", audio, filename=f"audio.{_INPUT_TYPES[mime]}", content_type=mime)
        if self.language is not None:
            form.add_field("language", self.language)
        if self.prompt is not None:
            form.add_field("prompt", self.prompt)
        try:
            async with self._client() as client:
                async with client.post(
                    f"{self.base_url.rstrip('/')}/audio/transcriptions",
                    data=form,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    self._check(response)
                    try:
                        result = await response.json()
                    except ValueError as exc:
                        raise StreamError("OpenAI transcription returned invalid JSON") from exc
                    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                        raise StreamError("OpenAI transcription response is missing text")
                    return str(result["text"])
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise StreamError("OpenAI transcription request failed") from exc


@dataclass(slots=True, kw_only=True)
class OpenAITTSTransport(_AudioAPI, TTSTransport):
    """Stream binary audio via ``/audio/speech``. No replay/retry of paid requests.

    PCM output is signed 16-bit little-endian mono at 24 kHz without a header.
    Network chunks are arbitrary byte boundaries, not PCM frames or standalone files.
    Close an abandoned generator with ``contextlib.aclosing`` to release its response.
    """

    model: str = "gpt-4o-mini-tts"
    voice: str = "coral"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = 1.0
    instructions: str | None = None

    async def synthesize(self, text: str, *, voice: str | None = None) -> AsyncGenerator[bytes]:
        if not text.strip():
            raise ValueError("Speech input must not be empty")
        if not 0.25 <= self.speed <= 4.0:
            raise ValueError("Speech speed must be between 0.25 and 4.0")
        if self.response_format not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            raise ValueError("Unsupported speech response format")
        payload = dict(
            model=self.model,
            input=text,
            voice=voice or self.voice,
            response_format=self.response_format,
            speed=self.speed,
        )
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        try:
            async with self._client() as client:
                async with client.post(
                    f"{self.base_url.rstrip('/')}/audio/speech",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    self._check(response)
                    async for chunk in response.content.iter_chunked(65536):
                        if chunk:
                            yield chunk
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise StreamError("OpenAI speech request failed") from exc
