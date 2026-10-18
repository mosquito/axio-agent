"""No-network tests for binary speech and multipart transcription contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from axio.exceptions import StreamError
from axio.transport import STTTransport, TTSTransport

from axio_transport_openai import OpenAISTTTransport, OpenAITTSTransport


class Response:
    def __init__(self, status: int = 200, result: Any = None) -> None:
        self.status = status
        self.result = {"text": "hello"} if result is None else result
        self.content = self
        self.exited = False
        self.block = False

    async def __aenter__(self) -> Response:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited = True

    async def json(self) -> Any:
        return self.result

    async def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        assert size > 0
        yield b"one"
        if self.block:
            await asyncio.Event().wait()
        yield b"two"


class Client:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.closed = True

    def post(self, url: str, **kwargs: Any) -> Response:
        self.calls.append((url, kwargs))
        return self.response


async def test_stt_multipart_and_protocol() -> None:
    client = Client(Response())
    transport = OpenAISTTTransport(api_key="api-only", session=client, language="en", prompt="names")  # type: ignore[arg-type]
    assert isinstance(transport, STTTransport)
    assert await transport.transcribe(b"RIFFdata") == "hello"
    url, request = client.calls[0]
    assert url == "https://api.openai.com/v1/audio/transcriptions"
    assert request["headers"] == {"Authorization": "Bearer api-only"}
    fields = {field[0]["name"]: field for field in request["data"]._fields}
    assert fields["file"][0]["filename"] == "audio.wav"
    assert fields["file"][2] == b"RIFFdata"
    assert fields["response_format"][2] == "json"
    assert fields["language"][2] == "en"
    assert fields["prompt"][2] == "names"
    assert client.response.exited and not client.closed


async def test_tts_binary_and_override() -> None:
    client = Client(Response())
    transport = OpenAITTSTransport(api_key="api-only", session=client, response_format="pcm", instructions="whisper")  # type: ignore[arg-type]
    assert isinstance(transport, TTSTransport)
    assert [part async for part in transport.synthesize("hello", voice="marin")] == [b"one", b"two"]
    url, request = client.calls[0]
    assert url.endswith("/audio/speech")
    assert request["json"] == {
        "model": "gpt-4o-mini-tts",
        "input": "hello",
        "voice": "marin",
        "response_format": "pcm",
        "speed": 1.0,
        "instructions": "whisper",
    }
    assert client.response.exited and not client.closed


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
async def test_http_errors_never_retry(status: int) -> None:
    for cls in (OpenAISTTTransport, OpenAITTSTransport):
        client = Client(Response(status))
        transport = cls(api_key="test", session=client)  # type: ignore[arg-type]
        with pytest.raises(StreamError, match=str(status)):
            if isinstance(transport, OpenAISTTTransport):
                await transport.transcribe(b"audio")
            else:
                _ = [x async for x in transport.synthesize("hello")]
        assert len(client.calls) == 1 and client.response.exited


@pytest.mark.parametrize("result", [{}, {"text": 12}, []])
async def test_bad_stt_response(result: Any) -> None:
    client = Client(Response(result=result))
    with pytest.raises(StreamError, match="missing text"):
        await OpenAISTTTransport(api_key="test", session=client).transcribe(b"audio")  # type: ignore[arg-type]
    assert client.response.exited


@pytest.mark.parametrize(
    "audio,mime", [(b"", "audio/wav"), (b"a", "audio/pcm;rate=16000"), (b"a" * 25_000_001, "audio/wav")]
)
async def test_invalid_stt_input(audio: bytes, mime: str) -> None:
    with pytest.raises(ValueError):
        await OpenAISTTTransport(api_key="test").transcribe(audio, mime)


async def test_early_close_releases_response_and_owned_client() -> None:
    client = Client(Response())
    with patch("axio_transport_openai.audio.aiohttp.ClientSession", return_value=client):
        async with aclosing(OpenAITTSTransport(api_key="test").synthesize("hello")) as stream:
            assert await anext(stream) == b"one"
    assert client.response.exited and client.closed


async def test_cancellation_releases_response_and_preserves_shared_client() -> None:
    response = Response()
    response.block = True
    client = Client(response)
    stream = OpenAITTSTransport(api_key="test", session=client).synthesize("hello")  # type: ignore[arg-type]
    assert await anext(stream) == b"one"
    task = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.exited and not client.closed


async def test_credentials_and_local_validation() -> None:
    with pytest.raises(ValueError, match="API_KEY"):
        await OpenAISTTTransport(api_key="").transcribe(b"audio")
    for text, speed in [("", 1.0), ("hello", 9.0)]:
        with pytest.raises(ValueError):
            _ = [x async for x in OpenAITTSTransport(api_key="test", speed=speed).synthesize(text)]
    client = Client(Response())
    client.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        await OpenAISTTTransport(api_key="test", session=client).transcribe(b"audio")  # type: ignore[arg-type]


async def test_connection_error_wrapped_without_retry() -> None:
    client = Client(Response())
    with patch.object(client, "post", side_effect=aiohttp.ClientConnectionError("lost")):
        with pytest.raises(StreamError, match="request failed"):
            await OpenAISTTTransport(api_key="test", session=client).transcribe(b"audio")  # type: ignore[arg-type]


async def test_registration_is_capability_specific() -> None:
    from importlib.metadata import entry_points

    assert next(iter(entry_points(group="axio.transport.stt", name="openai"))).load() is OpenAISTTTransport
    assert next(iter(entry_points(group="axio.transport.tts", name="openai"))).load() is OpenAITTSTransport
    assert all(
        ep.load() not in (OpenAISTTTransport, OpenAITTSTransport) for ep in entry_points(group="axio.transport")
    )


async def test_stt_invalid_json_cleanup() -> None:
    response = Response()
    client = Client(response)
    with patch.object(response, "json", side_effect=ValueError("bad json")):
        with pytest.raises(StreamError, match="invalid JSON"):
            await OpenAISTTTransport(api_key="test", session=client).transcribe(b"audio")  # type: ignore[arg-type]
    assert response.exited


async def test_speech_over_http() -> None:
    async def transcribe(request: web.Request) -> web.Response:
        assert request.headers["Authorization"] == "Bearer test"
        form = await request.post()
        upload = form["file"]
        assert isinstance(upload, web.FileField)
        assert upload.filename == "audio.wav" and upload.content_type == "audio/wav"
        assert upload.file.read() == b"RIFFdata"
        assert form["model"] == "gpt-4o-mini-transcribe"
        assert form["response_format"] == "json"
        return web.json_response({"text": "hello"})

    async def synthesize(request: web.Request) -> web.StreamResponse:
        assert request.headers["Authorization"] == "Bearer test"
        assert (await request.json())["input"] == "hello"
        response = web.StreamResponse(headers={"Content-Type": "audio/mpeg"})
        response.enable_chunked_encoding()
        await response.prepare(request)
        await response.write(b"first")
        await response.write(b"second")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    app.router.add_post("/v1/audio/speech", synthesize)
    async with TestServer(app) as server, aiohttp.ClientSession() as client:
        base_url = str(server.make_url("/v1/"))
        stt = OpenAISTTTransport(api_key="test", base_url=base_url, session=client)
        tts = OpenAITTSTransport(api_key="test", base_url=base_url, session=client)
        transcript = await stt.transcribe(b"RIFFdata")
        assert b"".join([chunk async for chunk in tts.synthesize(transcript)]) == b"firstsecond"
        assert not client.closed
