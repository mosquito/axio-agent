"""Tests for the Gemini Live WebSocket realtime transport."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import aiohttp
import pytest
from axio.blocks import AudioBlock, TextBlock
from axio.events import (
    AudioOutputDelta,
    SpeechStarted,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from axio.tool import Tool
from axio.types import StopReason

from axio_transport_google.realtime import (
    GeminiLiveSession,
    GeminiLiveTransport,
    _convert_realtime_tools,
)


class _StubWSMessage:
    def __init__(self, mtype: aiohttp.WSMsgType, data: str = "") -> None:
        self.type = mtype
        self.data = data


class _StubWebSocket:
    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self.sent: list[str] = []
        self._scripted = scripted
        self.closed = False
        self._extra: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    def push(self, ev: dict[str, Any] | None) -> None:
        self._extra.put_nowait(ev)

    def __aiter__(self) -> AsyncIterator[_StubWSMessage]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[_StubWSMessage]:
        for ev in self._scripted:
            yield _StubWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(ev))
            await asyncio.sleep(0)
        while True:
            next_event = await self._extra.get()
            if next_event is None:
                return
            yield _StubWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(next_event))

    async def close(self) -> None:
        self.closed = True

    def exception(self) -> BaseException | None:
        return None


def _decode(ws: _StubWebSocket) -> list[dict[str, Any]]:
    return [json.loads(s) for s in ws.sent]


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------


class TestOutbound:
    async def test_audio_block_sends_realtime_input(self) -> None:
        ws = _StubWebSocket([])
        s = GeminiLiveSession(ws=ws)
        await s.send(AudioBlock(media_type="audio/pcm", data=b"\x01\x02"))
        sent = _decode(ws)[0]
        assert sent == {
            "realtimeInput": {
                "audio": {
                    "data": base64.b64encode(b"\x01\x02").decode(),
                    "mimeType": "audio/pcm",
                }
            }
        }

    async def test_text_block_sends_client_content_with_turn_complete(self) -> None:
        ws = _StubWebSocket([])
        s = GeminiLiveSession(ws=ws)
        await s.send(TextBlock(text="hi"))
        sent = _decode(ws)[0]
        assert sent == {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "hi"}]}],
                "turnComplete": True,
            }
        }

    async def test_commit_sends_audio_stream_end(self) -> None:
        ws = _StubWebSocket([])
        s = GeminiLiveSession(ws=ws)
        await s.commit()
        assert _decode(ws) == [{"realtimeInput": {"audioStreamEnd": True}}]

    async def test_interrupt_sends_activity_end(self) -> None:
        ws = _StubWebSocket([])
        s = GeminiLiveSession(ws=ws)
        await s.interrupt()
        assert _decode(ws) == [{"realtimeInput": {"activityEnd": {}}}]

    async def test_send_tool_result(self) -> None:
        ws = _StubWebSocket([])
        s = GeminiLiveSession(ws=ws)
        await s.send_tool_result("tc-1", "add", "42")
        assert _decode(ws) == [
            {"toolResponse": {"functionResponses": [{"id": "tc-1", "name": "add", "response": {"output": "42"}}]}}
        ]


# ---------------------------------------------------------------------------
# Inbound translation
# ---------------------------------------------------------------------------


class TestInbound:
    async def _drain(self, scripted: list[dict[str, Any]]) -> tuple[_StubWebSocket, list[Any]]:
        ws = _StubWebSocket(scripted)
        ws.push(None)
        s = GeminiLiveSession(ws=ws)
        events = [ev async for ev in s.events()]
        return ws, events

    async def test_inline_audio_part_yields_audio_output_delta(self) -> None:
        pcm = b"\xaa\xbb"
        _, evs = await self._drain(
            [
                {
                    "serverContent": {
                        "modelTurn": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/pcm;rate=24000",
                                        "data": base64.b64encode(pcm).decode(),
                                    }
                                }
                            ]
                        }
                    }
                }
            ]
        )
        assert evs == [AudioOutputDelta(data=pcm, media_type="audio/pcm;rate=24000")]

    async def test_text_part_yields_text_delta(self) -> None:
        _, evs = await self._drain([{"serverContent": {"modelTurn": {"parts": [{"text": "hello"}]}}}])
        assert evs == [TextDelta(index=0, delta="hello")]

    async def test_input_transcription(self) -> None:
        _, evs = await self._drain([{"serverContent": {"inputTranscription": {"text": "user said"}}}])
        assert evs == [TranscriptDelta(role="user", delta="user said")]

    async def test_output_transcription(self) -> None:
        _, evs = await self._drain([{"serverContent": {"outputTranscription": {"text": "model said"}}}])
        assert evs == [TranscriptDelta(role="assistant", delta="model said")]

    async def test_interrupted_emits_speech_started(self) -> None:
        _, evs = await self._drain([{"serverContent": {"interrupted": True}}])
        assert evs == [SpeechStarted()]

    async def test_turn_complete_without_tools_is_end_turn(self) -> None:
        _, evs = await self._drain(
            [
                {
                    "serverContent": {"turnComplete": True},
                    "usageMetadata": {"promptTokenCount": 10, "responseTokenCount": 4},
                }
            ]
        )
        assert len(evs) == 1
        ev = evs[0]
        assert isinstance(ev, TurnComplete)
        assert ev.stop_reason is StopReason.end_turn
        assert ev.usage is not None
        assert ev.usage.input_tokens == 10
        assert ev.usage.output_tokens == 4

    async def test_tool_call_then_turn_complete_yields_tool_use_stop(self) -> None:
        _, evs = await self._drain(
            [
                {"toolCall": {"functionCalls": [{"id": "fc-1", "name": "add", "args": {"a": 1, "b": 2}}]}},
                {"serverContent": {"turnComplete": True}},
            ]
        )
        assert evs[0] == ToolUseStart(index=0, tool_use_id="fc-1", name="add")
        assert evs[1] == ToolInputDelta(index=0, tool_use_id="fc-1", partial_json='{"a": 1, "b": 2}')
        assert isinstance(evs[2], TurnComplete)
        assert evs[2].stop_reason is StopReason.tool_use

    async def test_setup_complete_is_ignored(self) -> None:
        _, evs = await self._drain([{"setupComplete": {}}])
        assert evs == []


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


async def test_convert_tools_wraps_in_function_declarations() -> None:
    async def add(a: int, b: int) -> str:
        return str(a + b)

    out = _convert_realtime_tools([Tool(name="add", handler=add, description="adds")])
    assert len(out) == 1
    decls = out[0]["functionDeclarations"]
    assert decls[0]["name"] == "add"
    assert decls[0]["description"] == "adds"
    assert "parameters" in decls[0]


async def test_convert_tools_empty_returns_empty_list() -> None:
    assert _convert_realtime_tools([]) == []


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class _StubHttpSession:
    def __init__(self, ws: _StubWebSocket) -> None:
        self._ws = ws
        self.connect_url: str | None = None
        self.connect_headers: dict[str, str] | None = None
        self.closed = False

    async def ws_connect(self, url: str, *, headers: dict[str, str] | None = None) -> _StubWebSocket:
        self.connect_url = url
        self.connect_headers = headers
        return self._ws

    async def close(self) -> None:
        self.closed = True


async def test_connect_sends_setup_with_system_tools_voice() -> None:
    async def add(a: int, b: int) -> str:
        return str(a + b)

    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)
    transport = GeminiLiveTransport(api_key="key-x", http_session=cast(Any, http), vertexai=False)
    session = await transport.connect(
        system="be brief",
        tools=[Tool(name="add", handler=add)],
        voice="Aoede",
    )
    assert isinstance(session, GeminiLiveSession)
    assert http.connect_url is not None and http.connect_url.endswith("?key=key-x")
    sent = _decode(ws)
    assert len(sent) == 1
    setup = sent[0]["setup"]
    assert setup["model"].startswith("models/")
    assert setup["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Aoede"
    assert setup["tools"][0]["functionDeclarations"][0]["name"] == "add"


async def test_connect_omits_tools_when_empty() -> None:
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)
    transport = GeminiLiveTransport(api_key="key-y", http_session=cast(Any, http), vertexai=False)
    await transport.connect(system="hi", tools=[])
    setup = json.loads(ws.sent[0])["setup"]
    assert "tools" not in setup


async def test_connect_passes_language_code() -> None:
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)
    transport = GeminiLiveTransport(
        api_key="k",
        http_session=cast(Any, http),
        vertexai=False,
        language_code="ru-RU",
    )
    await transport.connect(system="hi", tools=[], voice="Aoede")
    setup = json.loads(ws.sent[0])["setup"]
    assert setup["generationConfig"]["speechConfig"]["languageCode"] == "ru-RU"
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Aoede"


def test_nearest_vertex_live_region_passes_through_supported() -> None:
    from axio_transport_google.realtime import (
        SUPPORTED_VERTEX_LIVE_REGIONS,
        _nearest_vertex_live_region,
    )

    for region in SUPPORTED_VERTEX_LIVE_REGIONS:
        assert _nearest_vertex_live_region(region) == region


def test_nearest_vertex_live_region_picks_from_geo_prefix() -> None:
    from axio_transport_google.realtime import (
        SUPPORTED_VERTEX_LIVE_REGIONS,
        _nearest_vertex_live_region,
    )

    # European requests stay in Europe.
    assert _nearest_vertex_live_region("europe-west1") == "europe-west1"  # supported, pass-through
    nearest_eu = _nearest_vertex_live_region("europe-west2")
    assert nearest_eu in SUPPORTED_VERTEX_LIVE_REGIONS
    assert nearest_eu.startswith("europe-")

    # US-prefixed but unsupported region stays in the US.
    assert _nearest_vertex_live_region("us-west2") == "us-central1"

    # Africa / Middle East lean European (closer than US).
    assert _nearest_vertex_live_region("africa-south1") == "europe-west4"
    assert _nearest_vertex_live_region("me-west1") == "europe-west4"

    # Asia / Australia / global fall through to us-central1.
    for loc in ("asia-east1", "australia-southeast1", "global"):
        assert _nearest_vertex_live_region(loc) == "us-central1"


async def test_vertex_connect_coerces_global_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vertex Live is region-specific; ``location="global"`` must be coerced
    to a supported region so users with ``GOOGLE_CLOUD_LOCATION=global``
    don't get a 404 handshake."""
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)
    transport = GeminiLiveTransport(
        http_session=cast(Any, http),
        vertexai=True,
        project="p",
        location="global",
    )

    async def fake_token(self: GeminiLiveTransport) -> str:
        return "tok"

    monkeypatch.setattr(GeminiLiveTransport, "_get_vertex_token", fake_token)

    await transport.connect(system="hi", tools=[])
    assert http.connect_url is not None
    assert "global-aiplatform" not in http.connect_url
    assert "us-central1-aiplatform" in http.connect_url
    setup = json.loads(ws.sent[0])["setup"]
    assert "/locations/us-central1/" in setup["model"]


async def test_vertex_connect_uses_bearer_and_full_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)
    transport = GeminiLiveTransport(
        http_session=cast(Any, http),
        vertexai=True,
        project="my-proj",
        location="us-central1",
    )

    async def fake_token(self: GeminiLiveTransport) -> str:
        return "vertex-token-xyz"

    monkeypatch.setattr(GeminiLiveTransport, "_get_vertex_token", fake_token)

    await transport.connect(system="be brief", tools=[], voice="Charon")
    assert http.connect_url is not None and http.connect_url.startswith(
        "wss://us-central1-aiplatform.googleapis.com/ws/"
    )
    assert http.connect_headers is not None
    assert http.connect_headers["Authorization"] == "Bearer vertex-token-xyz"
    assert http.connect_headers["x-goog-user-project"] == "my-proj"
    setup = json.loads(ws.sent[0])["setup"]
    assert setup["model"] == (
        "projects/my-proj/locations/us-central1/publishers/google/models/gemini-live-2.5-flash-native-audio"
    )


def test_detect_system_language_picks_supported_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    from axio_transport_google.realtime import SUPPORTED_LIVE_LANGUAGES, detect_system_language

    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    import locale

    monkeypatch.setattr(locale, "getlocale", lambda: (None, None))
    detected = detect_system_language()
    assert detected == "ru-RU"
    assert detected in SUPPORTED_LIVE_LANGUAGES


def test_detect_system_language_returns_none_for_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    from axio_transport_google.realtime import detect_system_language

    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.setenv("LANG", "xx_YY.UTF-8")
    import locale

    monkeypatch.setattr(locale, "getlocale", lambda: (None, None))
    assert detect_system_language() is None


# Make pytest pick up async assertion failures from these helpers.
@pytest.fixture(autouse=True)
def _enforce_asyncio_marker() -> None:
    return None
