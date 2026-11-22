"""Tests for the OpenAI Realtime WebSocket transport.

Uses a stub WebSocket that records outbound messages and replays a scripted
sequence of inbound provider events.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from axio import RealtimeAgent, RealtimeToolResponse
from axio.blocks import AudioBlock, ImageBlock, TextBlock, VideoBlock
from axio.events import (
    AudioOutputDelta,
    AudioOutputEnd,
    Error,
    ProviderEvent,
    SpeechStarted,
    SpeechStopped,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
    TranscriptDelta,
    TranscriptFinal,
    TurnComplete,
    TurnStarted,
)
from axio.tool import Tool
from axio.types import StopReason

from axio_transport_openai.realtime import (
    OpenAIRealtimeSession,
    OpenAIRealtimeTransport,
    convert_realtime_tools,
)


class _StubWSMessage:
    def __init__(self, msg_type: aiohttp.WSMsgType, data: str = "") -> None:
        self.type = msg_type
        self.data = data


class _StubWebSocket:
    """Stub for ``aiohttp.ClientWebSocketResponse``."""

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


def _decode_sent(ws: _StubWebSocket) -> list[dict[str, Any]]:
    return [json.loads(s) for s in ws.sent]


# ---------------------------------------------------------------------------
# Outbound events
# ---------------------------------------------------------------------------


class TestOutboundEvents:
    async def test_audio_block_appends_to_input_buffer(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.send(AudioBlock(media_type="audio/pcm", data=b"\x01\x02\x03"))
        events = _decode_sent(ws)
        assert events == [{"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01\x02\x03").decode()}]

    async def test_text_block_creates_user_message(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.send(TextBlock(text="hello"))
        events = _decode_sent(ws)
        assert events == [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            }
        ]

    async def test_commit_sends_buffer_commit_only_after_audio(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.send(AudioBlock(media_type="audio/pcm", data=b"\x00\x01"))
        await session.commit()
        events = _decode_sent(ws)
        assert events[-2] == {"type": "input_audio_buffer.commit"}
        assert events[-1] == {"type": "response.create"}

    async def test_commit_text_only_skips_buffer_commit(self) -> None:
        """Text-only turns must not emit input_audio_buffer.commit — OpenAI
        rejects it with input_audio_buffer_commit_empty when the buffer is
        empty."""
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.send(TextBlock(text="hello"))
        await session.commit()
        events = _decode_sent(ws)
        kinds = [e["type"] for e in events]
        assert "input_audio_buffer.commit" not in kinds
        assert kinds[-1] == "response.create"

    async def test_interrupt_sends_response_cancel(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        assert session._translate({"type": "response.created", "response": {"id": "r"}}) == [
            TurnStarted(response_id="r")
        ]
        await session.interrupt()
        await session.interrupt()
        assert _decode_sent(ws) == [{"type": "response.cancel", "response_id": "r"}]

    async def test_send_tool_result_str(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.send_tool_result("call_42", "answer", "the answer is 42")
        events = _decode_sent(ws)
        assert events == [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_42",
                    "output": "the answer is 42",
                },
            },
            {"type": "response.create"},
        ]

    async def test_send_tool_result_text_blocks(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.send_tool_result("call_1", "concat", [TextBlock(text="a"), TextBlock(text="b")])
        first = _decode_sent(ws)[0]
        assert first["item"]["output"] == "ab"

    async def test_tool_batch_validates_before_sending_and_resumes_only_once(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        invalid = [
            RealtimeToolResponse("a", "one", "ok"),
            RealtimeToolResponse("b", "two", [ImageBlock(media_type="image/png", data=b"image")]),
        ]
        with pytest.raises(TypeError):
            await session.send_tool_results(invalid)
        assert not ws.sent
        await session.send_tool_results([])
        assert not ws.sent
        await session.send_tool_results([RealtimeToolResponse("a", "one", "1"), RealtimeToolResponse("b", "two", "2")])
        assert [event["type"] for event in _decode_sent(ws)] == [
            "conversation.item.create",
            "conversation.item.create",
            "response.create",
        ]
        assert [event["item"]["call_id"] for event in _decode_sent(ws)[:-1]] == ["a", "b"]

    async def test_interrupt_does_not_cancel_an_idle_or_completed_response(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.interrupt()
        session._translate({"type": "response.created", "response": {"id": "a"}})
        session._translate({"type": "response.done", "response": {"id": "a", "status": "completed"}})
        await session.interrupt()
        assert not ws.sent

    async def test_interrupt_cancels_a_requested_response_before_created_arrives(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.commit()
        await session.interrupt()
        await session.interrupt()
        assert _decode_sent(ws) == [{"type": "response.create"}, {"type": "response.cancel"}]

    async def test_late_done_does_not_mark_the_new_response_idle(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        session._translate({"type": "response.created", "response": {"id": "old"}})
        await session.interrupt()
        session._translate({"type": "response.created", "response": {"id": "new"}})
        session._translate({"type": "response.done", "response": {"id": "old", "status": "cancelled"}})
        await session.interrupt()
        assert _decode_sent(ws) == [
            {"type": "response.cancel", "response_id": "old"},
            {"type": "response.cancel", "response_id": "new"},
        ]

    async def test_send_unsupported_block_raises(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        with pytest.raises(TypeError):
            await session.send(VideoBlock(media_type="video/mp4", data=b"video"))

    async def test_panorama_is_one_ordered_message(self) -> None:
        ws = _StubWebSocket([])
        session = OpenAIRealtimeSession(ws=ws)
        await session.send(
            [
                TextBlock(text="Слева, угол −30°"),
                ImageBlock(media_type="image/jpeg", data=b"left"),
                TextBlock(text="Справа, угол +30°"),
                ImageBlock(media_type="image/png", data=b"right"),
            ]
        )
        assert _decode_sent(ws) == [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Слева, угол −30°"},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,bGVmdA=="},
                        {"type": "input_text", "text": "Справа, угол +30°"},
                        {"type": "input_image", "image_url": "data:image/png;base64,cmlnaHQ="},
                    ],
                },
            }
        ]

    @pytest.mark.parametrize(
        "bad",
        [
            [TextBlock(text="partial"), ImageBlock(media_type="image/jpeg", data=b"")],
            [TextBlock(text="partial"), AudioBlock(media_type="audio/pcm", data=b"12")],
            [AudioBlock(media_type="audio/pcm", data=b"12"), VideoBlock(media_type="video/mp4", data=b"video")],
        ],
    )
    async def test_invalid_message_does_not_send_partial_content(self, bad: list[Any]) -> None:
        ws = _StubWebSocket([])
        with pytest.raises((TypeError, ValueError)):
            await OpenAIRealtimeSession(ws=ws).send(bad)
        assert not ws.sent

    async def test_truncate_uses_played_duration_and_content_index(self) -> None:
        ws = _StubWebSocket([])
        await OpenAIRealtimeSession(ws=ws).truncate("assistant-item", 720, content_index=2)
        assert _decode_sent(ws) == [
            {
                "type": "conversation.item.truncate",
                "item_id": "assistant-item",
                "content_index": 2,
                "audio_end_ms": 720,
            }
        ]

    @pytest.mark.parametrize(
        "item,duration,index", [("", 0, 0), ("i", -1, 0), ("i", True, 0), ("i", 1.5, 0), ("i", 0, -1), ("i", 0, False)]
    )
    async def test_invalid_truncate_never_reaches_provider(self, item: str, duration: Any, index: Any) -> None:
        ws = _StubWebSocket([])
        with pytest.raises(ValueError):
            await OpenAIRealtimeSession(ws=ws).truncate(item, duration, content_index=index)
        assert not ws.sent


# ---------------------------------------------------------------------------
# Inbound event translation
# ---------------------------------------------------------------------------


class TestInboundTranslation:
    async def _drain(self, scripted: list[dict[str, Any]]) -> tuple[_StubWebSocket, list[Any]]:
        ws = _StubWebSocket(scripted)
        ws.push(None)  # close the iterator after scripted
        session = OpenAIRealtimeSession(ws=ws)
        collected = [ev async for ev in session.events()]
        return ws, collected

    async def test_audio_delta_decoded_to_audio_output_delta(self) -> None:
        pcm = b"\x10\x20\x30"
        for type_name in ("response.audio.delta", "response.output_audio.delta"):
            _, events = await self._drain(
                [
                    {
                        "type": type_name,
                        "delta": base64.b64encode(pcm).decode(),
                    }
                ]
            )
            assert events == [AudioOutputDelta(data=pcm, media_type="audio/pcm;rate=24000")]

    async def test_assistant_transcript_delta(self) -> None:
        for type_name in (
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
        ):
            _, events = await self._drain([{"type": type_name, "delta": "hel"}])
            assert events == [TranscriptDelta(role="assistant", delta="hel")]

    async def test_server_error_yields_axio_error(self) -> None:
        from axio.events import Error

        _, events = await self._drain([{"type": "error", "error": {"code": "bad", "message": "boom"}}])
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, Error)
        assert "bad" in str(ev.exception)
        assert "boom" in str(ev.exception)

    async def test_user_transcript_delta(self) -> None:
        _, events = await self._drain(
            [
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "hi",
                }
            ]
        )
        assert events == [TranscriptDelta(role="user", delta="hi")]

    async def test_text_delta(self) -> None:
        _, events = await self._drain([{"type": "response.text.delta", "delta": "answer"}])
        assert events == [TextDelta(index=0, delta="answer")]

    async def test_speech_events(self) -> None:
        _, events = await self._drain(
            [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "input_audio_buffer.speech_stopped"},
            ]
        )
        assert events == [SpeechStarted(), SpeechStopped()]

    async def test_function_call_lifecycle(self) -> None:
        _, events = await self._drain(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "name": "add", "call_id": "call_1"},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_1",
                    "delta": '{"a":1',
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_1",
                    "delta": ',"b":2}',
                },
                {
                    "type": "response.done",
                    "response": {
                        "output": [{"type": "function_call", "call_id": "call_1", "name": "add"}],
                        "usage": {"input_tokens": 10, "output_tokens": 4},
                    },
                },
            ]
        )
        assert events[0] == ToolUseStart(index=0, tool_use_id="call_1", name="add")
        assert events[1] == ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"a":1')
        assert events[2] == ToolInputDelta(index=0, tool_use_id="call_1", partial_json=',"b":2}')
        assert isinstance(events[3], TurnComplete)
        assert events[3].stop_reason is StopReason.tool_use
        assert events[3].usage is not None
        assert events[3].usage.input_tokens == 10
        assert events[3].usage.output_tokens == 4

    async def test_response_done_text_yields_end_turn(self) -> None:
        _, events = await self._drain(
            [
                {
                    "type": "response.done",
                    "response": {
                        "output": [{"type": "message"}],
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    },
                }
            ]
        )
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, TurnComplete)
        assert ev.stop_reason is StopReason.end_turn
        assert ev.usage is not None
        assert ev.usage.input_tokens == 5

    async def test_unknown_event_is_forwarded(self) -> None:
        _, events = await self._drain([{"type": "session.created"}])
        assert events == [ProviderEvent(provider="openai", kind="session.created", data={"type": "session.created"})]


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


async def test_convert_tools_uses_flat_function_shape() -> None:
    async def add(a: int, b: int) -> str:
        return str(a + b)

    out = convert_realtime_tools([Tool(name="add", handler=add, description="adds")])
    assert out[0]["type"] == "function"
    assert out[0]["name"] == "add"
    assert out[0]["description"] == "adds"
    assert "parameters" in out[0]
    assert "function" not in out[0]


# ---------------------------------------------------------------------------
# connect() wires up session.update with system / tools / voice
# ---------------------------------------------------------------------------


class _StubHttpSession:
    def __init__(self, ws: _StubWebSocket) -> None:
        self._ws = ws
        self.connect_url: str | None = None
        self.connect_headers: dict[str, str] | None = None
        self.closed = False

    async def ws_connect(self, url: str, *, headers: dict[str, str]) -> _StubWebSocket:
        self.connect_url = url
        self.connect_headers = headers
        return self._ws

    async def close(self) -> None:
        self.closed = True


async def test_connect_sends_session_update() -> None:
    async def add(a: int, b: int) -> str:
        return str(a + b)

    ws = _StubWebSocket([{"type": "session.updated"}])
    http = _StubHttpSession(ws)
    transport = OpenAIRealtimeTransport(api_key="sk-test", model="gpt-realtime", http_session=http)  # type: ignore[arg-type]
    session = await transport.connect(
        system="be brief",
        tools=[Tool(name="add", handler=add)],
        voice="marin",
    )
    assert isinstance(session, OpenAIRealtimeSession)
    assert http.connect_url == "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    assert http.connect_headers is not None
    assert http.connect_headers["Authorization"] == "Bearer sk-test"
    assert "OpenAI-Beta" not in http.connect_headers

    sent = _decode_sent(ws)
    assert len(sent) == 1
    update = sent[0]
    assert update["type"] == "session.update"
    s = update["session"]
    assert s["instructions"] == "be brief"
    assert s["type"] == "realtime"
    assert s["output_modalities"] == ["audio"]
    assert s["audio"]["output"]["voice"] == "marin"
    assert s["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert s["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert s["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert s["tools"][0]["name"] == "add"


@pytest.mark.parametrize(
    "status,reason,expected",
    [
        ("cancelled", None, StopReason.cancelled),
        ("failed", None, StopReason.error),
        ("incomplete", "max_output_tokens", StopReason.max_tokens),
        ("incomplete", "content_filter", StopReason.refusal),
        ("incomplete", "unknown", StopReason.error),
        ("unknown", None, StopReason.error),
    ],
)
async def test_ga_response_status(status: str, reason: str | None, expected: StopReason) -> None:
    session = OpenAIRealtimeSession(ws=_StubWebSocket([]))
    events = session._translate(
        {
            "type": "response.done",
            "response": {
                "status": status,
                "status_details": {"reason": reason},
                "output": [{"type": "function_call"}],
                "usage": {"input_tokens": 10, "output_tokens": 2, "input_token_details": {"cached_tokens": 4}},
            },
        }
    )
    final = events[-1]
    assert isinstance(final, TurnComplete)
    assert final.stop_reason == expected
    assert final.usage is not None and final.usage.cache_read_tokens == 4


async def test_ga_transcription_completion_does_not_duplicate_deltas() -> None:
    session = OpenAIRealtimeSession(ws=_StubWebSocket([]))
    assert session._translate({"type": "response.output_text.delta", "delta": "hi"}) == [
        TextDelta(index=0, delta="hi")
    ]
    for item in ("first", "second"):
        session._translate(
            {"type": "conversation.item.input_audio_transcription.delta", "item_id": item, "delta": "hel"}
        )
    for item in ("second", "first"):
        completed = {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item,
            "transcript": "hello",
        }
        assert session._translate(completed) == [
            TranscriptDelta(role="user", delta="lo", item_id=item),
            TranscriptFinal(role="user", text="hello", item_id=item),
            ProviderEvent(provider="openai", kind=completed["type"], data=completed),
        ]
    completed = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "whole",
        "transcript": "hi",
    }
    assert session._translate(completed) == [
        TranscriptDelta(role="user", delta="hi", item_id="whole"),
        TranscriptFinal(role="user", text="hi", item_id="whole"),
        ProviderEvent(provider="openai", kind=completed["type"], data=completed),
    ]


async def test_reject_rate_before_connecting() -> None:
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)
    transport = OpenAIRealtimeTransport(api_key="test", http_session=http)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="24000"):
        await transport.connect(system="", tools=[], input_audio_format="audio/pcm;rate=16000")
    assert http.connect_url is None
    session = OpenAIRealtimeSession(ws=ws)
    with pytest.raises(ValueError):
        await session.send(AudioBlock(media_type="audio/pcm;rate=16000", data=b"12"))  # type: ignore[arg-type]
    assert not ws.sent


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_readiness_requires_a_bounded_timeout(timeout: float) -> None:
    http = _StubHttpSession(_StubWebSocket([]))
    transport = OpenAIRealtimeTransport(api_key="test", http_session=http, ready_timeout=timeout)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        await transport.connect(system="", tools=[])
    assert http.connect_url is None


async def test_setup_failure_cleans_owned_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)

    async def fail(data: str) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(ws, "send_str", fail)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda: http)
    with pytest.raises(asyncio.CancelledError):
        await OpenAIRealtimeTransport(api_key="test").connect(system="", tools=[])
    assert ws.closed and http.closed


async def test_close_is_idempotent_and_preserves_shared_client() -> None:
    ws = _StubWebSocket([{"type": "session.updated"}])
    http = _StubHttpSession(ws)
    transport = OpenAIRealtimeTransport(api_key="test", http_session=http)  # type: ignore[arg-type]
    session = await transport.connect(system="", tools=[])
    await session.close()
    await session.close()
    assert ws.closed and not http.closed


async def test_owned_connect_failure_closes_http(monkeypatch: pytest.MonkeyPatch) -> None:
    http = _StubHttpSession(_StubWebSocket([]))

    async def fail(url: str, *, headers: dict[str, str]) -> _StubWebSocket:
        raise aiohttp.ClientConnectionError("offline")

    monkeypatch.setattr(http, "ws_connect", fail)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda: http)
    with pytest.raises(aiohttp.ClientConnectionError):
        await OpenAIRealtimeTransport(api_key="test").connect(system="", tools=[])
    assert http.closed


async def test_manual_vad_transcription_and_g711() -> None:
    ws = _StubWebSocket([{"type": "session.updated"}])
    http = _StubHttpSession(ws)
    transport = OpenAIRealtimeTransport(
        api_key="test",
        http_session=http,  # type: ignore[arg-type]
        server_vad=False,
        transcription_model="gpt-4o-mini-transcribe",
        input_noise_reduction=None,
    )
    await transport.connect(system="", tools=[], input_audio_format="audio/pcmu", output_audio_format="audio/pcma")
    audio = _decode_sent(ws)[0]["session"]["audio"]
    assert audio["input"]["turn_detection"] is None
    assert audio["input"]["noise_reduction"] is None
    assert audio["input"]["transcription"] == {"model": "gpt-4o-mini-transcribe"}
    assert "language" not in audio["input"]["transcription"]  # multilingual auto-detection
    assert audio["input"]["format"] == {"type": "audio/pcmu"}
    assert audio["output"]["format"] == {"type": "audio/pcma"}


async def test_final_transcripts_preserve_identity_even_when_cancelled_or_late() -> None:
    session = OpenAIRealtimeSession(ws=_StubWebSocket([]))
    identity: dict[str, Any] = {"item_id": "assistant-1", "response_id": "response-1", "content_index": 2}
    pcm = session._translate({"type": "response.output_audio.delta", "delta": "MTI=", **identity})
    assert pcm == [AudioOutputDelta(data=b"12", **identity)]
    assert session._translate({"type": "response.output_audio.done", **identity}) == [AudioOutputEnd(**identity)]
    assert session._translate({"type": "response.output_audio_transcript.delta", "delta": "Привет", **identity}) == [
        TranscriptDelta(role="assistant", delta="Привет", **identity)
    ]
    assert session._translate(
        {"type": "response.output_audio_transcript.done", "transcript": "Привет! Hello!", **identity}
    ) == [TranscriptFinal(role="assistant", text="Привет! Hello!", **identity)]
    assert session._translate({"type": "response.done", "response": {"id": "response-1", "status": "cancelled"}}) == [
        TurnComplete(stop_reason=StopReason.cancelled, response_id="response-1")
    ]
    # Input ASR is independently scheduled and may finish after the assistant.
    user: dict[str, Any] = {"item_id": "user-1", "content_index": 0}
    session._translate({"type": "conversation.item.input_audio_transcription.delta", "delta": "Добро", **user})
    completed = {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Good morning", **user}
    assert session._translate(completed) == [
        TranscriptFinal(role="user", text="Good morning", **user),
        ProviderEvent(provider="openai", kind=completed["type"], data=completed),
    ]


async def test_failed_input_transcription_keeps_identity_without_archiving_partial_text() -> None:
    session = OpenAIRealtimeSession(ws=_StubWebSocket([]))
    user: dict[str, Any] = {"item_id": "user-2", "content_index": 1}
    session._translate({"type": "conversation.item.input_audio_transcription.delta", "delta": "unfinished", **user})
    events = session._translate({"type": "conversation.item.input_audio_transcription.failed", **user})
    assert events[0] == TranscriptFinal(role="user", text="", status="failed", **user)
    assert isinstance(events[1], ProviderEvent)
    assert not any(isinstance(event, Error) for event in events)
    assert not session._transcripts


async def test_connect_waits_for_ack_and_preserves_early_events() -> None:
    early = {"type": "session.created", "session": {"id": "s"}}
    ws = _StubWebSocket([early])
    http = _StubHttpSession(ws)
    transport = OpenAIRealtimeTransport(api_key="test", http_session=http, ready_timeout=1)  # type: ignore[arg-type]
    connecting = asyncio.create_task(transport.connect(system="", tools=[]))
    await asyncio.sleep(0)
    assert not connecting.done()
    ws.push({"type": "session.updated"})
    session = await connecting
    ws.push(None)
    assert [event async for event in session.events()] == [
        ProviderEvent(provider="openai", kind="session.created", data=early),
        ProviderEvent(provider="openai", kind="session.updated", data={"type": "session.updated"}),
    ]
    await session.close()
    assert not http.closed


@pytest.mark.parametrize("failure", ["timeout", "closed", "rejected", "overflow"])
@pytest.mark.parametrize("owned", [False, True])
async def test_handshake_failure_closes_socket_and_only_owned_http(
    failure: str, owned: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = [{"type": "session.created"}] * 257 if failure == "overflow" else []
    ws = _StubWebSocket(scripted)
    if failure == "closed":
        ws.push(None)
    elif failure == "rejected":
        ws.push({"type": "error", "error": {"message": "private configuration"}})
    http = _StubHttpSession(ws)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda: http)
    transport = OpenAIRealtimeTransport(api_key="test", http_session=None if owned else http, ready_timeout=0.02)  # type: ignore[arg-type]
    from axio.exceptions import StreamError

    with pytest.raises((StreamError, TimeoutError)) as exc:
        await transport.connect(system="", tools=[])
    assert "private configuration" not in str(exc.value)
    assert ws.closed and http.closed is owned


async def test_cancel_during_readiness_cleans_owned_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda: http)
    connecting = asyncio.create_task(OpenAIRealtimeTransport(api_key="test").connect(system="", tools=[]))
    await asyncio.sleep(0)
    connecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connecting
    assert ws.closed and http.closed


async def test_malformed_audio_event_surfaces_error() -> None:
    from axio.events import Error

    ws = _StubWebSocket([{"type": "response.output_audio.delta", "delta": "!!!"}])
    ws.push(None)
    events = [event async for event in OpenAIRealtimeSession(ws=ws).events()]
    assert len(events) == 1 and isinstance(events[0], Error)


@pytest.mark.parametrize("payload", ["[]", "null", "invalid JSON"])
async def test_malformed_event_surfaces_error(payload: str) -> None:
    from axio.events import Error

    class BrokenWS(_StubWebSocket):
        async def _iter(self) -> AsyncIterator[_StubWSMessage]:
            yield _StubWSMessage(aiohttp.WSMsgType.TEXT, payload)

    events = [event async for event in OpenAIRealtimeSession(ws=BrokenWS([])).events()]
    assert len(events) == 1 and isinstance(events[0], Error)


async def test_websocket_error_surfaces_error() -> None:
    from axio.events import Error

    class BrokenWS(_StubWebSocket):
        async def _iter(self) -> AsyncIterator[_StubWSMessage]:
            yield _StubWSMessage(aiohttp.WSMsgType.ERROR)

    events = [event async for event in OpenAIRealtimeSession(ws=BrokenWS([])).events()]
    assert len(events) == 1 and isinstance(events[0], Error)


async def test_close_http_even_if_socket_close_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _StubWebSocket([])
    http = _StubHttpSession(ws)

    async def fail() -> None:
        raise RuntimeError("close failed")

    monkeypatch.setattr(ws, "close", fail)
    session = OpenAIRealtimeSession(ws=ws, http_session=http, own_http_session=True)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        await session.close()
    assert http.closed


async def test_ga_session_over_websocket() -> None:
    async def realtime(request: web.Request) -> web.WebSocketResponse:
        assert request.headers["Authorization"] == "Bearer test"
        assert "OpenAI-Beta" not in request.headers
        assert request.query["model"] == "gpt-realtime"
        assert request.query["custom"] == "kept"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        update = await ws.receive_json()
        assert update["session"]["type"] == "realtime"
        assert update["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        await ws.send_json({"type": "session.updated"})
        await ws.send_json({"type": "response.output_audio.delta", "delta": base64.b64encode(b"12").decode()})
        await ws.send_json({"type": "response.done", "response": {"status": "completed"}})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/realtime", realtime)
    async with TestServer(app) as server, aiohttp.ClientSession() as client:
        transport = OpenAIRealtimeTransport(
            api_key="test",
            base_url=str(server.make_url("/realtime?custom=kept")),
            http_session=client,
        )
        session = await transport.connect(system="", tools=[])
        try:
            events = [event async for event in session.events()]
            assert events == [
                ProviderEvent(provider="openai", kind="session.updated", data={"type": "session.updated"}),
                AudioOutputDelta(data=b"12"),
                TurnComplete(stop_reason=StopReason.end_turn),
            ]
        finally:
            await session.close()
        assert not client.closed


@pytest.mark.parametrize("slow_fails", [False, True])
async def test_agent_runs_tools_concurrently_then_sends_one_openai_continuation(slow_fails: bool) -> None:
    slow_started, release = asyncio.Event(), asyncio.Event()

    async def slow() -> str:
        slow_started.set()
        await release.wait()
        if slow_fails:
            raise ValueError("cannot finish")
        return "slow result"

    async def fast() -> dict[str, str]:
        await slow_started.wait()
        return {"result": "быстро"}

    scripted: list[dict[str, Any]] = [
        {"type": "session.updated"},
        {"type": "response.created", "response": {"id": "r"}},
        {
            "type": "response.output_item.added",
            "response_id": "r",
            "item": {"type": "function_call", "call_id": "slow-id", "name": "slow"},
        },
        {"type": "response.function_call_arguments.delta", "response_id": "r", "call_id": "slow-id", "delta": "{}"},
        {
            "type": "response.output_item.added",
            "response_id": "r",
            "item": {"type": "function_call", "call_id": "fast-id", "name": "fast"},
        },
        {"type": "response.function_call_arguments.delta", "response_id": "r", "call_id": "fast-id", "delta": "{}"},
        {
            "type": "response.done",
            "response": {"id": "r", "status": "completed", "output": [{"type": "function_call"}]},
        },
    ]
    ws = _StubWebSocket(scripted)
    ws.push(None)
    http = _StubHttpSession(ws)
    transport = OpenAIRealtimeTransport(api_key="test", http_session=http)  # type: ignore[arg-type]
    results: list[ToolResult] = []
    async with RealtimeAgent(
        system="",
        transport=transport,
        tools=[Tool(name="slow", handler=slow), Tool(name="fast", handler=fast)],
        input_audio_format="audio/pcm;rate=24000",
    ) as agent:
        async for event in agent.events():
            if isinstance(event, ToolResult):
                results.append(event)
                if event.tool_use_id == "fast-id":
                    assert not any(payload["type"] == "response.create" for payload in _decode_sent(ws))
                    assert event.response_id == "r" and not event.is_error
                    assert json.loads(event.content) == {"result": "быстро"}
                    release.set()
    outputs = [payload for payload in _decode_sent(ws) if payload["type"] == "conversation.item.create"]
    assert [output["item"]["call_id"] for output in outputs] == ["slow-id", "fast-id"]
    assert sum(payload["type"] == "response.create" for payload in _decode_sent(ws)) == 1
    assert [result.tool_use_id for result in results] == ["fast-id", "slow-id"]
    assert results[-1].is_error is slow_fails


@pytest.mark.parametrize(
    "usage",
    [
        {
            "type": "tokens",
            "input_tokens": 13,
            "output_tokens": 9,
            "total_tokens": 22,
            "input_token_details": {"audio_tokens": 13, "text_tokens": 0},
        },
        {"type": "duration", "seconds": 2.5},
        None,
        {"type": "future", "unknown": "value"},
    ],
)
async def test_input_asr_usage_is_forwarded_separately_from_response_usage(usage: Any) -> None:
    raw = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "user",
        "content_index": 0,
        "transcript": "Synthetic final",
        "usage": usage,
    }
    session = OpenAIRealtimeSession(ws=_StubWebSocket([]))
    response = session._translate(
        {
            "type": "response.done",
            "response": {"id": "reply", "status": "completed", "usage": {"input_tokens": 100, "output_tokens": 10}},
        }
    )
    assert isinstance(response[0], TurnComplete)
    events = session._translate(raw)
    provider = next(event for event in events if isinstance(event, ProviderEvent))
    assert provider.kind == raw["type"] and provider.data == raw
    assert response[0].usage is not None and response[0].usage.input_tokens == 100
    assert not any(isinstance(event, TurnComplete) for event in events)


async def test_failed_optional_asr_does_not_end_actual_realtime_agent_audio_dialogue() -> None:
    ws = _StubWebSocket(
        [
            {"type": "session.updated"},
            {"type": "conversation.item.input_audio_transcription.failed", "item_id": "user", "content_index": 0},
            {"type": "response.created", "response": {"id": "reply"}},
            {
                "type": "response.output_audio.delta",
                "response_id": "reply",
                "item_id": "answer",
                "content_index": 0,
                "delta": "MTI=",
            },
            {"type": "response.done", "response": {"id": "reply", "status": "completed"}},
        ]
    )
    http = _StubHttpSession(ws)
    agent = RealtimeAgent(
        system="Synthetic test",
        input_audio_format="audio/pcm;rate=24000",
        output_audio_format="audio/pcm;rate=24000",
        transport=OpenAIRealtimeTransport(
            api_key="synthetic",
            http_session=http,  # type: ignore[arg-type]
            transcription_model="gpt-4o-mini-transcribe",
        ),
    )
    await agent.connect()
    events = []
    try:
        async with asyncio.timeout(2):
            async for event in agent.events():
                events.append(event)
                if isinstance(event, TurnComplete):
                    break
        assert any(isinstance(event, TranscriptFinal) and event.status == "failed" for event in events)
        assert any(isinstance(event, AudioOutputDelta) and event.data == b"12" for event in events)
        assert not any(isinstance(event, Error) for event in events)
    finally:
        await agent.close()
