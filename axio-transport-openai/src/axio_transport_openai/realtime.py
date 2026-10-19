"""OpenAI Realtime WebSocket transport.

Implements :class:`axio.transport.RealtimeTransport` against the OpenAI
Realtime API (``wss://api.openai.com/v1/realtime``).  Provider events are
mapped to axio :class:`StreamEvent` variants.  The session's
``send`` / ``commit`` / ``interrupt`` / ``send_tool_result`` methods are
mapped to the corresponding client events.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import aiohttp
from axio.blocks import AudioBlock, ContentBlock, ImageBlock, TextBlock
from axio.events import (
    AudioOutputDelta,
    AudioOutputEnd,
    Error,
    ProviderEvent,
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    TranscriptDelta,
    TranscriptFinal,
    TurnComplete,
)
from axio.exceptions import StreamError
from axio.tool import Tool
from axio.transport import RealtimeSession, RealtimeTransport
from axio.types import StopReason, ToolCallID, ToolName, Usage
from yarl import URL

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_PCM_MEDIA_TYPE = "audio/pcm;rate=24000"

RESPONSE_STOP_REASONS = MappingProxyType(
    {
        "completed": StopReason.end_turn,
        "cancelled": StopReason.cancelled,
        "failed": StopReason.error,
        "incomplete": StopReason.error,
    }
)
INCOMPLETE_STOP_REASONS = MappingProxyType(
    {
        "max_output_tokens": StopReason.max_tokens,
        "content_filter": StopReason.refusal,
    }
)


def convert_realtime_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio ``Tool``s to OpenAI realtime tool entries (flat ``function`` shape)."""
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
        for t in tools
    ]


def audio_format(media_type: str) -> dict[str, Any]:
    """GA audio format objects. Never relabel a different PCM sample rate."""
    if media_type in ("audio/pcm", DEFAULT_PCM_MEDIA_TYPE):
        return {"type": "audio/pcm", "rate": 24000}
    if media_type in ("audio/pcmu", "audio/pcma"):
        return {"type": media_type}
    raise ValueError(f"OpenAI realtime: unsupported audio format {media_type!r}; PCM requires 24000 Hz mono")


@dataclass(slots=True)
class OpenAIRealtimeSession(RealtimeSession):
    """RealtimeSession backed by an aiohttp WebSocket.

    ``ws`` may be any object that quacks like
    :class:`aiohttp.ClientWebSocketResponse` (``send_str``, async iteration
    yielding :class:`aiohttp.WSMessage`-like objects, ``close``, ``closed``).
    Tests inject a stub.
    """

    ws: Any
    input_audio_media_type: str = DEFAULT_PCM_MEDIA_TYPE
    output_audio_media_type: str = DEFAULT_PCM_MEDIA_TYPE
    http_session: aiohttp.ClientSession | None = field(default=None, repr=False)
    own_http_session: bool = False
    _tool_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _next_tool_index: int = field(default=0, init=False, repr=False)
    _audio_appended: bool = field(default=False, init=False, repr=False)
    _transcripts: dict[tuple[str, int], str] = field(default_factory=dict, init=False, repr=False)
    pending_events: deque[dict[str, Any]] = field(default_factory=deque, init=False, repr=False)
    ws_messages: AsyncIterator[Any] | None = field(default=None, init=False, repr=False)

    async def _send_event(self, payload: dict[str, Any]) -> None:
        await self.ws.send_str(json.dumps(payload))

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        """Send an atomic text/image message or append homogeneous audio chunks.

        All blocks are validated before sending. Audio cannot be mixed with
        text/images in one call because it uses a separate provider buffer.
        """
        blocks = content if isinstance(content, list) else [content]
        payloads: list[dict[str, Any]] = []
        message: list[dict[str, Any]] = []
        for block in blocks:
            match block:
                case AudioBlock(data=data, media_type=media_type):
                    if audio_format(media_type) != audio_format(self.input_audio_media_type):
                        raise ValueError("Audio block format does not match the connected input format")
                    payloads.append(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(data).decode(),
                        }
                    )
                case TextBlock(text=text):
                    message.append({"type": "input_text", "text": text})
                case ImageBlock(data=data, media_type=media_type):
                    if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp") or not data:
                        raise ValueError("OpenAI realtime requires nonempty JPEG, PNG, GIF or WebP image bytes")
                    message.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{media_type};base64,{base64.b64encode(data).decode()}",
                        }
                    )
                case _:
                    raise TypeError(f"OpenAI realtime: send() does not support {type(block).__name__}")
        if message and payloads:
            raise ValueError("Send audio separately from a text/image message")
        if message:
            payloads.append(
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": message}}
            )
        for payload in payloads:
            await self._send_event(payload)
            if payload["type"] == "input_audio_buffer.append":
                self._audio_appended = True

    async def commit(self) -> None:
        # OpenAI rejects input_audio_buffer.commit on an empty buffer
        # ("input_audio_buffer_commit_empty").  Skip it for text-only turns.
        if self._audio_appended:
            await self._send_event({"type": "input_audio_buffer.commit"})
            self._audio_appended = False
        await self._send_event({"type": "response.create"})

    async def interrupt(self) -> None:
        await self._send_event({"type": "response.cancel"})

    async def truncate(self, item_id: str, audio_end_ms: int, *, content_index: int = 0) -> None:
        """Remove unplayed audio and its transcript from the server conversation."""
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item_id must be nonempty")
        for name, value in (("audio_end_ms", audio_end_ms), ("content_index", content_index)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        await self._send_event(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": audio_end_ms,
            }
        )

    async def send_tool_result(
        self, tool_use_id: ToolCallID, name: ToolName, content: str | list[ContentBlock]
    ) -> None:
        if isinstance(content, list):
            parts: list[str] = []
            for b in content:
                if isinstance(b, TextBlock):
                    parts.append(b.text)
                else:
                    raise TypeError(
                        f"OpenAI realtime tool result only supports str / TextBlock, got {type(b).__name__}"
                    )
            output = "".join(parts)
        else:
            output = content
        await self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": output,
                },
            }
        )
        await self._send_event({"type": "response.create"})

    async def read_payload(self) -> dict[str, Any] | None:
        """Read from the single socket iterator; return None on orderly closure."""
        if self.ws_messages is None:
            self.ws_messages = self.ws.__aiter__()
        async for msg in self.ws_messages:
            mtype = getattr(msg, "type", None)
            if mtype is aiohttp.WSMsgType.TEXT:
                payload = json.loads(msg.data)
                if not isinstance(payload, dict):
                    raise ValueError("Expected a realtime event object")
                return payload
            if mtype is aiohttp.WSMsgType.ERROR:
                raise StreamError("OpenAI realtime WebSocket failed")
            if mtype in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                return None
        return None

    async def wait_ready(self, timeout: float) -> None:
        """Wait for session.updated before input can race session configuration.

        Preserve lifecycle/early events for the event consumer. The bounded
        handshake buffer prevents a peer from filling memory before readiness.
        Call only before starting the events iterator.
        """
        async with asyncio.timeout(timeout):
            while True:
                payload = await self.read_payload()
                if payload is None:
                    raise StreamError("OpenAI realtime closed before session.updated")
                if payload.get("type") == "error":
                    raise StreamError("OpenAI realtime session configuration was rejected")
                if len(self.pending_events) >= 256:
                    raise StreamError("OpenAI realtime handshake exceeded the event buffer")
                self.pending_events.append(payload)
                if payload.get("type") == "session.updated":
                    return

    async def events(self) -> AsyncIterator[StreamEvent]:
        try:
            while True:
                payload = self.pending_events.popleft() if self.pending_events else await self.read_payload()
                if payload is None:
                    break
                for ev in self._translate(payload):
                    yield ev
        except (aiohttp.ClientError, ValueError, KeyError, TypeError, StreamError) as exc:
            error = StreamError("OpenAI realtime stream is invalid or disconnected")
            error.__cause__ = exc
            yield Error(exception=error)

    def _translate(self, ev: dict[str, Any]) -> list[StreamEvent]:
        """Map a raw OpenAI realtime server event to axio StreamEvents."""
        out: list[StreamEvent] = []
        t = ev.get("type")
        # Both the v1 (realtime=v1) and the newer GA schemas appear on this
        # endpoint depending on the model id.  Accept both event-name variants.
        if t in ("response.audio.delta", "response.output_audio.delta"):
            out.append(
                AudioOutputDelta(
                    data=base64.b64decode(ev["delta"], validate=True),
                    media_type=self.output_audio_media_type,
                    item_id=ev.get("item_id"),
                    response_id=ev.get("response_id"),
                    content_index=ev.get("content_index"),
                )
            )
        elif t in ("response.audio.done", "response.output_audio.done"):
            out.append(
                AudioOutputEnd(
                    item_id=ev.get("item_id"), response_id=ev.get("response_id"), content_index=ev.get("content_index")
                )
            )
        elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
            out.append(
                TranscriptDelta(
                    role="assistant",
                    delta=ev["delta"],
                    item_id=ev.get("item_id"),
                    response_id=ev.get("response_id"),
                    content_index=ev.get("content_index"),
                )
            )
        elif t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            out.append(
                TranscriptFinal(
                    role="assistant",
                    text=ev["transcript"],
                    item_id=ev.get("item_id"),
                    response_id=ev.get("response_id"),
                    content_index=ev.get("content_index"),
                )
            )
        elif t == "conversation.item.input_audio_transcription.delta":
            key = (ev.get("item_id", ""), ev.get("content_index", 0))
            self._transcripts[key] = self._transcripts.get(key, "") + ev["delta"]
            out.append(
                TranscriptDelta(
                    role="user", delta=ev["delta"], item_id=ev.get("item_id"), content_index=ev.get("content_index")
                )
            )
        elif t == "conversation.item.input_audio_transcription.completed":
            key = (ev.get("item_id", ""), ev.get("content_index", 0))
            previous = self._transcripts.pop(key, "")
            transcript = ev["transcript"]
            if transcript.startswith(previous) and transcript[len(previous) :]:
                out.append(
                    TranscriptDelta(
                        role="user",
                        delta=transcript[len(previous) :],
                        item_id=ev.get("item_id"),
                        content_index=ev.get("content_index"),
                    )
                )
            out.append(
                TranscriptFinal(
                    role="user", text=transcript, item_id=ev.get("item_id"), content_index=ev.get("content_index")
                )
            )
        elif t == "conversation.item.input_audio_transcription.failed":
            self._transcripts.pop((ev.get("item_id", ""), ev.get("content_index", 0)), None)
            out.append(
                TranscriptFinal(
                    role="user",
                    text="",
                    status="failed",
                    item_id=ev.get("item_id"),
                    content_index=ev.get("content_index"),
                )
            )
            out.append(Error(exception=StreamError("OpenAI realtime input transcription failed")))
        elif t == "input_audio_buffer.committed":
            self._audio_appended = False
            out.append(ProviderEvent(provider="openai", kind=t, data=ev))
        elif t in ("response.text.delta", "response.output_text.delta"):
            out.append(TextDelta(index=0, delta=ev["delta"]))
        elif t == "input_audio_buffer.speech_started":
            out.append(SpeechStarted())
        elif t == "input_audio_buffer.speech_stopped":
            out.append(SpeechStopped())
        elif t == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item["call_id"]
                idx = self._next_tool_index
                self._next_tool_index += 1
                self._tool_index[call_id] = idx
                out.append(ToolUseStart(index=idx, tool_use_id=call_id, name=item["name"]))
        elif t == "response.function_call_arguments.delta":
            call_id = ev["call_id"]
            idx = self._tool_index.get(call_id, 0)
            out.append(ToolInputDelta(index=idx, tool_use_id=call_id, partial_json=ev["delta"]))
        elif t == "response.done":
            resp = ev.get("response") or {}
            items = resp.get("output") or []
            status = resp.get("status", "completed")
            stop_reason = RESPONSE_STOP_REASONS.get(status, StopReason.error)
            if status == "completed" and any(it.get("type") == "function_call" for it in items):
                stop_reason = StopReason.tool_use
            elif status == "incomplete":
                details = resp.get("status_details") or {}
                stop_reason = INCOMPLETE_STOP_REASONS.get(details.get("reason", ""), StopReason.error)
            if stop_reason == StopReason.error:
                logger.warning("OpenAI realtime response failed or returned an unrecognized status/reason")
                out.append(Error(exception=StreamError("OpenAI realtime response failed")))
            usage_data = resp.get("usage")
            usage: Usage | None = None
            if usage_data:
                usage = Usage.reported(
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                    cache_read_tokens=(usage_data.get("input_token_details") or {}).get("cached_tokens", 0),
                )
            out.append(TurnComplete(stop_reason=stop_reason, usage=usage, response_id=resp.get("id")))
            self._tool_index.clear()
            self._next_tool_index = 0
        elif t == "error":
            err = ev.get("error") or {}
            msg = err.get("message") or str(err)
            code = err.get("code") or err.get("type") or "unknown"
            logger.error("OpenAI realtime error [%s]: %s", code, msg)
            out.append(Error(exception=StreamError(f"OpenAI realtime [{code}]: {msg}")))
        else:
            out.append(ProviderEvent(provider="openai", kind=str(t), data=ev))
        return out

    async def close(self) -> None:
        try:
            if not self.ws.closed:
                await self.ws.close()
        finally:
            if self.own_http_session and self.http_session is not None and not self.http_session.closed:
                await self.http_session.close()


@dataclass(slots=True)
class OpenAIRealtimeTransport(RealtimeTransport):
    """RealtimeTransport for OpenAI's Realtime WebSocket API.

    Server-VAD knobs live here so callers can tune barge-in sensitivity
    without touching the transport internals.  When AEC is imperfect or
    the room is loud, the defaults can cause the model's own audio to
    trip ``interrupt_response`` and cancel its response.  Bump
    ``vad_threshold`` higher, lengthen ``vad_silence_duration_ms``, or
    set ``vad_interrupt_response=False`` to require an explicit
    ``agent.interrupt()``.
    """

    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""), repr=False)
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_REALTIME_URL", OPENAI_REALTIME_URL))
    model: str = DEFAULT_REALTIME_MODEL
    ready_timeout: float = field(default=10.0, kw_only=True)
    """Maximum seconds to wait for session.updated, including early events."""
    http_session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)

    server_vad: bool = True
    """Set False for explicit utterance boundaries via commit()."""

    transcription_model: str | None = None
    """Opt in to user transcripts, e.g. gpt-4o-mini-transcribe."""

    vad_threshold: float = 0.5
    """Server VAD activation threshold (0..1).  Higher → less sensitive to
    quiet sounds and bleed-through.  Default 0.5."""

    vad_prefix_padding_ms: int = 300
    """Audio retained before a detected speech start."""

    vad_silence_duration_ms: int = 500
    """Silence required before the server considers a turn complete."""

    vad_create_response: bool = True
    """Whether the server auto-creates a response after each user turn."""

    vad_interrupt_response: bool = True
    """Whether incoming user speech cancels the model's in-flight response.
    Setting this to False makes interruption explicit (call
    ``agent.interrupt()``) — useful when AEC isn't strong enough and the
    model's own voice would otherwise loop back through the mic and
    interrupt itself."""

    input_noise_reduction: str | None = "near_field"
    """Server-side noise reduction profile applied to mic audio.  Options:
    ``"near_field"`` (close-talking mic / headset), ``"far_field"`` (laptop
    or conference mic), or ``None`` to disable."""

    async def connect(
        self,
        *,
        system: str,
        tools: list[Tool[Any]],
        voice: str | None = None,
        input_audio_format: str = "audio/pcm;rate=24000",
        output_audio_format: str = "audio/pcm;rate=24000",
    ) -> RealtimeSession:
        input_format = audio_format(input_audio_format)
        output_format = audio_format(output_audio_format)
        if not math.isfinite(self.ready_timeout) or self.ready_timeout <= 0:
            raise ValueError("ready_timeout must be positive and finite")
        if not self.api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for realtime API audio")
        own_session = self.http_session is None or self.http_session.closed
        client = aiohttp.ClientSession() if own_session else self.http_session
        assert client is not None
        ws: Any = None
        try:
            url = str(URL(self.base_url).update_query(model=self.model))
            ws = await client.ws_connect(url, headers={"Authorization": f"Bearer {self.api_key}"})
            audio_input: dict[str, Any] = {
                "format": input_format,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.vad_threshold,
                    "prefix_padding_ms": self.vad_prefix_padding_ms,
                    "silence_duration_ms": self.vad_silence_duration_ms,
                    "create_response": self.vad_create_response,
                    "interrupt_response": self.vad_interrupt_response,
                },
            }
            if not self.server_vad:
                audio_input["turn_detection"] = None
            if self.transcription_model is not None:
                audio_input["transcription"] = {"model": self.transcription_model}
            audio_input["noise_reduction"] = (
                {"type": self.input_noise_reduction} if self.input_noise_reduction is not None else None
            )
            audio_output: dict[str, Any] = {"format": output_format}
            if voice:
                audio_output["voice"] = voice
            await ws.send_str(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": system,
                            "output_modalities": ["audio"],
                            "audio": {"input": audio_input, "output": audio_output},
                            "tools": convert_realtime_tools(tools),
                        },
                    }
                )
            )
            session = OpenAIRealtimeSession(
                ws=ws,
                input_audio_media_type=input_audio_format,
                output_audio_media_type=output_audio_format,
                http_session=client,
                own_http_session=own_session,
            )
            await session.wait_ready(self.ready_timeout)
            return session
        except (Exception, asyncio.CancelledError):
            try:
                if ws is not None:
                    await ws.close()
            finally:
                if own_session:
                    await client.close()
            raise
