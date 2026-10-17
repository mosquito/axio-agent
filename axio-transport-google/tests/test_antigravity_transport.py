from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio import Agent, MemoryContextStore, Message, TextBlock, Tool
from axio.events import IterationEnd, IterationStart, ReasoningSignature, Refusal, TextDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.models import Capability, ModelSpec
from axio.transport import CompletionTransport, ImageGenTransport, RealtimeTransport, VideoGenTransport
from axio.types import StopReason
from test_antigravity_auth import account, local_config, serve

from axio_transport_google import GoogleTransport, VertexAITransport
from axio_transport_google.antigravity import (
    AntigravityAPIError,
    AntigravityFileCredentials,
    AntigravityOAuth,
    AntigravitySession,
    AntigravityTransport,
    save_account,
)

MODEL = "gemini-test"
CATALOG = {
    "models": {
        MODEL: {
            "displayName": "Gemini Test",
            "supportsThinking": True,
            "supportsImages": True,
            "supportsVideo": True,
            "supportedMimeTypes": {"audio/wav": True},
            "maxTokens": 1000000,
            "maxOutputTokens": 65000,
            "modelExperiments": {"untrustedPrompt": "must not enter request"},
        },
        "claude-test": {},
        "gemini-test-image": {},
    },
    "defaultAgentModelId": MODEL,
}


def stream_response(*chunks: dict[str, Any]) -> web.Response:
    return web.Response(
        text="".join("data: " + json.dumps({"response": chunk}) + "\n\n" for chunk in chunks),
        content_type="text/event-stream",
    )


def chunk(parts: list[dict[str, Any]], *, finish: str = "STOP") -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": parts}}
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate]}


@asynccontextmanager
async def transport_at(
    path: Path, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> AsyncIterator[AntigravityTransport]:
    store = AntigravityFileCredentials(path / "auth")
    await save_account(store, account(expired=False))
    async with serve(handler) as base, aiohttp.ClientSession() as http:
        session = AntigravitySession(store, AntigravityOAuth(http, local_config(base)), "alice")
        yield AntigravityTransport(session=http, account=session, base_url=base, retry_base_delay=0)


async def test_catalog_envelopes_and_shared_account(tmp_path: Path) -> None:
    calls: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(request.path)
        assert request.headers["Authorization"] == "Bearer old-access"
        if request.path.endswith("loadCodeAssist"):
            assert await request.json() == {"metadata": {"ideType": "ANTIGRAVITY"}}
            return web.json_response({"cloudaicompanionProject": {"id": "project"}})
        assert await request.json() == {"project": "project"}
        return web.json_response(CATALOG)

    async with transport_at(tmp_path, handler) as transport:
        await transport.fetch_models()
        assert transport.models.ids() == [MODEL]
        assert transport.model.context_window == 1000000
        assert transport.model.max_output_tokens == 65000
        assert transport.model.capabilities == frozenset(
            {
                Capability.text,
                Capability.tool_use,
                Capability.reasoning,
                Capability.vision,
                Capability.video,
                Capability.audio,
            }
        )
        assert "modelExperiments" not in transport.catalog[MODEL]
        selected = transport.with_model(transport.model)
        assert selected.account is transport.account
        assert selected._reader is transport._reader
        assert selected._session_id == transport._session_id
        assert isinstance(selected, CompletionTransport)
        assert not isinstance(selected, (ImageGenTransport, VideoGenTransport, RealtimeTransport))
        body = selected.build_payload([Message("user", [TextBlock("hello")])], [], "system")
        assert body["project"] == "project" and body["model"] == MODEL
        assert body["request"]["systemInstruction"]["parts"][0]["text"] == "system"
        assert len(calls) == 2


@pytest.mark.parametrize(
    "profile,catalog,message",
    [
        ({}, CATALOG, "Complete account setup"),
        ({"cloudaicompanionProject": "p"}, {"models": {}}, "No Gemini"),
        ({"cloudaicompanionProject": "p"}, {}, "no model catalogue"),
    ],
)
async def test_catalog_failure_is_not_static_fallback(
    tmp_path: Path, profile: dict[str, Any], catalog: dict[str, Any], message: str
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(profile if request.path.endswith("loadCodeAssist") else catalog)

    async with transport_at(tmp_path, handler) as transport:
        with pytest.raises(StreamError, match=message):
            await transport.fetch_models()
        assert not transport.models


async def test_two_tool_turns_preserve_parallel_signatures_and_usage(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    async def add(value: int) -> str:
        return str(value + 1)

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        assert request.query["alt"] == "sse"
        if len(requests) == 1:
            return stream_response(
                chunk(
                    [
                        {"text": "think", "thought": True, "thoughtSignature": "reason-proof"},
                        {"functionCall": {"name": "add", "args": {"value": 1}}, "thoughtSignature": "first-proof"},
                        {"functionCall": {"name": "add", "args": {"value": 5}}, "thoughtSignature": "second-proof"},
                    ]
                )
            )
        response = chunk([{"text": "2 and 6"}])
        response["modelVersion"] = "actual-serving-model"
        response["usageMetadata"] = {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 2,
            "cachedContentTokenCount": 3,
            "toolUsePromptTokenCount": 1,
            "totalTokenCount": 17,
        }
        return stream_response(response)

    async with transport_at(tmp_path, handler) as transport:
        transport.project = "project"
        transport.model = ModelSpec(MODEL, frozenset({Capability.text, Capability.tool_use, Capability.reasoning}))
        agent = Agent(system="system", transport=transport, tools=[Tool(name="add", handler=add)])
        events = [event async for event in agent.run_stream("use add twice", MemoryContextStore())]
        ends = [e for e in events if isinstance(e, IterationEnd)]
        starts = [e for e in events if isinstance(e, IterationStart)]
        assert len(ends) == len(starts) == 2
        assert starts[0].model == MODEL and starts[1].model == "actual-serving-model"
        assert ends[0].stop_reason == StopReason.tool_use
        assert ends[1].usage.input_tokens == 11 and ends[1].usage.output_tokens == 6
        assert ends[1].usage.cache_read_tokens == 3 and ends[1].usage.reasoning_tokens == 2
        assert len([e for e in events if isinstance(e, ToolUseStart)]) == 2
        assert len([e for e in events if isinstance(e, ReasoningSignature)]) == 1
        replay = requests[1]["request"]["contents"]
        parts = next(c["parts"] for c in replay if c["role"] == "model")
        assert [p["thoughtSignature"] for p in parts] == ["reason-proof", "first-proof", "second-proof"]
        assert requests[0]["request"]["sessionId"] == requests[1]["request"]["sessionId"]
        assert requests[0]["requestId"] != requests[1]["requestId"]


@pytest.mark.parametrize(
    "response,reason",
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, StopReason.refusal),
        (chunk([], finish="SAFETY"), StopReason.refusal),
        (chunk([{"text": "ok"}]), StopReason.end_turn),
    ],
)
async def test_terminal_events(tmp_path: Path, response: dict[str, Any], reason: StopReason) -> None:
    async def handler(request: web.Request) -> web.Response:
        return stream_response(response)

    async with transport_at(tmp_path, handler) as transport:
        transport.project, transport.model = "p", ModelSpec(MODEL)
        events = [e async for e in transport.stream([], [], "")]
        assert isinstance(events[0], IterationStart)
        assert isinstance(events[-1], IterationEnd) and events[-1].stop_reason == reason
        assert sum(isinstance(e, IterationEnd) for e in events) == 1
        assert sum(isinstance(e, Refusal) for e in events) == (reason == StopReason.refusal)


@pytest.mark.parametrize(
    "status,category,attempts",
    [
        (400, "request", 1),
        (401, "authentication", 2),
        (403, "permission", 1),
        (429, "quota", 3),
        (503, "transient", 3),
    ],
)
async def test_classified_errors_and_bounded_retry(tmp_path: Path, status: int, category: str, attempts: int) -> None:
    count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal count
        count += 1
        return web.json_response(
            {"error": {"status": "DENIED", "message": "secret"}}, status=status, headers={"Retry-After": "0"}
        )

    async with transport_at(tmp_path, handler) as transport:
        assert transport.account is not None
        with patch.object(transport.account, "access_token", AsyncMock(return_value="token")) as token:
            with pytest.raises(AntigravityAPIError) as caught:
                await transport.fetch_models()
            assert caught.value.category == category and caught.value.status == status
            assert "secret" not in str(caught.value)
            assert count == attempts
            assert token.await_count == (2 if status == 401 else 1)


async def test_retry_after_then_success(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.json_response({}, status=429, headers={"Retry-After": "7"})
        return stream_response(chunk([{"text": "ok"}]))

    async with transport_at(tmp_path, handler) as transport:
        transport.project, transport.model = "p", ModelSpec(MODEL)
        with patch("axio_transport_google.antigravity.transport.asyncio.sleep", AsyncMock()) as sleep:
            events = [e async for e in transport.stream([], [], "")]
            sleep.assert_awaited_once_with(7.0)
        assert sum(isinstance(e, TextDelta) for e in events) == 1


@pytest.mark.parametrize(
    "response", [chunk([{"text": "partial"}], finish=""), chunk([], finish="MISSING_THOUGHT_SIGNATURE")]
)
async def test_no_retry_after_events_or_terminal_error(tmp_path: Path, response: dict[str, Any]) -> None:
    calls = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return stream_response(response)

    async with transport_at(tmp_path, handler) as transport:
        transport.project, transport.model = "p", ModelSpec(MODEL)
        events = []
        with pytest.raises(StreamError):
            async for event in transport.stream([], [], ""):
                events.append(event)
        assert calls == 1 and not any(isinstance(e, IterationEnd) for e in events)


@pytest.mark.parametrize("transport_class", [GoogleTransport, VertexAITransport, AntigravityTransport])
def test_mcp_json_schema_preserves_nullable_types_and_keywords(transport_class: Any) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "value": {"type": ["string", "null"]},
            "nested": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"count": {"type": ["integer", "null"], "minimum": 1}},
            },
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    async def handler(**kwargs: Any) -> str:
        return "ok"

    tool: Tool[Any] = Tool(name="mcp", handler=handler, schema=MappingProxyType(schema))
    transport = transport_class()
    body = (
        transport.build_payload([], [tool], "")["request"]
        if isinstance(transport, AntigravityTransport)
        else transport._build_request_body([], [tool], "")
    )
    declaration = body["tools"][0]["functionDeclarations"][0]
    assert "parameters" not in declaration
    assert declaration["parametersJsonSchema"] == schema
    assert tool.input_schema == schema
