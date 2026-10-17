"""Subscription payloads, account catalogs, and complete agent tool loops."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field, replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio import Agent, MemoryContextStore
from axio.blocks import (
    AudioBlock,
    ImageBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from axio.events import IterationEnd, IterationStart, StreamEvent, TextDelta
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.tool import Tool
from axio.transport import CompletionTransport, EmbeddingTransport, RealtimeTransport
from axio.types import Usage
from axio_responses import ResponsesAPIError, ResponsesInterruptedError

from axio_transport_openai import ChatGPTTransport
from axio_transport_openai.chatgpt import build_chatgpt_payload
from axio_transport_openai.chatgpt_models import ChatGPTModel, parse_models
from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.credentials import save_account
from axio_transport_openai.oauth import SCOPES, ChatGPTAccount, ChatGPTOAuth, ChatGPTTokens
from axio_transport_openai.session import ChatGPTSession

MODEL = ModelSpec(
    "test-model",
    frozenset({Capability.text, Capability.vision, Capability.tool_use, Capability.reasoning}),
    context_window=64000,
    input_cost=2,
    output_cost=10,
)


async def echo(text: str) -> str:
    """Return text."""
    return text


def tool(name: str = "echo") -> Tool[Any]:
    return Tool(name=name, handler=echo)


def text_message(text: str = "hello", role: Literal["user", "assistant", "system"] = "user") -> Message:
    return Message(role=role, content=[TextBlock(text)])


def encode(*events: dict[str, Any]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def completed(text: str = "done") -> str:
    return encode(
        {"type": "response.created", "response": {"id": "resp-test", "model": "served-model"}},
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.completed",
            "response": {"status": "completed", "usage": {"input_tokens": 20, "output_tokens": 5}},
        },
    )


def tool_call(name: str = "echo") -> str:
    return encode(
        {"type": "response.created", "response": {"id": "resp-call", "model": "served-model"}},
        {"type": "response.reasoning_summary_text.delta", "delta": "thinking", "output_index": 0},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "reason-1", "encrypted_content": "opaque-proof"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "function_call", "name": name, "id": "fc-1", "call_id": "call-1"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-1",
            "output_index": 1,
            "delta": '{"text":"result"}',
        },
        {"type": "response.completed", "response": {"status": "completed", "output": [{"type": "function_call"}]}},
    )


def entry(slug: str, visibility: str = "list") -> dict[str, Any]:
    return {"slug": slug, "display_name": f"Display {slug}", "visibility": visibility}


def api_entry(slug: str) -> dict[str, Any]:
    return entry(slug) | {
        "input_modalities": ["text", "image"],
        "supports_parallel_tool_calls": True,
        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
        "context_window": 500_000,
        "max_context_window": 872_000,
    }


@dataclass
class Server:
    base_url: str = ""
    catalogs: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {
            "token-a": [entry("test-model"), api_entry("gpt-5.6"), entry("hidden", "hidden"), entry("unknown")],
            "token-b": [entry("second-account-model")],
        }
    )
    model_status: int = 200
    model_body: Any = None
    model_requests: list[str] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    status: int = 200
    error: dict[str, Any] = field(default_factory=dict)

    async def models(self, request: web.Request) -> web.Response:
        token = request.headers["Authorization"].removeprefix("Bearer ")
        self.model_requests.append(token)
        body = self.model_body if self.model_body is not None else {"models": self.catalogs[token]}
        return web.json_response(body, status=self.model_status, headers={"x-request-id": "req-models"})

    async def responses(self, request: web.Request) -> web.Response:
        self.tokens.append(request.headers["Authorization"])
        self.payloads.append(await request.json())
        if self.status != 200:
            return web.json_response(self.error, status=self.status, headers={"x-request-id": "req-inference"})
        body = self.replies.pop(0) if self.replies else completed()
        return web.Response(text=body, content_type="text/event-stream", headers={"x-request-id": "req-inference"})


@pytest.fixture
async def server() -> AsyncIterator[Server]:
    server = Server()
    app = web.Application()
    app.router.add_get("/v1/models", server.models)
    app.router.add_post("/v1/responses", server.responses)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    server.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}/v1"
    try:
        yield server
    finally:
        await runner.cleanup()


async def account(http: aiohttp.ClientSession, tmp_path: Path, key: str) -> ChatGPTSession:
    now = time.time()
    store = FileCredentialStore(tmp_path / f"{key}.auth")
    registered = ChatGPTAccount(
        key,
        await store.host_id(),
        "https://auth.openai.com",
        f"subject-{key}",
        tokens=ChatGPTTokens(
            "id-proof", f"token-{key}", "refresh", "Bearer", frozenset(SCOPES.split()), now, now + 3600
        ),
    )
    await save_account(store, registered)
    return ChatGPTSession(store, ChatGPTOAuth(http, registered.host_id), key)


@pytest.fixture
async def transport(server: Server, tmp_path: Path) -> AsyncIterator[ChatGPTTransport]:
    async with aiohttp.ClientSession() as http:
        session = await account(http, tmp_path, "a")
        try:
            yield ChatGPTTransport(
                account=session, session=http, model_specs=[MODEL], base_url=server.base_url, retry_base_delay=0
            )
        finally:
            await session.close()


def test_payload_omits_forbidden_fields_and_converts_system_role() -> None:
    messages = [text_message("prior rule", "system"), text_message()]
    payload = build_chatgpt_payload(messages, [tool()], "system prompt", MODEL, reasoning_effort="high")
    assert set(payload) == {"model", "input", "instructions", "stream", "store", "include", "reasoning"}
    assert payload["store"] is False and payload["stream"] is True
    assert payload["instructions"] == "system prompt"
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"] == {"summary": "auto", "effort": "high"}
    declarations = payload["input"][0]
    assert declarations["type"] == "additional_tools" and declarations["role"] == "developer"
    assert declarations["tools"][0]["name"] == "echo"
    assert "namespace" not in declarations["tools"][0]
    assert payload["input"][1]["role"] == "developer"
    assert messages[0].role == "system"


def test_tool_selection_changes_do_not_rewrite_history() -> None:
    history = [
        text_message(),
        Message(role="assistant", content=[ToolUseBlock("c", "old", {"text": "x"})]),
        Message(role="user", content=[ToolResultBlock("c", "x")]),
    ]
    first = build_chatgpt_payload(history, [tool("old")], "", MODEL)
    second = build_chatgpt_payload(history, [tool("new")], "", MODEL)
    without = build_chatgpt_payload(history, [], "", MODEL)
    assert first["input"][1:] == second["input"][1:] == without["input"]
    assert second["input"][0]["tools"][0]["name"] == "new"
    assert all(item.get("type") != "additional_tools" for item in without["input"])


@pytest.mark.parametrize(
    "block,role",
    [
        (AudioBlock("audio/wav", b"audio"), "user"),
        (VideoBlock("video/mp4", b"video"), "user"),
        (ImageBlock("image/png", b"image"), "assistant"),
        (ToolResultBlock("c", [AudioBlock("audio/wav", b"audio")]), "user"),
        (ToolResultBlock("c", [VideoBlock("video/mp4", b"video")]), "user"),
        (ProviderBlock("openai", "image_generation_call", {"type": "image_generation_call"}), "assistant"),
        (ProviderBlock("openai", "additional_tools", {"type": "additional_tools"}), "assistant"),
    ],
)
def test_unsupported_media_and_hosted_history_are_rejected(block: Any, role: Any) -> None:
    with pytest.raises(ValueError, match="does not support"):
        build_chatgpt_payload([Message(role=role, content=[block])], [], "", MODEL)


def test_images_and_encrypted_reasoning_survive_conversion() -> None:
    history = [
        Message(role="user", content=[ImageBlock("image/png", b"image")]),
        Message(
            role="assistant",
            content=[
                ReasoningBlock(signature="opaque", id="reason", provider="openai"),
                ToolUseBlock("c", "echo", {"text": "x"}),
            ],
        ),
        Message(role="user", content=[ToolResultBlock("c", [ImageBlock("image/png", b"output")])]),
    ]
    items = build_chatgpt_payload(history, [tool()], "", MODEL)["input"]
    assert items[1]["content"][0]["image_url"].startswith("data:image/png;base64,")
    assert items[2] == {"type": "reasoning", "id": "reason", "encrypted_content": "opaque", "summary": []}
    assert items[3]["call_id"] == items[4]["call_id"] == "c"
    assert items[4]["output"][0]["type"] == "input_image"


def test_unverified_capabilities_are_not_enabled() -> None:
    unknown = ModelSpec("unknown")
    with pytest.raises(ValueError, match="Tool use"):
        build_chatgpt_payload([], [tool()], "", unknown)
    with pytest.raises(ValueError, match="Image input"):
        build_chatgpt_payload([Message(role="user", content=[ImageBlock("image/png", b"x")])], [], "", unknown)
    with pytest.raises(ValueError, match="Reasoning"):
        build_chatgpt_payload([], [], "", unknown, reasoning_effort="high")


async def test_account_catalog_order_visibility_metadata_and_prices(
    transport: ChatGPTTransport, server: Server
) -> None:
    await transport.fetch_models()
    assert list(transport.models.keys()) == ["test-model", "gpt-5.6", "unknown"]
    assert [item.id for item in transport.catalog] == ["test-model", "gpt-5.6", "hidden", "unknown"]
    assert transport.model is transport.models["test-model"]
    selected = transport.model
    assert isinstance(selected, ChatGPTModel)
    assert selected.display_name == "Display test-model"
    assert selected.context_window == 64000 and Capability.tool_use in selected.capabilities
    assert math.isnan(selected.input_cost) and math.isnan(selected.output_cost)
    assert transport.models["unknown"].capabilities == {Capability.text, Capability.tool_use}
    assert Capability.tool_use in transport.models["gpt-5.6"].capabilities
    assert server.model_requests == ["token-a"]
    with pytest.raises(ValueError):
        transport.with_model("hidden")


@pytest.mark.parametrize("levels", [["low"], ["none", "medium"], ["future-effort"]])
def test_payload_enforces_catalog_efforts(levels: list[str]) -> None:
    model = parse_models(
        {
            "models": [
                entry("test")
                | {
                    "supported_reasoning_levels": [{"effort": effort} for effort in levels],
                }
            ]
        },
        {},
    )[0]
    for effort in levels:
        assert build_chatgpt_payload([], [], "", model, reasoning_effort=effort)["reasoning"]["effort"] == effort
    assert "effort" not in build_chatgpt_payload([], [], "", model)["reasoning"]
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        build_chatgpt_payload([], [], "", model, reasoning_effort="xhigh")


async def test_programmatic_effort_is_revalidated_after_model_change(
    transport: ChatGPTTransport,
    server: Server,
) -> None:
    server.catalogs["token-a"] = [
        api_entry("test-model"),
        entry("low-only")
        | {
            "supported_reasoning_levels": [{"effort": "low"}],
        },
    ]
    await transport.fetch_models()
    transport.reasoning_effort = "high"
    assert transport.build_payload([], [], "")["reasoning"]["effort"] == "high"
    selected = transport.with_model("low-only")
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        async for _ in selected.stream([text_message()], [], ""):
            pass
    assert not server.payloads


async def test_two_turn_agent_replays_tools_reasoning_and_history(transport: ChatGPTTransport, server: Server) -> None:
    server.replies = [tool_call(), completed("finished"), completed("follow-up")]
    context = MemoryContextStore()
    agent = Agent(system="system", transport=transport, tools=[tool()])
    assert await agent.run("first", context) == "finished"
    assert await agent.run("second", context) == "follow-up"
    assert len(server.payloads) == 3 and len(server.model_requests) == 1
    first, second, third = server.payloads
    assert first["input"][0] == second["input"][0] == third["input"][0]
    items = second["input"]
    assert any(item.get("encrypted_content") == "opaque-proof" for item in items)
    call = next(item for item in items if item.get("type") == "function_call")
    result = next(item for item in items if item.get("type") == "function_call_output")
    assert call["call_id"] == result["call_id"] == "call-1"
    assert result["output"] == "result"
    assert len(third["input"]) > len(second["input"])
    assert transport.last_usage == Usage(20, 5)
    assert server.tokens == ["Bearer token-a"] * 3


@pytest.mark.parametrize("model_id", ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "future-model"])
@pytest.mark.parametrize("with_metadata", [True, False])
async def test_current_model_catalog_preserves_agent_tool_declarations(
    transport: ChatGPTTransport, server: Server, model_id: str, with_metadata: bool
) -> None:
    server.catalogs["token-a"] = [api_entry(model_id) if with_metadata else entry(model_id)]
    server.replies = [tool_call(), completed("finished")]
    await transport.fetch_models()
    model = transport.model
    assert model is not None
    assert model.context_window == (500_000 if with_metadata else ModelSpec(model_id).context_window)
    assert model.max_output_tokens == ModelSpec(model_id).max_output_tokens
    assert math.isnan(model.input_cost) and math.isnan(model.output_cost)
    agent = Agent(system="Use the provided tool.", transport=transport, tools=[tool()])
    assert await agent.run("echo result", MemoryContextStore()) == "finished"
    assert len(server.payloads) == 2
    for payload in server.payloads:
        assert payload["model"] == model_id
        declaration = payload["input"][0]
        assert declaration["type"] == "additional_tools" and declaration["role"] == "developer"
        assert declaration["tools"][0]["name"] == "echo"
    assert any(item.get("output") == "result" for item in server.payloads[1]["input"])


async def test_selector_changes_current_declaration(transport: ChatGPTTransport, server: Server) -> None:
    class Selector:
        async def select(self, messages: Iterable[Message], tools: Iterable[Tool[Any]]) -> Iterable[Tool[Any]]:
            choices = list(tools)
            return [choices[1]] if any(m.role == "assistant" for m in messages) else [choices[0]]

    server.replies = [tool_call("old"), completed()]
    agent = Agent(system="", transport=transport, tools=[tool("old"), tool("new")], selector=Selector())
    await agent.run("go", MemoryContextStore())
    assert server.payloads[0]["input"][0]["tools"][0]["name"] == "old"
    assert server.payloads[1]["input"][0]["tools"][0]["name"] == "new"


async def test_model_clone_observes_rotated_credentials(transport: ChatGPTTransport, server: Server) -> None:
    await transport.fetch_models()
    clone = transport.with_model("gpt-5.6")
    assert clone.account is transport.account and clone.session is transport.session
    assert clone.models is transport.models
    async with transport.account.store.lock("a"):
        record = await transport.account.store.load("a")
        assert record is not None and record.account.tokens is not None
        tokens = replace(record.account.tokens, access_token="rotated-token")
        await transport.account.store.save(replace(record, account=replace(record.account, tokens=tokens)))
    _ = [event async for event in clone.stream([text_message()], [], "")]
    assert server.tokens == ["Bearer rotated-token"]
    assert server.payloads[0]["model"] == "gpt-5.6"
    assert transport.model is not None and transport.model.id == "test-model"


async def test_account_switch_isolates_catalog_and_old_transport(
    transport: ChatGPTTransport,
    server: Server,
    tmp_path: Path,
) -> None:
    await transport.fetch_models()
    other = await account(transport.session, tmp_path, "b")
    try:
        switched = transport.with_account(other)
        assert not switched.models
        await switched.fetch_models()
        assert list(switched.models.keys()) == ["second-account-model"]
        assert list(transport.models.keys()) == ["test-model", "gpt-5.6", "unknown"]
        _ = [event async for event in switched.stream([text_message()], [], "")]
        _ = [event async for event in transport.stream([text_message()], [], "")]
        assert server.tokens == ["Bearer token-b", "Bearer token-a"]
    finally:
        await other.close()


@pytest.mark.parametrize("status", [401, 403, 503])
async def test_failed_refresh_invalidates_catalog_without_fallback(
    transport: ChatGPTTransport,
    server: Server,
    status: int,
) -> None:
    await transport.fetch_models()
    clone = transport.with_model("test-model")
    server.model_status = status
    server.model_body = {"detail": "unavailable"}
    with pytest.raises(ResponsesAPIError) as raised:
        await transport.fetch_models()
    assert raised.value.status == status and raised.value.request_id == "req-models"
    assert not transport.models and not clone.models and not transport.catalog
    assert await transport.account.access_token() == "token-a"
    assert len(server.model_requests) == (4 if status == 503 else 2)


async def test_catalog_refresh_does_not_substitute_a_removed_selection(
    transport: ChatGPTTransport, server: Server
) -> None:
    await transport.fetch_models()
    server.catalogs["token-a"] = [entry("gpt-5.6")]
    await transport.fetch_models()
    with pytest.raises(ValueError, match="Select a visible model"):
        _ = [event async for event in transport.stream([], [], "")]
    assert not server.payloads


async def test_interrupted_inference_is_not_replayed(transport: ChatGPTTransport, server: Server) -> None:
    server.replies = [encode({"type": "response.output_text.delta", "delta": "partial"})]
    events: list[StreamEvent] = []
    with pytest.raises(ResponsesInterruptedError):
        async for event in transport.stream([text_message()], [], ""):
            events.append(event)
    assert isinstance(events[0], IterationStart)
    assert isinstance(events[1], TextDelta)
    assert not any(isinstance(event, IterationEnd) for event in events)
    assert len(server.payloads) == 1


async def test_inference_uses_shared_quota_recovery(transport: ChatGPTTransport, server: Server) -> None:
    await transport.fetch_models()
    server.status = 429
    server.error = {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}
    for selected in (transport, transport.with_model("gpt-5.6")):
        with pytest.raises(ResponsesAPIError):
            _ = [event async for event in selected.stream([], [], "")]
    assert len(server.payloads) == 1


async def test_concurrent_histories_do_not_share_tool_declarations(
    transport: ChatGPTTransport, server: Server
) -> None:
    async def run(name: str) -> None:
        _ = [event async for event in transport.stream([text_message(name)], [tool(name)], "")]

    await asyncio.gather(run("one"), run("two"))
    assert len(server.model_requests) == 1
    for payload in server.payloads:
        assert payload["input"][0]["tools"][0]["name"] == payload["input"][1]["content"][0]["text"]


def test_entry_point_and_protocol_surface(transport: ChatGPTTransport) -> None:
    selected = entry_points(group="axio.transport", name="chatgpt")
    assert len(selected) == 1 and next(iter(selected)).load() is ChatGPTTransport
    assert isinstance(transport, CompletionTransport)
    assert not isinstance(transport, EmbeddingTransport)
    assert not isinstance(transport, RealtimeTransport)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": []},
        {"models": [entry("x"), entry("x")]},
        {"models": [{}]},
        {"models": [entry("x") | {"display_name": None}]},
    ],
)
def test_malformed_catalog_is_not_silently_replaced(body: Any) -> None:
    with pytest.raises(ValueError):
        parse_models(body, {})


def test_api_fields_override_explicit_fallback_metadata() -> None:
    fallback = ModelSpec("future-model", MODEL.capabilities, context_window=64_000, max_output_tokens=32_000)
    body = {
        "models": [
            api_entry("future-model")
            | {"input_modalities": ["text"], "supported_reasoning_levels": [], "max_output_tokens": 16_000}
        ]
    }
    model = parse_models(body, {fallback.id: fallback})[0]
    assert model.capabilities == {Capability.text, Capability.tool_use}
    assert model.context_window == 500_000 and model.max_output_tokens == 16_000


@pytest.mark.parametrize("flag", [False, None, "true", 1, [], {}])
def test_parallel_tool_flag_does_not_disable_catalog_model_tools(flag: Any) -> None:
    model = parse_models({"models": [entry("future-model") | {"supports_parallel_tool_calls": flag}]}, {})[0]
    assert Capability.tool_use in model.capabilities


def test_serial_tool_support_can_use_explicit_fallback() -> None:
    model = parse_models({"models": [entry(MODEL.id) | {"supports_parallel_tool_calls": False}]}, {MODEL.id: MODEL})[0]
    assert Capability.tool_use in model.capabilities


@pytest.mark.parametrize("limit", [True, 0, -1, "500000", None])
def test_invalid_api_token_limit_uses_generic_default(limit: Any) -> None:
    model = parse_models({"models": [entry("future-model") | {"context_window": limit}]}, {})[0]
    assert model.context_window == ModelSpec("future-model").context_window


async def test_capabilities_are_refreshed_from_api_without_static_fallback(
    transport: ChatGPTTransport, server: Server
) -> None:
    server.catalogs["token-a"] = [api_entry("gpt-6-luna")]
    await transport.fetch_models()
    assert transport.model is not None and Capability.tool_use in transport.model.capabilities
    server.catalogs["token-a"] = [entry("gpt-6-luna")]
    await transport.fetch_models()
    assert transport.model is not None
    assert transport.model.capabilities == {Capability.text, Capability.tool_use}


async def test_model_request_retries_transient_error_with_backoff(transport: ChatGPTTransport, server: Server) -> None:
    server.model_status = 503
    with patch("axio_transport_openai.subscription.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(ResponsesAPIError):
            await transport.fetch_models()
    assert sleep.await_count == 2


async def test_unsupported_input_never_reaches_inference(transport: ChatGPTTransport, server: Server) -> None:
    with pytest.raises(ValueError, match="AudioBlock"):
        _ = [
            event
            async for event in transport.stream(
                [Message(role="user", content=[AudioBlock("audio/wav", b"audio")])], [], ""
            )
        ]
    assert not server.payloads


async def test_empty_catalog_does_not_use_known_models(transport: ChatGPTTransport, server: Server) -> None:
    server.catalogs["token-a"] = []
    await transport.fetch_models()
    assert not transport.models and transport.model is None
    with pytest.raises(ValueError, match="Select a visible model"):
        _ = [event async for event in transport.stream([], [], "")]
    assert not server.payloads


async def test_model_connection_errors_have_bounded_recovery(transport: ChatGPTTransport) -> None:
    with patch.object(transport.session, "get", side_effect=aiohttp.ClientConnectionError) as get:
        with pytest.raises(ResponsesInterruptedError):
            await transport.fetch_models()
    assert get.call_count == 3
    assert not transport.models


async def test_stream_close_closes_executor(transport: ChatGPTTransport) -> None:
    from collections.abc import AsyncGenerator

    from axio_transport_openai.subscription import ChatGPTResponses

    await transport.fetch_models()
    closed = False

    async def events(self: ChatGPTResponses, payload: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        nonlocal closed
        try:
            yield TextDelta(0, "partial")
            await asyncio.Future[None]()
        finally:
            closed = True

    with patch.object(ChatGPTResponses, "stream", events):
        stream = transport.stream([text_message()], [], "")
        assert isinstance(await anext(stream), IterationStart)
        await stream.aclose()
    assert closed


async def test_served_model_and_single_terminal_event(transport: ChatGPTTransport) -> None:
    events = [event async for event in transport.stream([text_message()], [], "")]
    assert isinstance(events[0], IterationStart) and events[0].model == "served-model"
    assert isinstance(events[-1], IterationEnd)
    assert sum(isinstance(event, IterationEnd) for event in events) == 1


async def test_build_payload_requires_catalog_and_does_not_mutate_it(transport: ChatGPTTransport) -> None:
    with pytest.raises(ValueError):
        transport.build_payload([], [], "")
    await transport.fetch_models()
    first = transport.build_payload([text_message()], [tool()], "")
    first["input"][0]["tools"][0]["name"] = "changed"
    second = transport.build_payload([text_message()], [tool()], "")
    assert second["input"][0]["tools"][0]["name"] == "echo"


@pytest.mark.parametrize("outcome", ["success", "missing", "catalog_error", "cancel"])
async def test_account_helper_closes_session_on_every_exit(
    transport: ChatGPTTransport, server: Server, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from axio_transport_openai import accounts
    from axio_transport_openai.credentials import CredentialError

    store = transport.account.store
    assert isinstance(store, FileCredentialStore)
    await store.select_account("a")
    opened: list[ChatGPTTransport] = []

    def make_transport(*, account: ChatGPTSession, session: aiohttp.ClientSession) -> ChatGPTTransport:
        result = ChatGPTTransport(account=account, session=session, base_url=server.base_url, max_attempts=1)
        opened.append(result)
        return result

    monkeypatch.setattr(accounts, "ChatGPTTransport", make_transport)
    if outcome == "catalog_error":
        server.model_status = 403
    model = "missing" if outcome == "missing" else "gpt-5.6"

    async def use() -> None:
        async with accounts.open_chatgpt(store, transport.session, model=model) as selected:
            assert selected.account is opened[0].account
            assert selected.model is not None and selected.model.id == model
            if outcome == "cancel":
                raise asyncio.CancelledError

    if outcome == "success":
        await use()
    else:
        expected = {"missing": CredentialError, "catalog_error": ResponsesAPIError, "cancel": asyncio.CancelledError}
        with pytest.raises(expected[outcome]):
            await use()
    assert not transport.session.closed
    with pytest.raises(RuntimeError, match="closed"):
        await opened[0].account.access_token()
    assert await store.selected_account() == "a"
    async with store.lock("a"):
        record = await store.load("a")
    assert record is not None and record.account.tokens is not None


def test_application_contract_preserves_explicit_account_selection() -> None:
    from axio.transport import TransportEntry

    entry = TransportEntry.from_target("chatgpt", ChatGPTTransport)
    assert entry.auth is not None and not entry.auto_select
    assert not entry.supports_temperature and not entry.supports_max_tokens and not entry.supports_debug
