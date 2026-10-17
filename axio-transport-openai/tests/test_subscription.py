"""Subscription recovery through a real local HTTP endpoint and credential store."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio.events import IterationEnd, StreamEvent, TextDelta
from axio_responses import Recovery, ResponsesAPIError, ResponsesIncompleteError, ResponsesInterruptedError

from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.credentials import save_account
from axio_transport_openai.oauth import SCOPES, ChatGPTAccount, ChatGPTOAuth, ChatGPTTokens
from axio_transport_openai.session import ChatGPTSession, PlanConsentRequired, SignInRequired
from axio_transport_openai.subscription import ChatGPTResponses


def sse(*events: dict[str, Any]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


COMPLETED = {"type": "response.completed", "response": {"status": "completed"}}
TEXT = {"type": "response.output_text.delta", "delta": "hello"}
QUOTA = "subscription_sharing_usage_limit_exceeded"
TRANSIENT = "subscription_sharing_usage_unavailable"


@dataclass
class Endpoint:
    replies: list[tuple[int, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    base_url: str = ""
    retry_after: str = "0"
    disconnect: bool = False

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(await request.json())
        self.headers.append(dict(request.headers))
        status, body = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        headers = {"X-Request-ID": f"req-{len(self.requests)}", "Retry-After": self.retry_after}
        if self.disconnect:
            response = web.StreamResponse(headers={**headers, "Content-Type": "text/event-stream"})
            response.content_length = 10000
            await response.prepare(request)
            await response.write(str(body).encode())
            assert request.transport is not None
            request.transport.close()
            return response
        if status == 200:
            return web.Response(text=body, content_type="text/event-stream", headers=headers)
        return web.json_response(body, status=status, headers=headers)


@pytest.fixture
async def endpoint() -> AsyncIterator[Endpoint]:
    endpoint = Endpoint()
    app = web.Application()
    app.router.add_post("/v1/responses", endpoint.handle)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    endpoint.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}/v1"
    try:
        yield endpoint
    finally:
        await runner.cleanup()


@pytest.fixture
async def client(endpoint: Endpoint, tmp_path: Path) -> AsyncIterator[ChatGPTResponses]:
    now = time.time()
    account = ChatGPTAccount(
        "issued-client",
        "host",
        "https://auth.openai.com",
        "subject",
        tokens=ChatGPTTokens(
            "id-secret", "access-secret", "refresh-secret", "Bearer", frozenset(SCOPES.split()), now, now + 3600
        ),
    )
    store = FileCredentialStore(tmp_path / "auth")
    await save_account(store, account)
    async with aiohttp.ClientSession() as http:
        session = ChatGPTSession(store, ChatGPTOAuth(http, "host"), account.client_id)
        try:
            yield ChatGPTResponses(session, http, endpoint.base_url, retry_base_delay=0)
        finally:
            await session.close()


async def collect(client: ChatGPTResponses) -> list[StreamEvent]:
    return [event async for event in client.stream({"stream": True, "store": False, "input": "hello"})]


@pytest.mark.parametrize(
    "status,body,recovery",
    [
        (429, {"error": {"code": QUOTA, "message": "limit"}}, Recovery.quota),
        (403, {"error": {"code": "subscription_sharing_user_not_eligible"}}, Recovery.permission),
        (
            400,
            {"error": {"code": "subscription_sharing_unsupported_capability", "param": "tools"}},
            Recovery.invalid_request,
        ),
        (401, {"error": {"code": "subscription_sharing_invalid_user"}}, Recovery.authentication),
        (403, {"error": {"code": "chatpass_v2_scope_not_authorized"}}, Recovery.permission),
        (401, {"detail": "identity or scope rejected"}, Recovery.authentication),
        (403, {"detail": ["region unavailable"]}, Recovery.permission),
    ],
)
async def test_terminal_admission_errors_keep_credentials(
    client: ChatGPTResponses,
    endpoint: Endpoint,
    status: int,
    body: Any,
    recovery: Recovery,
) -> None:
    endpoint.replies = [(status, body)]
    with pytest.raises(ResponsesAPIError) as raised:
        await collect(client)
    assert len(endpoint.requests) == 1
    error = raised.value
    assert error.status == status and error.body == body
    assert error.request_id == "req-1" and error.recovery is recovery
    assert await client.account.access_token() == "access-secret"
    assert endpoint.headers[0]["Authorization"] == "Bearer access-secret"


async def test_quota_pauses_shared_session_until_explicit_resume(client: ChatGPTResponses, endpoint: Endpoint) -> None:
    endpoint.replies = [(429, {"error": {"code": QUOTA}}), (200, sse(TEXT, COMPLETED))]
    with pytest.raises(ResponsesAPIError):
        await collect(client)
    with pytest.raises(ResponsesAPIError) as raised:
        await collect(replace(client))
    assert raised.value is client.account.plan_usage_error
    assert len(endpoint.requests) == 1
    client.account.resume_plan_usage()
    events = await collect(client)
    assert isinstance(events[-1], IterationEnd)
    assert len(endpoint.requests) == 2


@pytest.mark.parametrize("body", [{"detail": "routing unavailable"}, {"error": {"code": TRANSIENT}}])
async def test_temporary_admission_retries_with_backoff(
    client: ChatGPTResponses,
    endpoint: Endpoint,
    body: Any,
) -> None:
    endpoint.replies = [(503, body), (200, sse(TEXT, COMPLETED))]
    endpoint.retry_after = "2"
    with patch("axio_transport_openai.subscription.asyncio.sleep", new_callable=AsyncMock) as sleep:
        events = await collect(client)
    sleep.assert_awaited_once_with(2)
    assert [e.delta for e in events if isinstance(e, TextDelta)] == ["hello"]
    assert len(endpoint.requests) == 2
    assert await client.account.access_token() == "access-secret"


async def test_retry_budget_is_bounded(client: ChatGPTResponses, endpoint: Endpoint) -> None:
    endpoint.replies = [(503, {"detail": "unavailable"})]
    with pytest.raises(ResponsesAPIError) as raised:
        await collect(client)
    assert len(endpoint.requests) == client.max_attempts
    assert raised.value.request_id == "req-3"
    assert await client.account.access_token() == "access-secret"


@pytest.mark.parametrize("code", [QUOTA, TRANSIENT])
@pytest.mark.parametrize("kind", ["response.failed", "error"])
@pytest.mark.parametrize("prefix", [TEXT, {"type": "response.created", "response": {"id": "resp-1"}}])
async def test_stream_errors_never_replay_emitted_events(
    client: ChatGPTResponses,
    endpoint: Endpoint,
    code: str,
    kind: str,
    prefix: dict[str, Any],
) -> None:
    details = {"code": code, "param": "model", "message": "failed"}
    failure = (
        {"type": kind, "response": {"error": details}} if kind == "response.failed" else {"type": kind, **details}
    )
    endpoint.replies = [(200, sse(prefix, failure))]
    events: list[StreamEvent] = []
    with pytest.raises(ResponsesAPIError) as raised:
        async for event in client.stream({}):
            events.append(event)
    assert len(events) == 1 and not any(isinstance(e, IterationEnd) for e in events)
    assert len(endpoint.requests) == 1
    assert raised.value.code == code and raised.value.param == "model"
    assert raised.value.status == 200 and raised.value.request_id == "req-1"
    assert raised.value.body == failure
    assert await client.account.access_token() == "access-secret"


async def test_transient_stream_failure_before_events_can_retry(client: ChatGPTResponses, endpoint: Endpoint) -> None:
    endpoint.replies = [(200, sse({"type": "error", "code": TRANSIENT})), (200, sse(TEXT, COMPLETED))]
    events = await collect(client)
    assert len(endpoint.requests) == 2
    assert isinstance(events[-1], IterationEnd)


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", "unknown"])
async def test_incomplete_is_not_success_or_retry(client: ChatGPTResponses, endpoint: Endpoint, reason: str) -> None:
    incomplete = {
        "type": "response.incomplete",
        "response": {"status": "incomplete", "incomplete_details": {"reason": reason}},
    }
    endpoint.replies = [(200, sse(TEXT, incomplete))]
    with pytest.raises(ResponsesIncompleteError) as raised:
        await collect(client)
    assert raised.value.body == incomplete
    assert raised.value.event_type == "response.incomplete"
    assert len(endpoint.requests) == 1


@pytest.mark.parametrize("prefix", [TEXT, {"type": "response.refusal.delta", "delta": "Cannot"}])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_interrupted_stream_keeps_request_id_without_replay(
    client: ChatGPTResponses,
    endpoint: Endpoint,
    prefix: dict[str, Any],
    disconnect: bool,
) -> None:
    endpoint.replies = [(200, sse(prefix))]
    endpoint.disconnect = disconnect
    with pytest.raises(ResponsesInterruptedError) as raised:
        await collect(client)
    assert raised.value.request_id == "req-1"
    assert len(endpoint.requests) == 1


@pytest.mark.parametrize("identity_only", [False, True])
async def test_missing_permission_and_signin_have_distinct_errors(
    client: ChatGPTResponses,
    endpoint: Endpoint,
    identity_only: bool,
) -> None:
    async with client.account.store.lock(client.account.client_id):
        record = await client.account.store.load(client.account.client_id)
        assert record is not None and record.account.tokens is not None
        tokens = replace(record.account.tokens, scopes=frozenset({"openid"})) if identity_only else None
        await client.account.store.save(replace(record, account=replace(record.account, tokens=tokens)))
    with pytest.raises(PlanConsentRequired if identity_only else SignInRequired):
        await collect(client)
    assert not endpoint.requests


async def test_cancellation_is_not_retried(client: ChatGPTResponses, endpoint: Endpoint) -> None:
    endpoint.replies = [(503, {"detail": "unavailable"})]
    with patch("axio_transport_openai.subscription.asyncio.sleep", side_effect=asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await collect(client)
    assert len(endpoint.requests) == 1


async def test_openai_http_quota_error_is_structured(client: ChatGPTResponses, endpoint: Endpoint) -> None:
    from axio_transport_openai import OpenAITransport

    body = {"error": {"code": QUOTA, "param": "model", "message": "limit"}}
    endpoint.replies = [(429, body)]
    transport = OpenAITransport(api_key="api-key", base_url=endpoint.base_url, session=client.http, api="responses")
    with pytest.raises(ResponsesAPIError) as raised:
        _ = [event async for event in transport.stream([], [], "")]
    assert len(endpoint.requests) == 1
    assert raised.value.body == body and raised.value.request_id == "req-1"
    assert endpoint.headers[0]["Authorization"] == "Bearer api-key"


async def test_openai_stream_error_metadata_survives_agent_run(client: ChatGPTResponses, endpoint: Endpoint) -> None:
    from axio import Agent, MemoryContextStore

    from axio_transport_openai import OpenAITransport

    failure = {"type": "response.failed", "response": {"error": {"code": QUOTA, "message": "limit"}}}
    endpoint.replies = [(200, sse(TEXT, failure))]
    transport = OpenAITransport(api_key="api-key", base_url=endpoint.base_url, session=client.http, api="responses")
    agent = Agent(system="", transport=transport)
    with pytest.raises(ResponsesAPIError) as raised:
        await agent.run("hello", MemoryContextStore())
    assert raised.value.code == QUOTA and raised.value.body == failure
    assert raised.value.status == 200 and raised.value.request_id == "req-1"
    assert len(endpoint.requests) == 1
