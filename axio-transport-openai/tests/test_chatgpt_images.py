"""Native image requests through the shared account and real local HTTP server."""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio import HandlerError, Message, TextBlock, Tool
from axio.blocks import ImageBlock
from axio.models import Capability
from axio.transport import ImageEditTransport, ImageGenTransport
from axio_responses import Recovery, ResponsesInterruptedError

from axio_transport_openai import (
    CHATGPT_IMAGE_MODELS,
    ChatGPTImageModel,
    ChatGPTImageTools,
    ChatGPTImageTransport,
    ChatGPTTransport,
)
from axio_transport_openai.chatgpt_images import ChatGPTImageError, image_media_type
from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.credentials import CredentialRecord, save_account
from axio_transport_openai.oauth import CODEX_SCOPES, SCOPES, ChatGPTAccount, ChatGPTOAuth, ChatGPTTokens
from axio_transport_openai.session import ChatGPTSession, SignInRequired
from axio_transport_openai.subscription import CODEX_CLIENT_VERSION

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aE1cAAAAASUVORK5CYII=")
RESULT = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}


@dataclass
class Endpoint:
    body: Any = field(default_factory=lambda: RESULT)
    status: int = 200
    requests: list[dict[str, Any]] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    base_url: str = ""
    disconnect: bool = False
    pause: bool = False
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def images(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(await request.json())
        self.paths.append(request.path)
        self.headers.append(dict(request.headers))
        self.started.set()
        if self.pause:
            await self.release.wait()
        if self.disconnect:
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            response.content_length = 10000
            await response.prepare(request)
            await response.write(b'{"data":[')
            assert request.transport is not None
            request.transport.close()
            return response
        return web.json_response(self.body, status=self.status, headers={"X-Request-ID": "image-request"})

    async def models(self, request: web.Request) -> web.Response:
        assert request.query["client_version"] == CODEX_CLIENT_VERSION
        assert request.headers["ChatGPT-Account-ID"] == "workspace"
        return web.json_response({"models": [{"slug": "luna", "display_name": "Luna", "visibility": "list"}]})


@pytest.fixture
async def endpoint() -> AsyncIterator[Endpoint]:
    endpoint = Endpoint()
    app = web.Application()
    app.router.add_post("/images/generations", endpoint.images)
    app.router.add_post("/images/edits", endpoint.images)
    app.router.add_get("/models", endpoint.models)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    endpoint.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    try:
        yield endpoint
    finally:
        endpoint.release.set()
        await runner.cleanup()


@pytest.fixture
async def client(endpoint: Endpoint, tmp_path: Path) -> AsyncIterator[ChatGPTImageTransport]:
    now = time.time()
    account = ChatGPTAccount(
        "codex:workspace:subject",
        "host",
        "https://auth.openai.com",
        "subject",
        codex_account_id="workspace",
        tokens=ChatGPTTokens(
            "id-secret",
            "access-secret",
            "refresh-secret",
            "Bearer",
            frozenset(CODEX_SCOPES.split()),
            now,
            now + 3600,
            codex=True,
        ),
    )
    store = FileCredentialStore(tmp_path / "auth")
    await save_account(store, account)
    async with aiohttp.ClientSession() as http:
        session = ChatGPTSession(store, ChatGPTOAuth(http, "host"), account.client_id)
        try:
            yield ChatGPTImageTransport(account=session, session=http, base_url=endpoint.base_url)
        finally:
            await session.close()


async def test_image_model_catalog_and_request(client: ChatGPTImageTransport, endpoint: Endpoint) -> None:
    assert isinstance(client, ImageGenTransport)
    assert isinstance(client.model, ChatGPTImageModel)
    assert client.model.capabilities == frozenset({Capability.image_generation, Capability.image_editing})
    assert list(CHATGPT_IMAGE_MODELS.by_capability(Capability.image_generation).values()) == [client.model]
    images = await client.generate_images("An owl", size="1024x1536", quality="high", transparent_background=True)
    assert images == [PNG] and image_media_type(images[0]) == "image/png"
    assert endpoint.requests == [
        {
            "model": "gpt-image-2",
            "prompt": "An owl",
            "background": "transparent",
            "size": "1024x1536",
            "quality": "high",
        }
    ]
    assert endpoint.headers[0]["Authorization"] == "Bearer access-secret"
    assert endpoint.headers[0]["ChatGPT-Account-ID"] == "workspace"
    assert endpoint.headers[0]["originator"] == "axio"
    clone = client.with_model(client.model)
    assert clone.account is client.account and clone.session is client.session
    await clone.generate_images("Another owl")
    assert endpoint.headers[0]["x-codex-image-turn-id"] != endpoint.headers[1]["x-codex-image-turn-id"]


async def test_completion_catalog_and_native_tools(client: ChatGPTImageTransport, endpoint: Endpoint) -> None:
    transport = ChatGPTTransport(
        account=client.account, session=client.session, base_url=endpoint.base_url, image_base_url=endpoint.base_url
    )
    await transport.fetch_models()
    assert list(transport.models.keys()) == ["luna", "gpt-image-2"]
    assert transport.model is not None and transport.model.id == "luna"
    image_transport = transport.with_model("gpt-image-2")
    assert await image_transport.generate_images("An owl") == [PNG]
    with pytest.raises(ValueError, match="generate_images"):
        image_transport.build_payload([], [], "")

    async def echo(value: str) -> str:
        return value

    payload = transport.build_payload([Message("user", [TextBlock("Hi")])], [Tool("echo", echo)], "Help")
    assert payload["tools"][0]["name"] == "echo" and payload["tool_choice"] == "auto"
    assert payload["input"][0]["role"] == "user"
    assert not any(item.get("type") == "additional_tools" for item in payload["input"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": " "},
        {"n": 0},
        {"n": 2},
        {"n": True},
        {"model": "unknown"},
        {"size": "bogus"},
        {"quality": "bogus"},
        {"transparent_background": "yes"},
    ],
)
async def test_invalid_parameters_do_not_send(
    client: ChatGPTImageTransport, endpoint: Endpoint, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        await client.generate_images(**{"prompt": "owl", **kwargs})
    assert not endpoint.requests


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"data": []},
        {"data": [None]},
        {"data": [{"url": "https://invalid"}]},
        {"data": [{"b64_json": "!!!"}]},
        {"data": [{"b64_json": "dGV4dA=="}]},
        {"data": RESULT["data"] * 2},
    ],
)
async def test_missing_or_invalid_images_fail_without_replay(
    client: ChatGPTImageTransport, endpoint: Endpoint, body: Any
) -> None:
    endpoint.body = body
    with pytest.raises(ChatGPTImageError) as raised:
        await client.generate_images("owl")
    assert raised.value.status == 200 and raised.value.request_id == "image-request"
    assert len(endpoint.requests) == 1


@pytest.mark.parametrize(
    "status,code,recovery",
    [
        (401, "invalid_token", Recovery.authentication),
        (403, "denied", Recovery.permission),
        (400, "content_policy_violation", Recovery.invalid_request),
        (500, "server_error", Recovery.retry),
    ],
)
async def test_http_errors_never_repeat_generation(
    client: ChatGPTImageTransport, endpoint: Endpoint, status: int, code: str, recovery: Recovery
) -> None:
    endpoint.status = status
    endpoint.body = {"error": {"code": code, "message": "failed"}}
    with pytest.raises(ChatGPTImageError) as raised:
        await client.generate_images("owl")
    assert raised.value.code == code and raised.value.recovery is recovery
    assert len(endpoint.requests) == 1
    assert await client.account.access_token() == "access-secret"


@pytest.mark.parametrize("status", [200, 429])
async def test_quota_is_shared_until_explicit_resume(
    client: ChatGPTImageTransport, endpoint: Endpoint, status: int
) -> None:
    endpoint.status = status
    endpoint.body = {"error": {"type": "usage_limit_reached"}}
    for generator in (client, client.with_model("gpt-image-2")):
        with pytest.raises(ChatGPTImageError) as raised:
            await generator.generate_images("owl")
        assert raised.value.recovery is Recovery.quota
    assert len(endpoint.requests) == 1
    client.account.resume_plan_usage()
    endpoint.status, endpoint.body = 200, RESULT
    assert await client.generate_images("owl") == [PNG]


async def test_disconnect_has_unknown_outcome_without_replay(
    client: ChatGPTImageTransport, endpoint: Endpoint
) -> None:
    endpoint.disconnect = True
    with pytest.raises(ResponsesInterruptedError, match="outcome is unknown"):
        await client.generate_images("owl")
    assert len(endpoint.requests) == 1


async def test_cancellation_keeps_sessions_usable(client: ChatGPTImageTransport, endpoint: Endpoint) -> None:
    endpoint.pause = True
    task = asyncio.create_task(client.generate_images("owl"))
    await asyncio.wait_for(endpoint.started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(endpoint.requests) == 1
    endpoint.release.set()
    assert await client.generate_images("owl") == [PNG]
    assert not client.session.closed


async def test_public_plan_token_cannot_reach_image_endpoint(
    client: ChatGPTImageTransport, endpoint: Endpoint
) -> None:
    store = client.account.store
    now = time.time()
    public = ChatGPTAccount(
        "public-client",
        "host",
        "https://auth.openai.com",
        "subject",
        tokens=ChatGPTTokens("id", "public-token", "refresh", "Bearer", frozenset(SCOPES.split()), now, now + 3600),
    )
    await save_account(store, public)
    session = ChatGPTSession(store, ChatGPTOAuth(client.session, "host"), public.client_id)
    try:
        generator = ChatGPTImageTransport(account=session, session=client.session, base_url=endpoint.base_url)
        with pytest.raises(SignInRequired, match="login --codex"):
            await generator.generate_images("owl")
        assert not endpoint.requests
    finally:
        await session.close()


async def test_rotated_credentials_are_used_for_image_request(
    client: ChatGPTImageTransport, endpoint: Endpoint
) -> None:
    store, key = client.account.store, client.account.client_id
    async with store.lock(key):
        record = await store.load(key)
        assert record is not None and record.account.tokens is not None
        tokens = replace(record.account.tokens, received_at=time.time() - 10, expires_at=time.time() - 1)
        expired = replace(record.account, tokens=tokens)
        await store.save(CredentialRecord(expired))
    renewed = replace(
        expired,
        tokens=replace(
            tokens,
            access_token="renewed-access",
            refresh_token="renewed-refresh",
            received_at=time.time(),
            expires_at=time.time() + 3600,
        ),
    )
    with patch.object(ChatGPTOAuth, "refresh", new_callable=AsyncMock, return_value=renewed) as refresh:
        assert await client.generate_images("owl") == [PNG]
        refresh.assert_awaited_once()
    assert endpoint.headers[0]["Authorization"] == "Bearer renewed-access"
    async with store.lock(key):
        stored = await store.load(key)
    assert stored is not None and stored.account.tokens == renewed.tokens


async def test_edit_uses_sources_in_order_and_separate_route(
    client: ChatGPTImageTransport, endpoint: Endpoint
) -> None:
    assert isinstance(client, ImageEditTransport)
    png = ImageBlock(media_type="image/png", data=PNG)
    jpeg = ImageBlock(media_type="image/jpeg", data=b"\xff\xd8\xfftest")
    images = await client.edit_images(
        "Put the second image on the first",
        images=[png, jpeg],
        quality="low",
        size="1024x1024",
        transparent_background=True,
    )
    assert images == [PNG] and endpoint.paths == ["/images/edits"]
    assert endpoint.requests[0] == {
        "prompt": "Put the second image on the first",
        "model": "gpt-image-2",
        "size": "1024x1024",
        "quality": "low",
        "background": "transparent",
        "images": [
            {"image_url": "data:image/png;base64," + base64.b64encode(PNG).decode()},
            {"image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg.data).decode()},
        ],
    }
    assert endpoint.headers[0]["Authorization"] == "Bearer access-secret"
    assert endpoint.headers[0]["ChatGPT-Account-ID"] == "workspace"
    assert png.data == PNG
    transport = ChatGPTTransport(
        account=client.account, session=client.session, base_url=endpoint.base_url, image_base_url=endpoint.base_url
    )
    await transport.fetch_models()
    selected = transport.with_model("gpt-image-2")
    assert isinstance(selected, ImageEditTransport)
    assert await selected.edit_images("Make it red", images=[png]) == [PNG]
    assert endpoint.paths == ["/images/edits", "/images/edits"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"images": []},
        {"images": None},
        {"images": ["image.png"]},
        {"images": [ImageBlock(media_type="image/png", data=b"bad")]},
        {"images": [ImageBlock(media_type="image/jpeg", data=PNG)]},
        {"mask": ImageBlock(media_type="image/png", data=PNG)},
        {"prompt": ""},
        {"n": 2},
        {"model": "unknown"},
    ],
)
async def test_edit_invalid_input_never_falls_back_to_generation(
    client: ChatGPTImageTransport, endpoint: Endpoint, kwargs: dict[str, Any]
) -> None:
    arguments: dict[str, Any] = {"prompt": "Change it", "images": [ImageBlock(media_type="image/png", data=PNG)]}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        await client.edit_images(**arguments)
    assert not endpoint.requests


@pytest.mark.parametrize("failure", ["refusal", "invalid", "quota", "disconnect", "server"])
async def test_edit_failures_do_not_replay(client: ChatGPTImageTransport, endpoint: Endpoint, failure: str) -> None:
    if failure == "refusal":
        endpoint.status, endpoint.body = 400, {"error": {"code": "content_policy_violation"}}
    elif failure == "invalid":
        endpoint.body = {"data": []}
    elif failure == "quota":
        endpoint.status, endpoint.body = 429, {"error": {"type": "usage_limit_reached"}}
    elif failure == "server":
        endpoint.status, endpoint.body = 500, {"error": {"code": "server_error"}}
    else:
        endpoint.disconnect = True
    with pytest.raises((ChatGPTImageError, ResponsesInterruptedError)):
        await client.edit_images("Change it", images=[ImageBlock(media_type="image/png", data=PNG)])
    assert endpoint.paths == ["/images/edits"]
    if failure == "quota":
        with pytest.raises(ChatGPTImageError):
            await client.generate_images("A new image")
        assert len(endpoint.requests) == 1


async def test_edit_tool_cancellation_keeps_sources_without_partial_result(
    client: ChatGPTImageTransport, endpoint: Endpoint
) -> None:
    tools = ChatGPTImageTools(client, images={"source": ImageBlock(media_type="image/png", data=PNG)})
    endpoint.pause = True
    task = asyncio.create_task(tools.tools[1](prompt="Make red", images=["source"]))
    await asyncio.wait_for(endpoint.started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(tools.images) == ["source"]
    assert endpoint.paths == ["/images/edits"]
    endpoint.release.set()


async def test_separate_tools_schemas_and_generate_then_edit(
    client: ChatGPTImageTransport, endpoint: Endpoint
) -> None:
    tools = ChatGPTImageTools(client)
    draw, edit = tools.tools
    assert draw.name == "generate_image" and edit.name == "edit_image"
    assert draw.input_schema["required"] == ["prompt"]
    assert set(edit.input_schema["required"]) == {"prompt", "images"}
    assert edit.input_schema["properties"]["images"]["items"]["type"] == "string"
    assert "images" not in draw.input_schema["properties"]
    assert "mask" not in edit.input_schema["properties"]
    assert draw.access is not None and edit.access is not None
    assert [a.operation for a in edit.access({"images": ["source"]})] == ["external"]
    result = await draw(prompt="An owl")
    assert isinstance(result[0], TextBlock) and isinstance(result[1], ImageBlock)
    source_id = result[0].text.removeprefix("Image ID: ")
    assert tools.images[source_id] == result[1]
    edited = await edit(prompt="Add a red hat", images=[source_id])
    edited_id = edited[0].text.removeprefix("Image ID: ")
    assert edited_id != source_id
    assert tools.images[source_id] == result[1] and tools.images[edited_id] == edited[1]
    assert endpoint.paths == ["/images/generations", "/images/edits"]
    restored = ChatGPTImageTools(client, images=tools.images)
    assert dict(restored.images) == dict(tools.images)
    other = ChatGPTImageTools(client)
    with pytest.raises(ValueError, match="Unknown"):
        await other.edit_image("Change it", images=[source_id])
    assert len(endpoint.requests) == 2


async def test_edit_tool_requires_registered_sources(client: ChatGPTImageTransport, endpoint: Endpoint) -> None:
    tools = ChatGPTImageTools(client)
    image_id = tools.add_image(ImageBlock(media_type="image/png", data=PNG), image_id="attachment")
    assert image_id == "attachment"
    for sources in ([], ["unknown"], ["attachment", "unknown"]):
        with pytest.raises(HandlerError):
            await tools.tools[1](prompt="Change it", images=sources)
    with pytest.raises(ValueError, match="already exists"):
        tools.add_image(ImageBlock(media_type="image/png", data=PNG), image_id="attachment")
    assert not endpoint.requests
