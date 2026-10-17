from __future__ import annotations

import asyncio
import base64
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp_tiny_mcp import AuthorizationRequired
from yarl import URL

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.oauth import (
    MCPBrowserCallback,
    StoredOAuth,
    authorization_status,
    callback_response,
    credential_key,
    credential_store,
    open_oauth,
    sign_in,
    sign_out,
)
from axio_tools_mcp.session import MCPSession


@pytest.fixture
async def oauth_server() -> AsyncIterator[dict[str, Any]]:
    state: dict[str, Any] = {"refreshes": 0, "fail": False, "pages": []}

    async def resource(request: web.Request) -> web.Response:
        return web.Response(
            status=401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{state["base"]}/resource"'}
        )

    async def resource_metadata(request: web.Request) -> web.Response:
        return web.json_response({"resource": state["base"] + "/mcp", "authorization_servers": [state["base"]]})

    async def metadata(request: web.Request) -> web.Response:
        base = state["base"]
        return web.json_response(
            {
                "issuer": base,
                "authorization_endpoint": base + "/authorize",
                "token_endpoint": base + "/token",
                "registration_endpoint": base + "/register",
                "code_challenge_methods_supported": ["S256"],
                "response_types_supported": ["code"],
                "token_endpoint_auth_methods_supported": ["none"],
            }
        )

    async def register(request: web.Request) -> web.Response:
        data = await request.json()
        return web.json_response({**data, "client_id": "axio-fixture"})

    async def token(request: web.Request) -> web.Response:
        data = await request.post()
        assert data["resource"] == state["base"] + "/mcp"
        if state["fail"]:
            return web.json_response({"error": "invalid_grant", "error_description": "SECRET"}, status=400)
        if data["grant_type"] == "refresh_token":
            state["refreshes"] += 1
            assert data["refresh_token"] == "refresh-one"
            await asyncio.sleep(0.02)
            return web.json_response(
                {
                    "access_token": "access-two",
                    "token_type": "Bearer",
                    "refresh_token": "refresh-two",
                    "expires_in": 3600,
                }
            )
        verifier = str(data["code_verifier"])
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        assert challenge == state["challenge"]
        return web.json_response(
            {"access_token": "access-one", "token_type": "Bearer", "refresh_token": "refresh-one", "expires_in": 3600}
        )

    async def mcp(request: web.Request) -> web.Response:
        state["mcp_authorization"] = request.headers.get("Authorization")
        if state["mcp_authorization"] not in {"Bearer access-one", "Bearer access-two"}:
            return await resource(request)
        message = await request.json()
        if "id" not in message:
            return web.Response(status=202)
        result: dict[str, Any] = {"tools": []}
        if message["method"] == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "fixture", "version": "1"},
                "capabilities": {"tools": {}},
            }
        return web.json_response({"jsonrpc": "2.0", "id": message["id"], "result": result})

    app = web.Application()
    app.router.add_get("/mcp", resource)
    app.router.add_post("/mcp", mcp)
    app.router.add_get("/resource", resource_metadata)
    app.router.add_get("/.well-known/oauth-authorization-server", metadata)
    app.router.add_post("/register", register)
    app.router.add_post("/token", token)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    state["base"] = f"http://127.0.0.1:{runner.addresses[0][1]}"
    try:
        yield state
    finally:
        await runner.cleanup()


async def browser(state: dict[str, Any], url: str) -> None:
    query = URL(url).query
    state["challenge"] = query["code_challenge"]
    state["redirect"] = query["redirect_uri"]
    assert query["resource"] == state["base"] + "/mcp"
    callback = URL(query["redirect_uri"]).with_query(state=query["state"], code="fixture-code")
    async with aiohttp.ClientSession() as http:
        async with http.get(callback) as response:
            state["pages"].append((response.status, await response.text(), dict(response.headers)))


async def test_login_refresh_across_sessions_logout_and_callback_page(
    oauth_server: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = MCPServerConfig("fixture", url=oauth_server["base"] + "/mcp", oauth={})
    session = MCPSession(config)
    with pytest.raises(AuthorizationRequired):
        await session.connect()
    assert session.http is None and session.oauth is None
    await sign_in(config, lambda url: browser(oauth_server, url))
    await session.connect()
    try:
        assert await session.list_tools() == []
        assert oauth_server["mcp_authorization"] == "Bearer access-one"
    finally:
        await session.close()
    assert await authorization_status(config) == "Credentials saved"
    status, html, headers = oauth_server["pages"][0]
    assert status == 200 and "MCP authorization saved" in html
    assert "<svg" in html and "prefers-color-scheme: dark" in html
    assert "Close tab (5)" in html and "history.replaceState" in html
    assert "script-src 'nonce-" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    assert "access-one" not in html and "fixture-code" not in html
    store = credential_store()
    assert store.path.stat().st_mode & 0o777 == 0o600
    first, second = await open_oauth(config), await open_oauth(config)
    try:
        assert await first.token() == "access-one"
        results = await asyncio.gather(first.token(rejected="access-one"), second.token(rejected="access-one"))
        assert tuple(results) == ("access-two", "access-two")
        assert oauth_server["refreshes"] == 1
        await sign_out(config)
        assert await authorization_status(config) == "Sign-in required"
        with pytest.raises(AuthorizationRequired):
            await first.token()
    finally:
        await first.__aexit__()
        await second.__aexit__()
    # The stored callback and registration remain usable after a fresh login.
    previous = oauth_server["redirect"]
    await sign_in(config, lambda url: browser(oauth_server, url))
    assert oauth_server["redirect"] == previous


async def test_failed_exchange_returns_error_page(
    oauth_server: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    oauth_server["fail"] = True
    config = MCPServerConfig("fixture", url=oauth_server["base"] + "/mcp", oauth={})
    with pytest.raises(Exception, match="rejected"):
        await sign_in(config, lambda url: browser(oauth_server, url))
    status, html, _ = oauth_server["pages"][0]
    assert status == 400 and "Sign-in didn't finish" in html
    assert "SECRET" not in html and "MCP authorization saved" not in html
    assert await authorization_status(config) == "Sign-in required"


async def test_wrong_state_does_not_consume_attempt_and_cancellation_closes_listener(
    oauth_server: dict[str, Any], tmp_path: Path
) -> None:
    config = MCPServerConfig("fixture", url=oauth_server["base"] + "/mcp", oauth={})
    store = credential_store(tmp_path / "auth" / "mcp.auth")
    started = asyncio.Event()

    async def wait_browser(url: str) -> None:
        async with aiohttp.ClientSession() as http:
            query = URL(url).query
            async with http.get(URL(query["redirect_uri"]).with_query(state="wrong", code="bad")) as response:
                assert response.status == 400
        started.set()

    async with MCPBrowserCallback(open_browser=wait_browser) as callback:
        uri = callback.redirect_uri
        async with StoredOAuth(config, uri, store) as auth:
            task = asyncio.create_task(callback.login(auth))
            await asyncio.wait_for(started.wait(), 3)
            assert auth.pending
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not auth.pending
    async with aiohttp.ClientSession() as http:
        with pytest.raises(aiohttp.ClientConnectorError):
            await http.get(uri)
    async with store.lock(credential_key(config)):
        record = await store.load(credential_key(config))
        assert record and not record.credentials["tokens"]


def test_result_pages_have_fixed_text_and_no_remote_assets() -> None:
    for success in (True, False):
        response = callback_response(success)
        assert response.content_type == "text/html"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert "https:" not in response.headers["Content-Security-Policy"]


def test_callback_port_separates_registrations() -> None:
    from dataclasses import replace

    config = MCPServerConfig("fixture", url="https://example.com/mcp", oauth={})
    assert credential_key(config) == credential_key(replace(config, oauth={"redirect_port": 0}))
    assert credential_key(config) != credential_key(replace(config, oauth={"redirect_port": 8765}))
    assert credential_key(replace(config, oauth={"redirect_port": 8765})) != credential_key(
        replace(config, oauth={"redirect_port": 8766})
    )
