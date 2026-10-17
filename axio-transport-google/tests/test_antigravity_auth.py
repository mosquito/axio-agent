from __future__ import annotations

import asyncio
import base64
import hashlib
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import aiohttp
import pytest
from aiohttp import web
from axio.credentials import CredentialConflict, CredentialError

from axio_transport_google.antigravity import (
    AntigravityAccount,
    AntigravityFileCredentials,
    AntigravityOAuth,
    AntigravitySession,
    AntigravitySQLiteCredentials,
    OAuthConfig,
    OAuthError,
    SignInRequired,
    Tokens,
    list_accounts,
    save_account,
)


@asynccontextmanager
async def serve(handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def local_config(base: str) -> OAuthConfig:
    return OAuthConfig(
        client_id="test-client",
        client_secret="public-test-secret",
        token_url=base + "/token",
        userinfo_url=base + "/user",
        revoke_url=base + "/revoke",
    )


def account(subject: str = "alice", *, expired: bool = True) -> AntigravityAccount:
    return AntigravityAccount(
        subject,
        subject,
        "test-client",
        subject + "@example.test",
        Tokens("old-access", "private-refresh", time.time() + (-1 if expired else 3600)),
    )


@pytest.mark.parametrize("manual", [False, True])
async def test_sign_in_and_pkce(manual: bool) -> None:
    query: dict[str, list[str]] = {}
    callback = ""

    async def handler(request: web.Request) -> web.Response:
        if request.path == "/user":
            assert request.headers["Authorization"] == "Bearer access"
            return web.json_response({"id": "stable-google-subject", "email": "a@example.test"})
        form = await request.post()
        verifier = str(form["code_verifier"])
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        assert query["code_challenge"] == [challenge]
        assert form["code"] == "test-code"
        return web.json_response({"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    async with serve(handler) as base, aiohttp.ClientSession() as http:
        config = replace(local_config(base), redirect_uri=f"http://127.0.0.1:{port}/callback")
        oauth = AntigravityOAuth(http, config)

        async def open_browser(url: str) -> None:
            nonlocal query, callback
            query = parse_qs(urlsplit(url).query)
            callback = config.redirect_uri + "?" + urlencode({"code": "test-code", "state": query["state"][0]})
            if not manual:
                async with http.get(config.redirect_uri + "?state=wrong&code=bad") as response:
                    assert response.status == 400
                async with http.get(callback) as response:
                    assert response.status == 200
                    assert response.content_type == "text/html"
                    assert response.headers["Cache-Control"] == "no-store"
                    assert response.headers["Referrer-Policy"] == "no-referrer"
                    page = await response.text()
                    assert "ANTIGRAVITY SIGN-IN" in page
                    assert "Google account connected" in page
                    assert "Axio owl" in page
                    assert "Close tab (5)" in page
                    assert "prefers-color-scheme: dark" in page
                    assert "test-code" not in page
                    assert "stable-google-subject" not in page

        async def read_callback() -> str:
            return callback

        result = await oauth.sign_in(
            open_browser=open_browser, read_callback=read_callback if manual else None, manual=manual
        )
        assert result.subject == "stable-google-subject"
        assert result.account_id != config.client_id
        assert result.tokens is not None and result.tokens.refresh_token == "refresh"
        assert "refresh" not in repr(result)
        assert query["code_challenge_method"] == ["S256"]
        # The listener must be closed after successful authorization.
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))


@pytest.mark.parametrize("denied", [False, True])
async def test_browser_failure_page_does_not_claim_success(denied: bool) -> None:
    calls = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response({"error": "invalid_grant"}, status=400)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    async with serve(handler) as base, aiohttp.ClientSession() as http:
        config = replace(local_config(base), redirect_uri=f"http://127.0.0.1:{port}/callback")

        async def show(url: str) -> None:
            state = parse_qs(urlsplit(url).query)["state"][0]
            query = {"state": state, "error": "access_denied"} if denied else {"state": state, "code": "bad-code"}
            async with http.get(config.redirect_uri + "?" + urlencode(query)) as response:
                assert response.status == 400
                page = await response.text()
                assert "Sign-in didn't finish." in page
                assert 'main class="error"' in page
                assert "Google account connected" not in page
                assert "bad-code" not in page

        with pytest.raises(OAuthError):
            await AntigravityOAuth(http, config).sign_in(open_browser=show)
    assert calls == (0 if denied else 1)


@pytest.mark.parametrize(
    "query", ["state=wrong&code=x", "state={state}&error=access_denied", "state={state}&code=x&code=y"]
)
async def test_bad_manual_callback_never_exchanges(query: str) -> None:
    seen = ""
    async with aiohttp.ClientSession() as http:
        oauth = AntigravityOAuth(http)

        async def show(url: str) -> None:
            nonlocal seen
            seen = parse_qs(urlsplit(url).query)["state"][0]

        async def read() -> str:
            return oauth.config.redirect_uri + "?" + query.format(state=seen)

        with pytest.raises(OAuthError):
            await oauth.sign_in(open_browser=show, read_callback=read, manual=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_sign_in_timeout_or_cancel_cleans_listener(cancel: bool) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    opened = asyncio.Event()

    async def show(url: str) -> None:
        opened.set()

    async with aiohttp.ClientSession() as http:
        config = replace(OAuthConfig(), redirect_uri=f"http://127.0.0.1:{port}/callback")
        task = asyncio.create_task(AntigravityOAuth(http, config).sign_in(open_browser=show, timeout=0.1))
        await opened.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


@pytest.mark.parametrize("store_class", [AntigravityFileCredentials, AntigravitySQLiteCredentials])
async def test_shared_refresh_restart_and_account_isolation(tmp_path: Path, store_class: Any) -> None:
    path = tmp_path / "auth"
    store = store_class(path)
    await save_account(store, account())
    await save_account(store, account("bob"))
    calls = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return web.json_response({"access_token": "renewed", "expires_in": 3600})

    async with serve(handler) as base, aiohttp.ClientSession() as http:
        oauth = AntigravityOAuth(http, local_config(base))
        sessions = [AntigravitySession(store_class(path), oauth, "alice") for _ in range(4)]
        assert await asyncio.gather(*(session.access_token() for session in sessions)) == ["renewed"] * 4
        assert calls == 1
        async with store.lock("alice"):
            saved = await store.load("alice")
        assert saved.tokens.refresh_token == "private-refresh"
        async with store.lock("bob"):
            bob = await store.load("bob")
        assert bob.tokens.access_token == "old-access"
        assert all(item.tokens is None for item in await list_accounts(store))
        assert "private-refresh" not in repr(saved)
        # A new process can select and reuse the saved account without another exchange.
        await store.select_account("alice")
        reopened = store_class(path)
        assert await reopened.selected_account() == "alice"
        assert await AntigravitySession(reopened, oauth, "alice").access_token() == "renewed"
        assert calls == 1


@pytest.mark.parametrize(
    "error,status,cleared", [("invalid_grant", 400, True), ("temporarily_unavailable", 503, False)]
)
async def test_refresh_errors(tmp_path: Path, error: str, status: int, cleared: bool) -> None:
    store = AntigravityFileCredentials(tmp_path / "auth")
    original = account()
    await save_account(store, original)

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": error}, status=status)

    async with serve(handler) as base, aiohttp.ClientSession() as http:
        session = AntigravitySession(store, AntigravityOAuth(http, local_config(base)), "alice")
        with pytest.raises(SignInRequired if cleared else OAuthError):
            await session.access_token()
    async with store.lock("alice"):
        current = await store.load("alice")
    assert current is not None
    assert (current.tokens is None) is cleared
    if not cleared:
        assert current == original


async def test_refresh_save_failure_can_be_retried_without_another_exchange(tmp_path: Path) -> None:
    class FailingStore(AntigravityFileCredentials):
        fail = False

        async def save(self, record: AntigravityAccount) -> None:
            if self.fail:
                raise CredentialError("disk full")
            await super().save(record)

    store = FailingStore(tmp_path / "auth")
    await save_account(store, account())
    store.fail = True
    calls = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response({"access_token": "new", "refresh_token": "rotated", "expires_in": 3600})

    async with serve(handler) as base, aiohttp.ClientSession() as http:
        session = AntigravitySession(store, AntigravityOAuth(http, local_config(base)), "alice")
        with pytest.raises(CredentialError):
            await session.access_token()
        store.fail = False
        assert await session.access_token() == "new"
        assert calls == 1


async def test_login_conflict_and_logout_during_refresh(tmp_path: Path) -> None:
    store = AntigravityFileCredentials(tmp_path / "auth")
    original = account()
    await save_account(store, original)
    with pytest.raises(CredentialConflict):
        await save_account(store, account())
    started = asyncio.Event()
    finish = asyncio.Event()

    async def handler(request: web.Request) -> web.Response:
        if request.path == "/revoke":
            return web.Response(status=503)
        started.set()
        await finish.wait()
        return web.json_response({"access_token": "new", "expires_in": 3600})

    async with serve(handler) as base, aiohttp.ClientSession() as http:
        session = AntigravitySession(store, AntigravityOAuth(http, local_config(base)), "alice")
        refresh = asyncio.create_task(session.access_token())
        await started.wait()
        refresh.cancel()
        logout = asyncio.create_task(session.logout(revoke=True))
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await refresh
        assert await logout is False  # Local sign-out succeeded; remote revocation did not.
        with pytest.raises(SignInRequired):
            await session.access_token()
