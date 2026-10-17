"""Exercise browser callbacks and signed ID tokens against a local OAuth server."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import aiohttp
import jwt
import pytest
from aiohttp import web
from cryptography.hazmat.primitives.asymmetric import rsa

from axio_transport_openai.oauth import (
    CALLBACK_PATH,
    DYNAMIC_CLIENT_ID,
    PLAN_SCOPE,
    RESOURCE,
    SCOPES,
    ChatGPTAccount,
    ChatGPTOAuth,
    OAuthDenied,
    OAuthError,
    OAuthValidationError,
)


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeOAuth:
    def __init__(self, key: rsa.RSAPrivateKey, session: aiohttp.ClientSession) -> None:
        self.key = key
        self.session = session
        self.issuer = ""
        self.client_id = "oaiapp_test"
        self.kid = "key-1"
        self.authorizations: list[dict[str, str]] = []
        self.exchanges: list[dict[str, str]] = []
        self.callback_changes: dict[str, str | None] = {}
        self.claim_changes: dict[str, Any] = {}
        self.token_changes: dict[str, Any] = {}
        self.discovery_changes: dict[str, Any] = {}
        self.drop_claims: set[str] = set()
        self.bad_signature = False
        self.algorithm = "RS256"
        self.jwks_body: object | None = None
        self.token_body: object | None = None
        self.token_status = 200
        self.token_delay = False
        self.exchange_started = asyncio.Event()
        self.invalid_grants = 0
        self.jwks_requests = 0
        self.callback_url = ""
        self.callback_status = 0
        self.callback_html = ""
        self.callback_headers: dict[str, str] = {}
        self.duplicate_callback = False
        self.duplicate_status = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/.well-known/openid-configuration", self.discovery)
        app.router.add_get("/authorize", self.authorize)
        app.router.add_post("/token", self.token)
        app.router.add_get("/jwks", self.jwks)
        return app

    async def discovery(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "issuer": self.issuer,
                "authorization_endpoint": self.issuer + "/authorize",
                "token_endpoint": self.issuer + "/token",
                "jwks_uri": self.issuer + "/jwks",
                **self.discovery_changes,
            }
        )

    async def authorize(self, request: web.Request) -> web.Response:
        params = dict(request.query)
        self.authorizations.append(params)
        assert params["response_type"] == "code"
        assert params["resource"] == RESOURCE
        assert params["scope"] == SCOPES
        assert params["code_challenge_method"] == "S256"
        callback = {
            "state": params["state"],
            "code": f"code-{len(self.authorizations)}",
            "scope": SCOPES,
        }
        if params["client_id"] == DYNAMIC_CLIENT_ID:
            callback["client_id"] = self.client_id
        for key, value in self.callback_changes.items():
            if value is None:
                callback.pop(key, None)
            else:
                callback[key] = value
        self.callback_url = params["redirect_uri"] + "?" + urlencode(callback)
        raise web.HTTPFound(self.callback_url)

    async def token(self, request: web.Request) -> web.Response:
        assert request.content_type == "application/x-www-form-urlencoded"
        assert "Authorization" not in request.headers
        form = {key: str(value) for key, value in (await request.post()).items()}
        self.exchanges.append(form)
        assert form["client_id"] == self.client_id
        assert form["resource"] == RESOURCE
        params = self.authorizations[-1]
        if form["grant_type"] == "refresh_token":
            assert set(form) == {"grant_type", "client_id", "refresh_token", "resource"}
            assert form["refresh_token"] == "refresh-secret"
        else:
            assert set(form) == {"grant_type", "code", "client_id", "code_verifier", "redirect_uri", "resource"}
            assert form["grant_type"] == "authorization_code"
            assert form["redirect_uri"] == params["redirect_uri"]
            assert form["code"] == f"code-{len(self.authorizations)}"
            digest = hashlib.sha256(form["code_verifier"].encode()).digest()
            assert base64.urlsafe_b64encode(digest).decode().rstrip("=") == params["code_challenge"]
        self.exchange_started.set()
        if self.token_delay:
            await asyncio.Event().wait()
        if self.invalid_grants:
            self.invalid_grants -= 1
            return web.json_response({"error": "invalid_grant"}, status=400)
        if self.token_body is not None:
            return web.json_response(self.token_body, status=self.token_status)
        now = int(time.time())
        claims = {
            "iss": self.issuer,
            "aud": form["client_id"],
            "sub": "subject-1",
            "email": "owl@example.test",
            "name": "Owl",
            "nonce": params["nonce"],
            "iat": now,
            "exp": now + 3600,
            **self.claim_changes,
        }
        for name in self.drop_claims:
            claims.pop(name, None)
        if form["grant_type"] == "refresh_token":
            claims.pop("nonce", None)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048) if self.bad_signature else self.key
        if self.algorithm == "RS256":
            token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": self.kid})
        else:
            token = jwt.encode(claims, "s" * 32, algorithm=self.algorithm, headers={"kid": self.kid})
        return web.json_response(
            {
                "id_token": token,
                "access_token": "opaque-access-secret",
                "refresh_token": "refresh-secret",
                "scope": SCOPES,
                "token_type": "Bearer",
                "expires_in": 3600,
                **self.token_changes,
            }
        )

    async def jwks(self, request: web.Request) -> web.Response:
        self.jwks_requests += 1
        if self.jwks_body is not None:
            return web.json_response(self.jwks_body)
        public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        return web.json_response({"keys": [{**public, "kid": self.kid, "use": "sig", "alg": "RS256"}]})

    async def browser(self, url: str) -> None:
        async with self.session.get(url) as response:
            self.callback_status = response.status
            self.callback_html = await response.text()
            self.callback_headers = dict(response.headers)
        if self.duplicate_callback:
            async with self.session.get(self.callback_url) as response:
                self.duplicate_status = response.status


@pytest.fixture
async def server(signing_key: rsa.RSAPrivateKey) -> AsyncIterator[FakeOAuth]:
    async with aiohttp.ClientSession() as session:
        server = FakeOAuth(signing_key, session)
        runner = web.AppRunner(server.app(), access_log=None, shutdown_timeout=0.1)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            server.issuer = f"http://127.0.0.1:{runner.addresses[0][1]}"
            yield server
        finally:
            await runner.cleanup()


@pytest.fixture
def oauth(server: FakeOAuth, monkeypatch: pytest.MonkeyPatch) -> ChatGPTOAuth:
    # A protocol regression must fail promptly instead of waiting for a person.
    monkeypatch.setattr(
        ChatGPTOAuth.sign_in,
        "__kwdefaults__",
        {"open_browser": server.browser, "read_callback": None, "timeout": 5, "port": 0},
    )
    return ChatGPTOAuth(session=server.session, host_id="urn:uuid:stable-host", issuer=server.issuer)


async def assert_callback_closed(server: FakeOAuth) -> None:
    async with asyncio.timeout(2):
        with pytest.raises(aiohttp.ClientConnectorError):
            async with server.session.get(server.authorizations[-1]["redirect_uri"]):
                pass


async def test_initial_registration(server: FakeOAuth, oauth: ChatGPTOAuth, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="aiohttp.access")
    account = await oauth.sign_in(open_browser=server.browser)
    assert account.client_id == server.client_id
    assert account.subject == "subject-1"
    assert account.issuer == server.issuer
    assert account.host_id == oauth.host_id
    assert account.email == "owl@example.test"
    assert account.name == "Owl"
    assert account.tokens is not None
    assert account.tokens.plan_enabled
    assert account.tokens.access_token == "opaque-access-secret"
    assert account.tokens.refresh_token == "refresh-secret"
    assert account.tokens.expires_at == account.tokens.received_at + 3600
    params = server.authorizations[0]
    assert params["agent_name_hint"] == "Axio"
    assert params["client_id"] == DYNAMIC_CLIENT_ID
    assert params["ext_agent_host_id"] == oauth.host_id
    assert "id_token_hint" not in params and "login_hint" not in params
    redirect = urlsplit(params["redirect_uri"])
    assert redirect.hostname == "127.0.0.1" and redirect.path == CALLBACK_PATH and redirect.port
    assert server.callback_status == 200
    assert "You're signed in." in server.callback_html
    assert "Axio owl" in server.callback_html
    assert "ChatGPT plan connected" in server.callback_html
    assert server.callback_headers["Cache-Control"] == "no-store"
    assert "default-src 'none'" in server.callback_headers["Content-Security-Policy"]
    script_nonce = re.search(r'<script nonce="([^"]+)"', server.callback_html)
    assert script_nonce is not None
    assert f"'nonce-{script_nonce[1]}'" in server.callback_headers["Content-Security-Policy"]
    assert "Close tab (5)" in server.callback_html
    for secret in (account.tokens.id_token, "opaque-access-secret", "refresh-secret", params["state"], "code-1"):
        assert secret not in repr(account)
        assert secret not in repr(account.tokens)
        assert secret not in server.callback_html
        assert secret not in caplog.text
    await assert_callback_closed(server)


@pytest.mark.parametrize("signed_out", [False, True])
async def test_returning_account(server: FakeOAuth, oauth: ChatGPTOAuth, signed_out: bool) -> None:
    first = await oauth.sign_in(open_browser=server.browser)
    selected = replace(first, tokens=None) if signed_out else first
    second = await oauth.sign_in(selected, open_browser=server.browser)
    assert second.client_id == first.client_id and second.subject == first.subject
    assert second.tokens is not first.tokens
    before, after = server.authorizations
    assert after["client_id"] == first.client_id
    assert after["ext_agent_host_id"] == before["ext_agent_host_id"]
    assert after["login_hint"] == first.email
    assert "agent_name_hint" not in after
    if signed_out:
        assert "id_token_hint" not in after
    else:
        assert first.tokens is not None and after["id_token_hint"] == first.tokens.id_token
    for name in ("state", "nonce", "code_challenge"):
        assert before[name] != after[name]
    assert server.jwks_requests == 1
    await assert_callback_closed(server)


@pytest.mark.parametrize("client_id", [None, "", DYNAMIC_CLIENT_ID])
async def test_missing_issued_client_id(server: FakeOAuth, oauth: ChatGPTOAuth, client_id: str | None) -> None:
    server.callback_changes["client_id"] = client_id
    with pytest.raises(OAuthValidationError, match="issued client ID"):
        await oauth.sign_in(open_browser=server.browser)
    assert not server.exchanges
    assert "You're signed in." not in server.callback_html
    await assert_callback_closed(server)


async def test_changed_returning_client_id(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    account = await oauth.sign_in(open_browser=server.browser)
    server.callback_changes["client_id"] = "oaiapp_other"
    with pytest.raises(OAuthValidationError, match="changed.*client ID"):
        await oauth.sign_in(account, open_browser=server.browser)
    assert len(server.exchanges) == 1
    assert account.client_id == "oaiapp_test"
    await assert_callback_closed(server)


@pytest.mark.parametrize("state", [None, "wrong", "неверное"])
async def test_invalid_state_precedes_denial(server: FakeOAuth, oauth: ChatGPTOAuth, state: str | None) -> None:
    server.callback_changes = {"state": state, "error": "access_denied"}
    with pytest.raises(OAuthValidationError, match="state"):
        await oauth.sign_in(open_browser=server.browser)
    assert not server.exchanges
    await assert_callback_closed(server)


async def test_denied_consent(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    server.callback_changes = {"error": "access_denied", "error_description": "secret-do-not-display"}
    with pytest.raises(OAuthDenied) as info:
        await oauth.sign_in(open_browser=server.browser)
    assert info.value.code == "access_denied"
    assert "secret-do-not-display" not in str(info.value)
    assert "secret-do-not-display" not in server.callback_html
    assert not server.exchanges
    await assert_callback_closed(server)


@pytest.mark.parametrize("changes", [{"code": None}, {"code": ""}, {"error": "unknown-secret"}])
async def test_invalid_callback(server: FakeOAuth, oauth: ChatGPTOAuth, changes: dict[str, str | None]) -> None:
    server.callback_changes = changes
    with pytest.raises(OAuthError):
        await oauth.sign_in(open_browser=server.browser)
    assert not server.exchanges
    await assert_callback_closed(server)


@pytest.mark.parametrize(
    "claims",
    [
        {"nonce": "wrong"},
        {"nonce": 10},
        {"iss": "https://other.example"},
        {"aud": "oaiapp_other"},
        {"exp": 1},
        {"iat": 9999999999},
        {"sub": ""},
        {"sub": 42},
        {"aud": ["oaiapp_test", "other"]},
        {"azp": "other"},
    ],
)
async def test_invalid_claims(server: FakeOAuth, oauth: ChatGPTOAuth, claims: dict[str, Any]) -> None:
    server.claim_changes = claims
    with pytest.raises(OAuthValidationError):
        await oauth.sign_in(open_browser=server.browser)
    assert server.callback_status == 400
    assert "You're signed in." not in server.callback_html
    await assert_callback_closed(server)


@pytest.mark.parametrize("claim", ["iss", "aud", "sub", "exp", "iat", "nonce"])
async def test_missing_claims(server: FakeOAuth, oauth: ChatGPTOAuth, claim: str) -> None:
    server.drop_claims.add(claim)
    with pytest.raises(OAuthValidationError):
        await oauth.sign_in(open_browser=server.browser)


async def test_invalid_signature(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    server.bad_signature = True
    with pytest.raises(OAuthValidationError, match="verification failed"):
        await oauth.sign_in(open_browser=server.browser)
    assert "You're signed in." not in server.callback_html
    await assert_callback_closed(server)


async def test_reject_symmetric_algorithm(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    server.algorithm = "HS256"
    with pytest.raises(OAuthValidationError, match="algorithm"):
        await oauth.sign_in(open_browser=server.browser)
    assert server.jwks_requests == 0


async def test_returning_identity_cannot_change(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    account = await oauth.sign_in(open_browser=server.browser)
    server.claim_changes = {"sub": "other-subject"}
    with pytest.raises(OAuthValidationError, match="selected account"):
        await oauth.sign_in(account, open_browser=server.browser)
    assert account.subject == "subject-1" and account.tokens is not None
    assert account.tokens.access_token == "opaque-access-secret"
    assert "You're signed in." not in server.callback_html
    await assert_callback_closed(server)


async def test_email_is_only_a_label(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    account = await oauth.sign_in(open_browser=server.browser)
    server.claim_changes = {"email": "changed@example.test"}
    new = await oauth.sign_in(account, open_browser=server.browser)
    assert new.subject == account.subject and new.email != account.email


async def test_explicit_returning_client_id(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    account = await oauth.sign_in(open_browser=server.browser)
    server.callback_changes["client_id"] = account.client_id
    assert (await oauth.sign_in(account, open_browser=server.browser)).client_id == account.client_id


async def test_duplicate_callback_parameters(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    async def browser(url: str) -> None:
        async with server.session.get(url, allow_redirects=False) as response:
            callback = response.headers["Location"] + "&state=duplicate"
        async with server.session.get(callback) as response:
            assert response.status == 400

    with pytest.raises(OAuthValidationError, match="Duplicate"):
        await oauth.sign_in(open_browser=browser)
    assert not server.exchanges
    await assert_callback_closed(server)


async def test_browser_disconnect_does_not_discard_account(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    async def browser(url: str) -> None:
        async with server.session.get(url, allow_redirects=False) as response:
            callback = urlsplit(response.headers["Location"])
        reader, writer = await asyncio.open_connection(callback.hostname, callback.port)
        writer.write(f"GET {callback.path}?{callback.query} HTTP/1.1\r\nHost: {callback.netloc}\r\n\r\n".encode())
        await writer.drain()
        await server.exchange_started.wait()
        writer.close()
        await writer.wait_closed()

    account = await oauth.sign_in(open_browser=browser)
    assert account.subject == "subject-1"
    await assert_callback_closed(server)


@pytest.mark.parametrize("identity_only", [True, False])
async def test_missing_plan_permission(server: FakeOAuth, oauth: ChatGPTOAuth, identity_only: bool) -> None:
    server.token_changes = {"scope": "openid email profile"}
    if identity_only:
        server.token_changes.update(access_token=None, refresh_token=None, token_type=None, expires_in=None)
    account = await oauth.sign_in(open_browser=server.browser)
    assert account.tokens is not None and not account.tokens.plan_enabled
    assert PLAN_SCOPE not in account.tokens.scopes
    assert server.callback_status == 200
    assert "You're signed in." in server.callback_html
    assert "ChatGPT plan access is not enabled" in server.callback_html


@pytest.mark.parametrize(
    "changes",
    [
        {"id_token": None},
        {"id_token": "not-a-jwt"},
        {"expires_in": None},
        {"expires_in": True},
        {"expires_in": -1},
        {"expires_in": "3600"},
        {"expires_in": float("inf")},
        {"token_type": "MAC"},
        {"scope": [PLAN_SCOPE]},
        {"access_token": None},
    ],
)
async def test_invalid_token_response(server: FakeOAuth, oauth: ChatGPTOAuth, changes: dict[str, Any]) -> None:
    server.token_changes = changes
    with pytest.raises(OAuthValidationError):
        await oauth.sign_in(open_browser=server.browser)
    assert "You're signed in." not in server.callback_html
    await assert_callback_closed(server)


async def test_jwks_rotation(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    account = await oauth.sign_in(open_browser=server.browser)
    server.kid = "rotated-key"
    server.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    await oauth.sign_in(account, open_browser=server.browser)
    assert server.jwks_requests == 2


@pytest.mark.parametrize("body", [{"keys": []}, {"keys": [{"kty": "RSA", "kid": "missing-fields"}]}, []])
async def test_invalid_jwks(server: FakeOAuth, oauth: ChatGPTOAuth, body: object) -> None:
    server.jwks_body = body
    with pytest.raises(OAuthError):
        await oauth.sign_in(open_browser=server.browser)
    await assert_callback_closed(server)


async def test_expired_code_restarts_with_issued_id(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    server.invalid_grants = 1
    result = await oauth.sign_in(open_browser=server.browser)
    first, second = server.authorizations
    assert first["client_id"] == DYNAMIC_CLIENT_ID and second["client_id"] == result.client_id
    assert "agent_name_hint" not in second
    assert first["ext_agent_host_id"] == second["ext_agent_host_id"]
    for name in ("state", "nonce", "code_challenge"):
        assert first[name] != second[name]
    assert server.callback_status == 200
    await assert_callback_closed(server)


async def test_expired_code_retry_is_bounded(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    server.invalid_grants = 3
    with pytest.raises(OAuthError) as info:
        await oauth.sign_in(open_browser=server.browser)
    assert info.value.code == "invalid_grant"
    assert len(server.authorizations) == 2
    await assert_callback_closed(server)


async def test_http_error_does_not_leak_body(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    server.token_status = 503
    server.token_body = {"error": "secret", "error_description": "refresh-secret"}
    with pytest.raises(OAuthError) as info:
        await oauth.sign_in(open_browser=server.browser)
    assert info.value.status == 503
    assert "secret" not in str(info.value) and "secret" not in server.callback_html
    assert len(server.exchanges) == 1


@pytest.mark.parametrize("during_exchange", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_cleanup_on_timeout_or_cancel(
    server: FakeOAuth,
    oauth: ChatGPTOAuth,
    during_exchange: bool,
    cancel: bool,
) -> None:
    opened = asyncio.Event()

    async def browser(url: str) -> None:
        if during_exchange:
            server.token_delay = True
            opened.set()
            await server.browser(url)
        else:
            server.authorizations.append({key: values[0] for key, values in parse_qs(urlsplit(url).query).items()})
            opened.set()

    task = asyncio.create_task(oauth.sign_in(open_browser=browser, timeout=10 if cancel else 0.2))
    await asyncio.wait_for(opened.wait(), timeout=2)
    if during_exchange:
        await asyncio.wait_for(server.exchange_started.wait(), timeout=2)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    await assert_callback_closed(server)


async def test_callback_consumed_once(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    server.duplicate_callback = True
    await oauth.sign_in(open_browser=server.browser)
    assert server.duplicate_status == 409
    assert len(server.exchanges) == 1


async def test_browser_failure_closes_listener(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    async def browser(url: str) -> None:
        server.authorizations.append({key: values[0] for key, values in parse_qs(urlsplit(url).query).items()})
        raise RuntimeError("Browser unavailable")

    with pytest.raises(RuntimeError, match="Browser unavailable"):
        await oauth.sign_in(open_browser=browser)
    await assert_callback_closed(server)


async def test_occupied_port_does_not_launch_browser(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(OSError):
            await oauth.sign_in(open_browser=server.browser, port=listener.getsockname()[1])
    assert not server.authorizations


@pytest.mark.parametrize("changes", [{"issuer": "https://other.test"}, {"token_endpoint": "http://other.test/token"}])
async def test_discovery_cannot_change_issuer(server: FakeOAuth, oauth: ChatGPTOAuth, changes: dict[str, Any]) -> None:
    server.discovery_changes = changes
    with pytest.raises(OAuthValidationError):
        await oauth.sign_in(open_browser=server.browser)
    assert not server.authorizations


async def test_account_from_different_host_rejected(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    selected = ChatGPTAccount("oaiapp_test", "different-host", server.issuer, "subject-1")
    with pytest.raises(OAuthValidationError):
        await oauth.sign_in(selected, open_browser=server.browser)
    assert not server.authorizations


def test_import_does_not_require_textual() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['textual'] = None; import axio_transport_openai.oauth"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


async def test_refresh_validates_new_id_without_authorization_nonce(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    account = await oauth.sign_in()
    server.token_changes["refresh_token"] = "rotated-refresh"
    refreshed = await oauth.refresh(account)
    assert refreshed.subject == account.subject
    assert refreshed.tokens is not None
    assert refreshed.tokens.refresh_token == "rotated-refresh"
    assert server.exchanges[-1]["grant_type"] == "refresh_token"
    assert "scope" not in server.exchanges[-1]


@pytest.mark.parametrize(
    "claim_changes",
    [
        {"sub": "another-user"},
        {"iss": "https://another.test"},
        {"aud": "another-client"},
        {"exp": 1},
    ],
)
async def test_refreshed_identity_must_match(
    server: FakeOAuth,
    oauth: ChatGPTOAuth,
    claim_changes: dict[str, Any],
) -> None:
    account = await oauth.sign_in()
    server.claim_changes = claim_changes
    with pytest.raises(OAuthValidationError):
        await oauth.refresh(account)


async def test_refresh_rejects_invalid_signature(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    account = await oauth.sign_in()
    server.bad_signature = True
    with pytest.raises(OAuthValidationError):
        await oauth.refresh(account)


async def test_refresh_validation_timeout_does_not_report_retryable_exchange(
    server: FakeOAuth,
    oauth: ChatGPTOAuth,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = await oauth.sign_in()

    async def unavailable(self: ChatGPTOAuth, *args: Any) -> dict[str, Any]:
        raise TimeoutError

    monkeypatch.setattr(ChatGPTOAuth, "_verify", unavailable)
    with pytest.raises(OAuthValidationError) as error:
        await oauth.refresh(account)
    assert not error.value.temporary
    assert server.exchanges[-1]["grant_type"] == "refresh_token"


@pytest.mark.parametrize("invalid", ["", "state", "origin"])
async def test_pasted_callback_reuses_normal_validation(
    server: FakeOAuth,
    oauth: ChatGPTOAuth,
    invalid: str,
) -> None:
    pasted = ""

    async def display(url: str) -> None:
        nonlocal pasted
        async with server.session.get(url, allow_redirects=False) as response:
            pasted = response.headers["Location"]
        if invalid == "state":
            parsed = urlsplit(pasted)
            query = dict(parse_qs(parsed.query))
            query["state"] = ["wrong"]
            pasted = parsed._replace(query=urlencode(query, doseq=True)).geturl()
        elif invalid == "origin":
            pasted = pasted.replace("127.0.0.1", "example.test")

    async def read_callback() -> str:
        return pasted

    if invalid:
        with pytest.raises(OAuthValidationError):
            await oauth.sign_in(open_browser=display, read_callback=read_callback)
        assert not server.exchanges
    else:
        account = await oauth.sign_in(open_browser=display, read_callback=read_callback)
        assert account.client_id == server.client_id
        assert len(server.exchanges) == 1
    await assert_callback_closed(server)


async def test_browser_callback_cancels_waiting_manual_input(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    stopped = asyncio.Event()
    browser_task: asyncio.Task[None] | None = None

    async def browser(url: str) -> None:
        nonlocal browser_task
        browser_task = asyncio.create_task(server.browser(url))

    async def read_callback() -> str:
        try:
            await asyncio.Future[None]()
            return ""
        finally:
            stopped.set()

    await oauth.sign_in(open_browser=browser, read_callback=read_callback)
    assert stopped.is_set()
    assert browser_task is not None
    await browser_task


async def test_manual_terminal_sign_in_end_to_end(
    server: FakeOAuth,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import os

    from axio_transport_openai import auth
    from axio_transport_openai.credential_stores import FileCredentialStore

    read_fd, write_fd = os.pipe()
    original_input = auth.CallbackInput
    original_open = auth.TerminalAuthorization.open

    def client(session: aiohttp.ClientSession, host_id: str) -> ChatGPTOAuth:
        return ChatGPTOAuth(session, host_id, issuer=server.issuer)

    async def open_remote(self: auth.TerminalAuthorization, url: str) -> None:
        await original_open(self, url)
        async with server.session.get(url, allow_redirects=False) as response:
            # The local browser cannot reach remote loopback. Paste its address.
            os.write(write_fd, (response.headers["Location"] + "\n").encode())

    monkeypatch.setattr(auth, "ChatGPTOAuth", client)
    monkeypatch.setattr(auth, "CallbackInput", lambda: original_input(read_fd))
    monkeypatch.setattr(auth.TerminalAuthorization, "open", open_remote)
    path = tmp_path / "chatgpt.auth"
    try:
        account = await auth.sign_in(credentials=path, manual=True, timeout=2)
        store = FileCredentialStore(path)
        async with store.lock(account.client_id):
            record = await store.load(account.client_id)
        assert record is not None and record.account == account
        assert account.host_id == await store.host_id()
        output = capsys.readouterr()
        assert server.issuer + "/authorize?" in output.out
        assert "Paste redirect URL" in output.err
        assert "opaque-access-secret" not in output.out + output.err
        assert "refresh-secret" not in output.out + output.err
    finally:
        os.close(read_fd)
        os.close(write_fd)


async def test_sign_in_timeout_cancels_manual_input(server: FakeOAuth, oauth: ChatGPTOAuth) -> None:
    stopped = asyncio.Event()

    async def display(url: str) -> None:
        pass

    async def read_callback() -> str:
        try:
            await asyncio.Future[None]()
            return ""
        finally:
            stopped.set()

    with pytest.raises(TimeoutError):
        await oauth.sign_in(open_browser=display, read_callback=read_callback, timeout=0.05)
    assert stopped.is_set()
