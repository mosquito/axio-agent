"""Sign in with ChatGPT for public clients, independent of CLI and TUI code."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import os
import secrets
import sys
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlencode

import aiohttp
import jwt
from aiohttp import web
from axio.exceptions import AxioError
from yarl import URL

from ._oauth_page import sign_in_page

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"
CALLBACK_PATH = "/auth/callback"
TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)


class OAuthError(AxioError):
    """Sign-in failed. Diagnostics never include tokens or callback parameters.

    request_not_sent confirms that the relevant request was never sent. During
    refresh, discovery failures also set it because no exchange was attempted.
    """

    def __init__(
        self, message: str, *, code: str = "", status: int | None = None, request_not_sent: bool = False
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_not_sent = request_not_sent

    @property
    def temporary(self) -> bool:
        return not isinstance(self, OAuthValidationError) and (
            self.status is None or self.status == 429 or self.status >= 500
        )


class OAuthValidationError(OAuthError):
    """The callback, token response, or account identity could not be verified."""


class OAuthDenied(OAuthError):
    """The user declined authorization."""


@dataclass(frozen=True, slots=True)
class ChatGPTTokens:
    """Credentials and granted scopes from one validated sign-in."""

    id_token: str = field(repr=False)
    access_token: str | None = field(repr=False)
    refresh_token: str | None = field(repr=False)
    token_type: str | None
    scopes: frozenset[str]
    received_at: float
    expires_at: float | None

    @property
    def plan_enabled(self) -> bool:
        """Whether the token response granted ChatGPT plan usage."""
        return PLAN_SCOPE in self.scopes


@dataclass(frozen=True, slots=True)
class ChatGPTAccount:
    """One issued client registration bound to a host and verified identity.

    Email is a display label. Use issuer, client_id, and subject for identity.
    Retain the registration with tokens=None after signing out.
    """

    client_id: str = field(metadata={"credential_key": True})
    host_id: str
    issuer: str
    subject: str
    email: str | None = None
    name: str | None = None
    tokens: ChatGPTTokens | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _Authorization:
    client_id: str
    code: str = field(repr=False)
    verifier: str = field(repr=False)
    nonce: str = field(repr=False)
    redirect_uri: str


async def launch_browser(url: str) -> bool:
    """Try a local browser. SSH sessions use the manual authorization path."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    try:
        return await asyncio.to_thread(webbrowser.open, url)
    except (webbrowser.Error, OSError):
        return False


def print_authorization_url(url: str) -> None:
    """Print a usable authorization URL without an optional ID-token hint."""
    print("Open this URL in your browser:", file=sys.stderr)
    print(URL(url).without_query_params("id_token_hint"), flush=True)


async def open_browser(url: str) -> None:
    if not await launch_browser(url):
        print_authorization_url(url)


def required_string(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise OAuthValidationError(f"Missing or invalid {name}.")
    return value


def optional_string(data: dict[str, Any], name: str) -> str | None:
    if data.get(name) is None:
        return None
    return required_string(data, name)


@dataclass(slots=True)
class ChatGPTOAuth:
    """Run browser authorization using an aiohttp session supplied by the caller.

    Persist host_id before the first sign-in and reuse it on this host. This
    client returns new account records; it never writes credentials or mutates
    the selected account. Use a session without default authorization headers.
    issuer can be changed to an HTTP loopback origin for local protocol tests.
    """

    session: aiohttp.ClientSession = field(repr=False)
    host_id: str
    agent_name: str = "Axio"
    issuer: str = ISSUER
    _jwks: jwt.PyJWKSet | None = field(default=None, init=False, repr=False)
    _jwks_url: str = field(default="", init=False, repr=False)
    _jwks_expires: float = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.host_id.strip() or not self.agent_name.strip():
            raise ValueError("host_id and agent_name must not be empty.")
        origin = URL(self.issuer)
        if (
            not origin.is_absolute()
            or origin.user is not None
            or origin.query_string
            or origin.fragment
            or origin.path != "/"
            or (origin.scheme != "https" and not (origin.scheme == "http" and origin.host == "127.0.0.1"))
        ):
            raise ValueError("issuer must be an HTTPS origin or an HTTP 127.0.0.1 test origin.")

    async def sign_in(
        self,
        account: ChatGPTAccount | None = None,
        *,
        open_browser: Callable[[str], Awaitable[None]] = open_browser,
        read_callback: Callable[[], Awaitable[str]] | None = None,
        timeout: float = 300,
        port: int = 0,
    ) -> ChatGPTAccount:
        """Register or reauthorize an account and validate its ID token.

        The timeout covers discovery, browser interaction, and token validation.
        Cancellation propagates and closes the callback listener. The default
        port selects an available port; the callback always uses 127.0.0.1 and
        /auth/callback. An expired code starts one fresh authorization attempt
        with the issued client ID. Missing plan consent is a successful sign-in
        whose tokens.plan_enabled is false.
        """
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive.")
        if account is not None and (
            account.host_id != self.host_id
            or account.issuer != self.issuer
            or not account.subject
            or not account.client_id
            or account.client_id == DYNAMIC_CLIENT_ID
        ):
            raise OAuthValidationError("The selected account does not match this host or issuer.")
        async with asyncio.timeout(timeout):
            discovery = await self._json("GET", self.issuer + "/.well-known/openid-configuration")
            if discovery.get("issuer") != self.issuer:
                raise OAuthValidationError("The discovery issuer did not match.")
            authorize_url = self._endpoint(discovery, "authorization_endpoint")
            token_url = self._endpoint(discovery, "token_endpoint")
            jwks_url = self._endpoint(discovery, "jwks_uri")
            client_id = account.client_id if account else DYNAMIC_CLIENT_ID

            async def complete(authorization: _Authorization) -> ChatGPTAccount:
                nonlocal client_id
                client_id = authorization.client_id
                return await self._exchange(authorization, token_url, jwks_url, account)

            for attempt in range(2):
                try:
                    return await self._authorize(
                        authorize_url, client_id, account, open_browser, port, complete, read_callback
                    )
                except OAuthError as exc:
                    if exc.code != "invalid_grant" or attempt != 0:
                        raise
        raise AssertionError("Authorization attempts exhausted without a result.")

    async def _exchange(
        self,
        authorization: _Authorization,
        token_url: str,
        jwks_url: str,
        account: ChatGPTAccount | None,
    ) -> ChatGPTAccount:
        data = await self._json(
            "POST",
            token_url,
            form={
                "grant_type": "authorization_code",
                "client_id": authorization.client_id,
                "code": authorization.code,
                "code_verifier": authorization.verifier,
                "redirect_uri": authorization.redirect_uri,
                "resource": RESOURCE,
            },
        )
        tokens = self._tokens(data, time.time())
        claims = await self._verify(tokens.id_token, authorization.client_id, authorization.nonce, jwks_url)
        subject = required_string(claims, "sub")
        if account is not None and subject != account.subject:
            raise OAuthValidationError("The signed-in identity does not match the selected account.")
        return ChatGPTAccount(
            client_id=authorization.client_id,
            host_id=self.host_id,
            issuer=self.issuer,
            subject=subject,
            email=optional_string(claims, "email"),
            name=optional_string(claims, "name"),
            tokens=tokens,
        )

    async def refresh(self, account: ChatGPTAccount) -> ChatGPTAccount:
        """Rotate credentials. The caller must lock and persist the account.

        Do not call this concurrently for one registration. ChatGPTSession
        supplies the lock, persistence, and recovery after a failed write.
        """
        self._check_account(account)
        previous = account.tokens
        if previous is None or previous.refresh_token is None:
            raise OAuthValidationError("The account has no refresh token.")
        try:
            async with asyncio.timeout(30):
                discovery = await self._discovery()
                token_endpoint = self._endpoint(discovery, "token_endpoint")
        except OAuthError as exc:
            # No refresh request was attempted, even if discovery reached the server.
            exc.request_not_sent = True
            raise
        except TimeoutError:
            raise OAuthError("OAuth discovery timed out.", request_not_sent=True) from None
        async with asyncio.timeout(30):
            data = await self._json(
                "POST",
                token_endpoint,
                form={
                    "grant_type": "refresh_token",
                    "client_id": account.client_id,
                    "refresh_token": previous.refresh_token,
                    "resource": RESOURCE,
                },
            )
        # An omitted scope retains the original grant. An omitted ID token
        # retains the login hint; it is not a new assertion of identity.
        new_identity = "id_token" in data
        data.setdefault("id_token", previous.id_token)
        data.setdefault("scope", " ".join(sorted(previous.scopes)))
        required_string(data, "refresh_token")
        required_string(data, "access_token")
        tokens = self._tokens(data, time.time())
        if new_identity:
            try:
                async with asyncio.timeout(30):
                    claims = await self._verify(
                        tokens.id_token, account.client_id, None, self._endpoint(discovery, "jwks_uri")
                    )
            except (OAuthError, TimeoutError):
                # The exchange already succeeded. Even a temporary JWKS failure
                # must not cause the caller to replay the spent refresh token.
                raise OAuthValidationError("Could not verify the rotated ID token.") from None
            if required_string(claims, "sub") != account.subject:
                raise OAuthValidationError("The refreshed identity did not match the account.")
        return replace(account, tokens=tokens)

    async def revoke(self, account: ChatGPTAccount) -> None:
        """Revoke the renewable session; an empty HTTP 200 confirms revocation."""
        self._check_account(account)
        if account.tokens is None or account.tokens.refresh_token is None:
            raise OAuthValidationError("The account has no refresh token to revoke.")
        try:
            async with asyncio.timeout(30):
                discovery = await self._discovery()
                async with self.session.post(
                    self._endpoint(discovery, "revocation_endpoint"),
                    data={
                        "token": account.tokens.refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": account.client_id,
                    },
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        raise OAuthError("OAuth revocation failed.", status=response.status)
        except (aiohttp.ClientError, TimeoutError):
            raise OAuthError("Could not confirm OAuth revocation.") from None

    def _check_account(self, account: ChatGPTAccount) -> None:
        if (
            account.host_id != self.host_id
            or account.issuer != self.issuer
            or not account.subject
            or not account.client_id
            or account.client_id == DYNAMIC_CLIENT_ID
        ):
            raise OAuthValidationError("The selected account does not match this host or issuer.")

    async def _discovery(self) -> dict[str, Any]:
        discovery = await self._json("GET", self.issuer + "/.well-known/openid-configuration")
        if discovery.get("issuer") != self.issuer:
            raise OAuthValidationError("The discovery issuer did not match.")
        return discovery

    def _endpoint(self, discovery: dict[str, Any], key: str) -> str:
        value = required_string(discovery, key)
        url = URL(value)
        # This flow uses OpenAI's own endpoints, never URLs supplied by a JWT.
        if not url.is_absolute() or url.origin() != URL(self.issuer).origin() or url.user or url.fragment:
            raise OAuthValidationError("Invalid OAuth discovery endpoint.")
        return value

    async def submit_callback(self, url: str, redirect_uri: str) -> None:
        """Deliver a pasted redirect URL to this sign-in's loopback listener."""
        try:
            callback = URL(url.strip())
            if callback.user is not None or callback.fragment or callback.with_query(None) != URL(redirect_uri):
                raise ValueError
        except ValueError:
            raise OAuthValidationError(
                "Paste the full redirect URL from this sign-in's browser address bar."
            ) from None
        try:
            async with self.session.get(callback, allow_redirects=False) as response:
                await response.read()
                if response.status not in (200, 400):
                    raise OAuthError("The pasted callback could not be accepted.")
        except aiohttp.ClientError:
            raise OAuthError("Could not deliver the pasted callback to the local listener.") from None

    async def _authorize(
        self,
        authorize_url: str,
        client_id: str,
        account: ChatGPTAccount | None,
        open_browser: Callable[[str], Awaitable[None]],
        port: int,
        complete: Callable[[_Authorization], Awaitable[ChatGPTAccount]],
        read_callback: Callable[[], Awaitable[str]] | None,
    ) -> ChatGPTAccount:
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=")
        result: asyncio.Future[ChatGPTAccount | Exception] = asyncio.get_running_loop().create_future()
        redirect_uri = ""
        consumed = False

        async def callback(request: web.Request) -> web.Response:
            nonlocal consumed
            page_nonce = secrets.token_urlsafe(24)
            headers = {
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                    f"script-src 'nonce-{page_nonce}'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
                ),
                "X-Content-Type-Options": "nosniff",
            }
            if consumed or result.done():
                return web.Response(status=409, text="This sign-in callback was already used.", headers=headers)
            consumed = True
            outcome: ChatGPTAccount | Exception
            try:
                query = request.query
                if any(len(query.getall(key)) != 1 for key in ("state", "code", "client_id", "error") if key in query):
                    raise OAuthValidationError("Duplicate OAuth callback parameters.")
                if not secrets.compare_digest(query.get("state", "").encode(), state.encode()):
                    raise OAuthValidationError("The OAuth callback state did not match.")
                if "error" in query:
                    if query["error"] == "access_denied":
                        raise OAuthDenied("Authorization was declined.", code="access_denied")
                    raise OAuthError("Authorization failed.")
                issued_id = query.get("client_id", client_id)
                if not issued_id or issued_id == DYNAMIC_CLIENT_ID:
                    raise OAuthValidationError("Registration did not return an issued client ID.")
                if client_id != DYNAMIC_CLIENT_ID and issued_id != client_id:
                    raise OAuthValidationError("The callback changed the selected client ID.")
                code = query.get("code", "")
                if not code:
                    raise OAuthValidationError("The callback did not contain an authorization code.")
                signed_in = await complete(_Authorization(issued_id, code, verifier, nonce, redirect_uri))
                assert signed_in.tokens is not None
                outcome = signed_in
                response = web.Response(
                    text=sign_in_page(success=True, plan_enabled=signed_in.tokens.plan_enabled, nonce=page_nonce),
                    content_type="text/html",
                    headers=headers,
                )
            except Exception as exc:
                # Forward failures to sign_in() without logging callback secrets.
                outcome = exc
                response = web.Response(
                    status=400,
                    text=sign_in_page(success=False, nonce=page_nonce),
                    content_type="text/html",
                    headers=headers,
                )
            # Send the complete page before allowing sign_in() to close the listener.
            try:
                await response.prepare(request)
                await response.write_eof()
            except ConnectionError:
                # Closing the browser must not discard an already validated account.
                pass
            finally:
                if not result.done():
                    result.set_result(outcome)
            return response

        app = web.Application()
        app.router.add_get(CALLBACK_PATH, callback, allow_head=False)
        # aiohttp's default access log includes the code and state query parameters.
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        input_task: asyncio.Task[None] | None = None
        try:
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            bound_port = runner.addresses[0][1]
            redirect_uri = f"http://127.0.0.1:{bound_port}{CALLBACK_PATH}"
            params = {
                "client_id": client_id,
                "ext_agent_host_id": self.host_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": SCOPES,
                "resource": RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": challenge.decode("ascii"),
            }
            if client_id == DYNAMIC_CLIENT_ID:
                params["agent_name_hint"] = self.agent_name
            if account is not None:
                if account.email:
                    params["login_hint"] = account.email
                if account.tokens is not None:
                    params["id_token_hint"] = account.tokens.id_token
            await open_browser(f"{authorize_url}?{urlencode(params)}")
            if read_callback is not None:

                async def pasted_callback() -> None:
                    assert read_callback is not None
                    await self.submit_callback(await read_callback(), redirect_uri)

                input_task = asyncio.create_task(pasted_callback())
                await asyncio.wait((result, input_task), return_when=asyncio.FIRST_COMPLETED)
                if not result.done():
                    await input_task
            response = await result
            if isinstance(response, Exception):
                raise response
            return response
        finally:
            if input_task is not None:
                input_task.cancel()
                await asyncio.gather(input_task, return_exceptions=True)
            result.cancel()
            await runner.cleanup()

    async def _json(self, method: str, url: str, *, form: dict[str, str] | None = None) -> dict[str, Any]:
        try:
            async with self.session.request(method, url, data=form, allow_redirects=False) as response:
                if response.status != 200:
                    code = ""
                    # Only retain a known error code; provider descriptions can contain secrets.
                    try:
                        error = await response.json()
                        value = error.get("error") if isinstance(error, dict) else None
                        if isinstance(value, str) and value in TERMINAL_REFRESH_ERRORS | {"invalid_client"}:
                            code = value
                    except (ValueError, aiohttp.ClientError):
                        pass
                    raise OAuthError("OAuth request failed.", code=code, status=response.status)
                try:
                    data = await response.json()
                except (ValueError, aiohttp.ClientError, TimeoutError):
                    raise OAuthValidationError("Invalid successful OAuth response.") from None
        except aiohttp.ClientConnectorError:
            raise OAuthError("Could not connect to the OAuth server.", request_not_sent=True) from None
        except (aiohttp.ClientError, ValueError, TimeoutError):
            raise OAuthError("Could not read the OAuth response.") from None
        if not isinstance(data, dict):
            raise OAuthValidationError("OAuth response must be a JSON object.")
        return data

    async def _verify(self, token: str, client_id: str, nonce: str | None, jwks_url: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
            kid = required_string(header, "kid")
            if header.get("alg") != "RS256":
                raise OAuthValidationError("Unsupported ID token signature algorithm.")
            cached = self._jwks is not None and self._jwks_url == jwks_url and time.monotonic() < self._jwks_expires
            if not cached:
                await self._load_jwks(jwks_url)
            key = self._signing_key(kid)
            if key is None and cached:
                await self._load_jwks(jwks_url)
                key = self._signing_key(kid)
            if key is None:
                raise OAuthValidationError("No matching ID token signing key.")
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=client_id,
                leeway=5,
                options={"require": ["iss", "aud", "sub", "exp", "iat"] + (["nonce"] if nonce is not None else [])},
            )
        except (jwt.PyJWTError, ValueError, TypeError):
            raise OAuthValidationError("ID token verification failed.") from None
        if nonce is not None:
            actual_nonce = required_string(claims, "nonce")
            if not secrets.compare_digest(actual_nonce.encode(), nonce.encode()):
                raise OAuthValidationError("The ID token nonce did not match.")
        audience = claims["aud"]
        if (isinstance(audience, list) and len(audience) > 1) or "azp" in claims:
            if claims.get("azp") != client_id:
                raise OAuthValidationError("The ID token authorized party did not match.")
        return claims

    async def _load_jwks(self, url: str) -> None:
        self._jwks = jwt.PyJWKSet.from_dict(await self._json("GET", url))
        self._jwks_url = url
        self._jwks_expires = time.monotonic() + 300

    def _signing_key(self, kid: str) -> jwt.PyJWK | None:
        assert self._jwks is not None
        keys = [
            key
            for key in self._jwks.keys
            if key.key_id == kid
            and key.key_type == "RSA"
            and key.public_key_use in (None, "sig")
            and key.algorithm_name == "RS256"
        ]
        return keys[0] if len(keys) == 1 else None

    @staticmethod
    def _tokens(data: dict[str, Any], received_at: float) -> ChatGPTTokens:
        id_token = required_string(data, "id_token")
        scope = data.get("scope", "")
        if not isinstance(scope, str):
            raise OAuthValidationError("Invalid granted scopes.")
        access_token = optional_string(data, "access_token")
        refresh_token = optional_string(data, "refresh_token")
        token_type = optional_string(data, "token_type")
        scopes = frozenset(scope.split())
        expires_at = None
        if access_token is not None:
            expires_in = data.get("expires_in")
            if (
                isinstance(expires_in, bool)
                or not isinstance(expires_in, (int, float))
                or not math.isfinite(expires_in)
                or expires_in <= 0
                or token_type is None
                or token_type.lower() != "bearer"
            ):
                raise OAuthValidationError("Invalid access token expiry or type.")
            expires_at = received_at + expires_in
        if PLAN_SCOPE in scopes and access_token is None:
            raise OAuthValidationError("Plan permission was granted without an access token.")
        return ChatGPTTokens(id_token, access_token, refresh_token, token_type, scopes, received_at, expires_at)
