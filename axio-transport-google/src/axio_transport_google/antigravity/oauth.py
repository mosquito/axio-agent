"""Google installed-app OAuth for the Antigravity subscription client."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlencode, urlsplit

import aiohttp
from aiohttp import web
from axio.exceptions import AxioError

from .page import page_headers, sign_in_page


class OAuthError(AxioError):
    """An authorization failure with safe diagnostics."""

    def __init__(self, message: str, *, code: str = "", status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True, slots=True)
class OAuthConfig:
    # Published installed-app configuration, not an account credential.
    # https://github.com/dvcrn/antigravity-oauth-proxy/blob/main/internal/credentials/types.go
    client_id: str = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
    client_secret: str = field(default="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", repr=False)
    redirect_uri: str = "http://localhost:51121/oauth-callback"
    authorization_url: str = "https://accounts.google.com/o/oauth2/v2/auth"
    token_url: str = "https://oauth2.googleapis.com/token"
    userinfo_url: str = "https://www.googleapis.com/oauth2/v2/userinfo"
    revoke_url: str = "https://oauth2.googleapis.com/revoke"
    scopes: tuple[str, ...] = (
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    )


@dataclass(frozen=True, slots=True)
class Tokens:
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AntigravityAccount:
    account_id: str = field(metadata={"credential_key": True})
    subject: str
    client_id: str
    email: str = ""
    tokens: Tokens | None = field(default=None, repr=False)
    revision: str = field(default_factory=lambda: secrets.token_hex(16))


def callback_code(url: str, config: OAuthConfig, state: str) -> str:
    """Validate the exact redirect target and state before accepting a code."""
    actual, expected = urlsplit(url), urlsplit(config.redirect_uri)
    if (actual.scheme, actual.netloc, actual.path) != (expected.scheme, expected.netloc, expected.path):
        raise OAuthError("The redirect URL does not match this sign-in.")
    if actual.fragment:
        raise OAuthError("The redirect URL must not contain a fragment.")
    query = parse_qs(actual.query, keep_blank_values=True)
    states = query.get("state", [])
    if len(states) != 1 or not secrets.compare_digest(states[0], state):
        raise OAuthError("The OAuth state does not match this sign-in.")
    if "error" in query:
        raise OAuthError("Google authorization was denied.", code="access_denied")
    codes = query.get("code", [])
    if len(codes) != 1 or not codes[0]:
        raise OAuthError("The redirect URL must contain one authorization code.")
    return codes[0]


class AntigravityOAuth:
    """Authorize through a caller-owned HTTP session and UI callbacks."""

    def __init__(self, http: aiohttp.ClientSession, config: OAuthConfig | None = None) -> None:
        self.http = http
        self.config = config or OAuthConfig()

    async def token_request(self, form: dict[str, str], previous: Tokens | None = None) -> Tokens:
        form = {**form, "client_id": self.config.client_id, "client_secret": self.config.client_secret}
        try:
            async with self.http.post(
                self.config.token_url, data=form, timeout=aiohttp.ClientTimeout(total=60), allow_redirects=False
            ) as response:
                status = response.status
                data = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, ValueError):
            raise OAuthError("Could not complete the Google token exchange.") from None
        if not isinstance(data, dict):
            raise OAuthError("Google returned an invalid token response.", status=status)
        if status != 200:
            code = data.get("error")
            known = {"invalid_grant", "invalid_client", "access_denied", "temporarily_unavailable", "invalid_scope"}
            code = code if isinstance(code, str) and code in known else "oauth_error"
            raise OAuthError(f"Google token exchange failed: {code}.", code=code, status=status)
        access, refresh, expires = data.get("access_token"), data.get("refresh_token"), data.get("expires_in")
        if refresh is None and previous is not None:
            refresh = previous.refresh_token
        if (
            not isinstance(access, str)
            or not access
            or not isinstance(refresh, str)
            or not refresh
            or not isinstance(expires, (int, float))
            or isinstance(expires, bool)
            or not math.isfinite(expires)
            or expires <= 0
            or not isinstance(data.get("token_type", "Bearer"), str)
            or data.get("token_type", "Bearer").lower() != "bearer"
        ):
            raise OAuthError("Google returned incomplete tokens; authorize offline access again.")
        scope = data.get("scope")
        scopes = tuple(scope.split()) if isinstance(scope, str) else (previous.scopes if previous else ())
        return Tokens(access, refresh, time.time() + expires, scopes)

    async def exchange(self, code: str, verifier: str) -> AntigravityAccount:
        tokens = await self.token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": self.config.redirect_uri,
            }
        )
        try:
            async with self.http.get(
                self.config.userinfo_url,
                headers={"Authorization": f"Bearer {tokens.access_token}"},
                timeout=aiohttp.ClientTimeout(total=30),
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise OAuthError("Google could not verify the account identity.", status=response.status)
                identity = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, ValueError):
            raise OAuthError("Could not read the Google account identity.") from None
        if not isinstance(identity, dict) or not isinstance(identity.get("id"), str) or not identity["id"]:
            raise OAuthError("Google did not return a stable account identity.")
        subject = identity["id"]
        email = identity.get("email", "")
        if not isinstance(email, str):
            email = ""
        key = hashlib.sha256(f"{self.config.client_id}:{subject}".encode()).hexdigest()
        return AntigravityAccount(key, subject, self.config.client_id, email, tokens)

    async def refresh(self, tokens: Tokens) -> Tokens:
        return await self.token_request({"grant_type": "refresh_token", "refresh_token": tokens.refresh_token}, tokens)

    async def revoke(self, tokens: Tokens) -> bool:
        """Return whether Google confirmed revocation; never print the token."""
        try:
            async with self.http.post(
                self.config.revoke_url,
                data={"token": tokens.refresh_token},
                timeout=aiohttp.ClientTimeout(total=30),
                allow_redirects=False,
            ) as response:
                return response.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False

    async def sign_in(
        self,
        *,
        open_browser: Callable[[str], Awaitable[None]],
        read_callback: Callable[[], Awaitable[str]] | None = None,
        manual: bool = False,
        timeout: float = 300,
    ) -> AntigravityAccount:
        """Start the loopback listener before opening the authorization URL.

        In manual mode no listener is opened. The caller supplies the full
        browser redirect URL through read_callback. Both paths validate state.
        """
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("The sign-in timeout must be positive and finite.")
        target = urlsplit(self.config.redirect_uri)
        if (
            target.scheme != "http"
            or target.hostname not in {"localhost", "127.0.0.1"}
            or not target.port
            or target.query
            or target.fragment
            or target.username
        ):
            raise ValueError("OAuth requires an explicit loopback callback port and path.")
        if manual and read_callback is None:
            raise ValueError("Manual sign-in requires a callback reader.")
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        url = (
            self.config.authorization_url
            + "?"
            + urlencode(
                {
                    "client_id": self.config.client_id,
                    "redirect_uri": self.config.redirect_uri,
                    "response_type": "code",
                    "scope": " ".join(self.config.scopes),
                    "access_type": "offline",
                    "prompt": "consent",
                    "state": state,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                }
            )
        )
        future: asyncio.Future[AntigravityAccount | Exception] = asyncio.get_running_loop().create_future()
        runner: web.AppRunner | None = None
        reader: asyncio.Task[None] | None = None
        consumed = False

        async def accept(request: web.Request) -> web.Response:
            nonlocal consumed
            nonce = secrets.token_urlsafe(24)
            headers = page_headers(nonce)
            if consumed or future.done():
                return web.Response(status=409, text="This sign-in has already received a callback.", headers=headers)
            outcome: AntigravityAccount | Exception
            try:
                code = callback_code(str(request.url), self.config, state)
            except OAuthError as error:
                if error.code != "access_denied":
                    return web.Response(
                        status=400,
                        text=sign_in_page(success=False, nonce=nonce),
                        content_type="text/html",
                        headers=headers,
                    )
                outcome = error
                consumed = True
            else:
                consumed = True
                try:
                    outcome = await self.exchange(code, verifier)
                except Exception as error:
                    outcome = error
            success = isinstance(outcome, AntigravityAccount)
            response = web.Response(
                status=200 if success else 400,
                text=sign_in_page(success=success, nonce=nonce),
                content_type="text/html",
                headers=headers,
            )
            # Deliver the complete page before sign_in() closes the listener.
            try:
                await response.prepare(request)
                await response.write_eof()
            except ConnectionError:
                pass
            finally:
                if not future.done():
                    future.set_result(outcome)
            return response

        async def read_manual() -> None:
            nonlocal consumed
            assert read_callback is not None
            try:
                code = callback_code(await read_callback(), self.config, state)
                if consumed or future.done():
                    return
                consumed = True
                account = await self.exchange(code, verifier)
                if not future.done():
                    future.set_result(account)
            except Exception as exc:
                if not future.done():
                    future.set_result(exc)

        try:
            async with asyncio.timeout(timeout):
                if not manual:
                    app = web.Application()
                    app.router.add_get(target.path, accept)
                    runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
                    await runner.setup()
                    await web.TCPSite(runner, "127.0.0.1", target.port).start()
                await open_browser(url)
                if read_callback is not None:
                    reader = asyncio.create_task(read_manual())
                result = await future
                if isinstance(result, Exception):
                    raise result
                return result
        finally:
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if runner is not None:
                await runner.cleanup()
            if future.done() and not future.cancelled():
                future.exception()
            else:
                future.cancel()
