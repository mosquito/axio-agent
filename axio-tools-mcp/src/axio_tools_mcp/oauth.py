"""Adapt the MCP client's OAuth flow to Axio credentials and browser pages."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from string import Template
from typing import Any

from aiohttp import web
from aiohttp_tiny_mcp import LoopbackCallback, OAuth, OAuthCallback
from aiohttp_tiny_mcp.client.oauth import AuthorizationAttempt
from axio.credentials import FileCredentialStore, default_credentials_path
from yarl import URL

from .config import MCPServerConfig


@dataclass(frozen=True, slots=True)
class OAuthRecord:
    key: str = field(metadata={"credential_key": True})
    # The dependency owns the versioned record schema. Keep its complete JSON,
    # including provider extensions, inside the core's typed credential envelope.
    payload: str = field(repr=False)

    @property
    def credentials(self) -> dict[str, Any]:
        value = json.loads(self.payload)
        if not isinstance(value, dict):
            raise ValueError("Invalid MCP OAuth credentials.")
        return value

    @classmethod
    def from_credentials(cls, key: str, value: dict[str, Any]) -> OAuthRecord:
        return cls(key, json.dumps(value, allow_nan=False))


def credential_store(path: Path | None = None) -> FileCredentialStore[OAuthRecord]:
    return FileCredentialStore(path or default_credentials_path("mcp.auth"), record_type=OAuthRecord)


def credential_key(config: MCPServerConfig) -> str:
    options = config.oauth or {}
    identity = [config.url, options.get("issuer") or None, options.get("client_id") or None]
    if port := options.get("redirect_port"):
        # A different fixed callback needs its own client registration.
        identity.append(str(port))
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class StoredOAuth(OAuth):
    """Serialize token operations across sessions and processes using the core store."""

    def __init__(self, config: MCPServerConfig, redirect_uri: str, store: FileCredentialStore[OAuthRecord]) -> None:
        assert config.url is not None
        self.store = store
        self.key = credential_key(config)
        self.operation_lock = asyncio.Lock()
        options = config.oauth or {}
        scope = str(options["scope"]).split() if options.get("scope") else None
        super().__init__(
            config.url,
            redirect_uri=redirect_uri,
            issuer=str(options["issuer"]) if options.get("issuer") else None,
            client_id=str(options["client_id"]) if options.get("client_id") else None,
            scopes=scope,
            registration={"client_name": "Axio"},
            load=self.load_credentials,
            save=self.save_credentials,
            allow_http=True,  # The dependency permits only loopback HTTP with this option.
            timeout=config.timeout,
        )

    async def load_credentials(self) -> dict[str, Any] | None:
        record = await self.store.load(self.key)
        return record.credentials if record else None

    async def save_credentials(self, value: dict[str, Any]) -> None:
        await self.store.save(OAuthRecord.from_credentials(self.key, value))

    @asynccontextmanager
    async def coordinated(self) -> AsyncIterator[None]:
        async with self.operation_lock, self.store.lock(self.key):
            # Another process can rotate or remove tokens between requests.
            self.loaded = False
            self.record = None
            yield

    async def token(self, *, rejected: str | None = None) -> str:
        async with self.coordinated():
            return await super().token(rejected=rejected)

    async def begin(self, *, scopes: Sequence[str] | None = None) -> AuthorizationAttempt:
        async with self.coordinated():
            return await super().begin(scopes=scopes)

    async def complete(self, attempt: AuthorizationAttempt, callback_url: str) -> None:
        async with self.coordinated():
            await self.discover()
            await super().complete(attempt, callback_url)

    async def invalidate(self, rejected: str) -> None:
        async with self.coordinated():
            await self.discover()
            await super().invalidate(rejected)


async def saved_redirect(config: MCPServerConfig, store: FileCredentialStore[OAuthRecord]) -> str | None:
    key = credential_key(config)
    async with store.lock(key):
        record = await store.load(key)
    if record:
        redirects = record.credentials.get("registration", {}).get("redirect_uris", [])
        if redirects and isinstance(redirects[0], str):
            uri = URL(redirects[0])
            if uri.scheme == "http" and uri.host == "127.0.0.1" and uri.path == "/callback":
                return str(uri)
        raise ValueError("Stored MCP OAuth callback is invalid.")
    return None


async def open_oauth(config: MCPServerConfig) -> StoredOAuth:
    store = credential_store()
    redirect = await saved_redirect(config, store)
    auth = StoredOAuth(config, redirect or "http://127.0.0.1:8765/callback", store)
    await auth.__aenter__()
    return auth


def callback_response(success: bool) -> web.Response:
    nonce = secrets.token_urlsafe(24)
    template = Template(files(__package__).joinpath("oauth.html").read_text(encoding="utf-8"))
    page = template.substitute(
        title="You're connected." if success else "Sign-in didn't finish.",
        message="You can close this tab and return to Axio." if success else "Return to Axio and try again.",
        status="MCP authorization saved" if success else "MCP authorization was not completed",
        state="success" if success else "error",
        nonce=nonce,
        button_label="Close tab (5)" if success else "Close tab",
    )
    return web.Response(
        status=200 if success else 400,
        text=page,
        content_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": (
                "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                f"script-src 'nonce-{nonce}'; base-uri 'none'; frame-ancestors 'none'"
            ),
        },
    )


class MCPCallback(OAuthCallback):
    @staticmethod
    def response(success: bool, text: str) -> web.Response:
        return callback_response(success)


class MCPBrowserCallback(LoopbackCallback):
    """Keep the dependency's listener and state validation with Axio result pages."""

    async def login(self, auth: OAuth, *, scopes: Sequence[str] | None = None) -> None:
        if auth.redirect_uri != self.redirect_uri or self.callback is not None:
            raise ValueError("The callback is unavailable for this authorization.")
        callback = MCPCallback(auth)
        self.callback = callback
        try:
            async with asyncio.timeout(self.timeout):
                attempt = await callback.begin(scopes=scopes)
                assert self.open_browser is not None
                await self.open_browser(attempt.url)
                await callback.wait()
        finally:
            callback.cancel()
            self.callback = None

    async def handle(self, request: web.Request) -> web.Response:
        if request.host != URL(self.redirect_uri).authority or self.callback is None:
            return callback_response(False)
        return await self.callback.handle(request)


async def sign_in(config: MCPServerConfig, open_browser: Callable[[str], Awaitable[None]]) -> None:
    store = credential_store()
    redirect = await saved_redirect(config, store)
    port = URL(redirect).port if redirect else int((config.oauth or {}).get("redirect_port", 0))
    async with MCPBrowserCallback(port=port or 0, open_browser=open_browser) as callback:
        async with StoredOAuth(config, callback.redirect_uri, store) as auth:
            await callback.login(auth)


async def sign_out(config: MCPServerConfig) -> None:
    """Forget local tokens; preserve the client registration for another sign-in."""
    store = credential_store()
    key = credential_key(config)
    async with store.lock(key):
        record = await store.load(key)
        if record:
            await store.save(
                OAuthRecord.from_credentials(key, {**record.credentials, "tokens": {}, "expires_at": None})
            )


async def authorization_status(config: MCPServerConfig) -> str:
    store = credential_store()
    key = credential_key(config)
    async with store.lock(key):
        record = await store.load(key)
    return "Credentials saved" if record and record.credentials.get("tokens") else "Sign-in required"
