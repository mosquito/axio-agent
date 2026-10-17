"""Pinned Antigravity accounts backed by the shared core credential stores."""

from __future__ import annotations

import secrets
import time
from dataclasses import replace

from axio.credentials import (
    CredentialConflict,
    CredentialError,
    Credentials,
    CredentialStore,
    FileCredentialStore,
    SQLiteCredentialStore,
    complete_operation,
)

from .oauth import AntigravityAccount, AntigravityOAuth, OAuthError, Tokens


class SignInRequired(CredentialError):
    """The selected account needs Google authorization."""


class AntigravityFileCredentials(FileCredentialStore[AntigravityAccount]):
    filename = "antigravity.auth"


class AntigravitySQLiteCredentials(SQLiteCredentialStore[AntigravityAccount]):
    filename = "antigravity.sqlite3"


async def list_accounts(store: Credentials[AntigravityAccount]) -> list[AntigravityAccount]:
    """Return metadata only; tokens never leave the storage/session boundary."""
    return [replace(account, tokens=None) for account in await store.list_records()]


async def resolve_account(store: Credentials[AntigravityAccount], account_id: str | None = None) -> str:
    selected = account_id if account_id is not None else await store.selected_account()
    if selected is None:
        accounts = await list_accounts(store)
        if len(accounts) != 1:
            raise SignInRequired("Sign in or select an account with axio-antigravity login/select.")
        selected = accounts[0].account_id
    async with store.lock(selected):
        if await store.load(selected) is None:
            raise SignInRequired("The Antigravity account was not found. Run axio-antigravity login.")
    return selected


async def save_account(
    store: CredentialStore[AntigravityAccount],
    account: AntigravityAccount,
    *,
    expected: AntigravityAccount | None = None,
) -> None:
    async def commit() -> None:
        async with store.lock(account.account_id):
            current = await store.load(account.account_id)
            if current != expected:
                raise CredentialConflict("The account changed during sign-in. Select it and sign in again.")
            if expected and (account.subject, account.client_id) != (expected.subject, expected.client_id):
                raise CredentialConflict("Sign-in returned a different Google account.")
            await store.save(account)

    await complete_operation(commit())


class AntigravitySession:
    """Share one selected account across transports without stale token caches."""

    def __init__(self, store: CredentialStore[AntigravityAccount], oauth: AntigravityOAuth, account_id: str) -> None:
        self.store = store
        self.oauth = oauth
        self.account_id = account_id
        self.pending: tuple[AntigravityAccount, AntigravityAccount] | None = None

    async def access_token(self, *, rejected: str | None = None) -> str:
        return await complete_operation(self.refresh_if_needed(rejected))

    async def refresh_if_needed(self, rejected: str | None) -> str:
        async with self.store.lock(self.account_id):
            current = await self.store.load(self.account_id)
            if self.pending is not None:
                previous, updated = self.pending
                if current == previous:
                    await self.store.save(updated)
                    current = updated
                self.pending = None
            if current is None or current.tokens is None:
                raise SignInRequired("Sign in to the selected Antigravity account again.")
            if current.client_id != self.oauth.config.client_id:
                raise SignInRequired("This account uses a different OAuth client configuration.")
            tokens = current.tokens
            if tokens.expires_at > time.time() + 60 and tokens.access_token != rejected:
                return tokens.access_token
            try:
                refreshed = await self.oauth.refresh(tokens)
            except OAuthError as error:
                if error.code == "invalid_grant":
                    await self.store.save(replace(current, tokens=None, revision=secrets.token_hex(16)))
                    raise SignInRequired("Google authorization expired or was revoked. Sign in again.") from None
                raise
            updated = replace(current, tokens=refreshed, revision=secrets.token_hex(16))
            self.pending = current, updated
            await self.store.save(updated)
            self.pending = None
            return refreshed.access_token

    async def logout(self, *, revoke: bool = False) -> bool:
        """Clear local tokens and return whether remote revocation was confirmed."""
        return await complete_operation(self.clear_tokens(revoke))

    async def clear_tokens(self, revoke: bool) -> bool:
        async with self.store.lock(self.account_id):
            current = await self.store.load(self.account_id)
            if current is None:
                self.pending = None
                return False
            tokens: Tokens | None = current.tokens
            if self.pending is not None and self.pending[0] == current:
                tokens = self.pending[1].tokens
            confirmed = await self.oauth.revoke(tokens) if revoke and tokens is not None else False
            await self.store.save(replace(current, tokens=None, revision=secrets.token_hex(16)))
            self.pending = None
            return confirmed
