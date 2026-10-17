"""Antigravity account operations for transport plugins."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from axio.credentials import CredentialError

from .session import AntigravityFileCredentials


class AntigravityAuth:
    async def has_accounts(self, credentials: Path | None = None) -> bool:
        return bool(await AntigravityFileCredentials(credentials).list_records())

    async def prompt_login_if_needed(self, credentials: Path | None, account: str | None) -> None:
        from .cli import sign_in

        store = AntigravityFileCredentials(credentials)
        if account is not None or await store.list_records():
            return
        if not sys.stdin.isatty():
            raise CredentialError("No Antigravity account is saved. Run axio-antigravity login first.")
        print("No Antigravity account is saved. Starting Google sign-in.")
        await sign_in(store)

    async def describe_accounts(self, credentials: Path | None, transport: Any) -> str:
        from .cli import describe_accounts

        return await describe_accounts(AntigravityFileCredentials(credentials), self.account_id(transport))

    async def sign_in(self, credentials: Path | None, account: str | None) -> str:
        from .cli import sign_in

        result = await sign_in(AntigravityFileCredentials(credentials), account_id=account)
        return result.account_id

    async def select_account(self, credentials: Path | None, account: str) -> None:
        await AntigravityFileCredentials(credentials).select_account(account)

    async def logout(self, transport: Any) -> None:
        await transport.account.logout()

    def account_id(self, transport: Any) -> str:
        return str(transport.account.account_id)
