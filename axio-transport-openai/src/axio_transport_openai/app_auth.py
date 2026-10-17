"""Account operations for applications using the transport entry point."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from .auth import sign_in
from .credential_stores import FileCredentialStore
from .credentials import CredentialError


async def list_accounts(store: FileCredentialStore) -> list[Any]:
    from .accounts import list_accounts as read

    return await read(store)


async def _confirm_chatgpt_login() -> bool:
    """Read terminal confirmation with cancellable input and no worker thread."""
    print("No ChatGPT account is saved. Sign in now? [y/N] ", end="", flush=True)
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[str] = loop.create_future()
    fd = sys.stdin.fileno()

    def read_answer() -> None:
        if answer.done():
            return
        try:
            answer.set_result(sys.stdin.readline())
        except Exception as exc:
            answer.set_exception(exc)

    loop.add_reader(fd, read_answer)
    try:
        return (await answer).strip().lower() in {"y", "yes"}
    finally:
        loop.remove_reader(fd)


async def _offer_chatgpt_login(store: FileCredentialStore, client_id: str | None) -> None:
    """Offer first-run sign-in before opening the REPL's network sessions."""
    if client_id is not None or await store.selected_account() is not None or await list_accounts(store):
        return
    if not sys.stdin.isatty():
        raise CredentialError(
            "No ChatGPT account is saved. Start axio-repl --transport chatgpt in a terminal to sign in."
        )
    if not await _confirm_chatgpt_login():
        raise CredentialError("ChatGPT sign-in was cancelled.")
    account = await sign_in(credentials=store.path)
    await store.select_account(account.client_id)
    print("Signed in with ChatGPT.")


class ChatGPTAuth:
    async def has_accounts(self, credentials: Path | None = None) -> bool:
        return bool(await FileCredentialStore(credentials).list_records())

    async def prompt_login_if_needed(self, credentials: Path | None, account: str | None) -> None:
        await _offer_chatgpt_login(FileCredentialStore(credentials), account)

    async def describe_accounts(self, credentials: Path | None, transport: Any) -> str:
        store = FileCredentialStore(credentials)
        selected = await store.selected_account()
        active = self.account_id(transport)
        lines = []
        for account in await list_accounts(store):
            marks = ", ".join(
                label
                for label, matches in (
                    ("selected", account.client_id == selected),
                    ("active", account.client_id == active),
                )
                if matches
            )
            lines.append(f"{account.client_id}  {account.email or account.subject}  {marks}".rstrip())
        return "\n".join(lines) or "No ChatGPT accounts are saved."

    async def sign_in(self, credentials: Path | None, account: str | None) -> str:
        result = await sign_in(credentials=credentials, client_id=account)
        return result.client_id

    async def select_account(self, credentials: Path | None, account: str) -> None:
        await FileCredentialStore(credentials).select_account(account)

    async def logout(self, transport: Any) -> None:
        await transport.account.logout()

    def account_id(self, transport: Any) -> str:
        return str(transport.account.client_id)
