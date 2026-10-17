"""Account selection and session lifetime for terminal applications."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace

import aiohttp

from .chatgpt import ChatGPTTransport
from .credential_stores import LocalCredentialStore
from .credentials import CredentialError
from .oauth import ChatGPTAccount, ChatGPTOAuth
from .session import ChatGPTSession


async def list_accounts(store: LocalCredentialStore) -> list[ChatGPTAccount]:
    """Return account metadata without exposing persisted tokens."""
    return [replace(record.account, tokens=None) for record in await store.list_records()]


async def resolve_account(store: LocalCredentialStore, client_id: str | None = None) -> str:
    """Use an explicit registration, saved selection, or the only saved account."""
    selected = client_id if client_id is not None else await store.selected_account()
    if selected is None:
        accounts = await list_accounts(store)
        if len(accounts) != 1:
            raise CredentialError("Select an account with axio-chatgpt select, or sign in with axio-chatgpt login.")
        selected = accounts[0].client_id
    async with store.lock(selected):
        if await store.load(selected) is None:
            raise CredentialError("The selected registration was not found. Run axio-chatgpt accounts.")
    return selected


@asynccontextmanager
async def open_chatgpt(
    store: LocalCredentialStore,
    http: aiohttp.ClientSession,
    *,
    client_id: str | None = None,
    model: str | None = None,
) -> AsyncIterator[ChatGPTTransport]:
    """Open a pinned account session, fetch its models, and close renewal on exit.

    The caller owns the HTTP session. Closing this context retains credentials.
    Model selection is exact; hidden and unavailable models are rejected.
    """
    selected = await resolve_account(store, client_id)
    oauth = ChatGPTOAuth(http, await store.host_id())
    async with ChatGPTSession(store, oauth, selected) as account:
        transport = ChatGPTTransport(account=account, session=http)
        await transport.fetch_models()
        if not transport.models:
            raise CredentialError("This ChatGPT account has no visible models.")
        if model is not None:
            if model not in transport.models:
                raise CredentialError("The model is not visible for this account. Run axio-chatgpt models.")
            transport = transport.with_model(model)
        yield transport
