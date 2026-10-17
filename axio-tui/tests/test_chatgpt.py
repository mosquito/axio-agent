from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
from axio.transport import TransportCatalog, TransportEntry
from axio_transport_openai.chatgpt import ChatGPTTransport
from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.credentials import CredentialRecord, save_account
from axio_transport_openai.oauth import SCOPES, ChatGPTAccount, ChatGPTTokens
from axio_transport_openai.session import ChatGPTSession

from axio_tui.sqlite_config import ProjectConfig, connect_config
from axio_tui.transport_registry import TransportRegistry


def account(client_id: str) -> ChatGPTAccount:
    now = time.time()
    return ChatGPTAccount(
        client_id,
        "host",
        "https://auth.openai.com",
        "subject",
        "same@example.test",
        tokens=ChatGPTTokens(
            "id-token", client_id + "-access", "refresh", "Bearer", frozenset(SCOPES.split()), now, now + 3600
        ),
    )


async def test_registry_shares_live_credentials_and_pins_old_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileCredentialStore(tmp_path / "auth" / "chatgpt.auth")
    await save_account(store, replace(account("one"), host_id=await store.host_id()))
    await save_account(store, replace(account("two"), host_id=await store.host_id()))
    monkeypatch.setattr(
        "axio_tui.transport_registry.discover_transports",
        lambda: TransportCatalog({"chatgpt": TransportEntry.from_target("chatgpt", ChatGPTTransport)}),
    )
    monkeypatch.setattr("axio_tui.transport_registry.discover_transport_settings", lambda: {})

    async def catalog(self: Any) -> dict[str, Any]:
        return {
            "models": [{"slug": "model-" + self.account.client_id, "display_name": "Fixture", "visibility": "list"}]
        }

    monkeypatch.setattr("axio_transport_openai.subscription.ChatGPTResponses.fetch_models", catalog)
    config_connection = await connect_config(tmp_path / "tui.db")
    config = ProjectConfig(config_connection)
    await config.set("transport.chatgpt.credentials", str(store.path))
    await config.set("transport.chatgpt.client_id", "one")
    registry = TransportRegistry()
    sessions: list[ChatGPTSession] = []
    try:
        async with aiohttp.ClientSession() as http:
            await registry.init(http, config=config)
            source = registry.get_transport("chatgpt")
            clone = registry.make_transport("chatgpt", source.model)
            assert clone.account is source.account
            sessions.append(source.account)
            async with store.lock("one"):
                record = await store.load("one")
                assert record is not None and record.account.tokens is not None
                await store.save(
                    CredentialRecord(
                        replace(record.account, tokens=replace(record.account.tokens, access_token="rotated"))
                    )
                )
            assert await clone.account.access_token() == "rotated"
            await registry.save_settings("chatgpt", {"credentials": str(store.path), "client_id": "two"})
            new = registry.get_transport("chatgpt")
            sessions.append(new.account)
            assert new.account.client_id == "two"
            assert clone.account.client_id == "one"
            assert list(new.models.keys()) == ["model-two"]
            assert await clone.account.access_token() == "rotated"
            assert await config.get_prefix("transport.chatgpt.") == {
                "transport.chatgpt.credentials": str(store.path),
                "transport.chatgpt.client_id": "two",
            }
            await registry.close()
            assert all(s._closed for s in sessions)
    finally:
        await registry.close()
        await config_connection.close()


async def test_failed_discovery_keeps_settings_and_retires_old_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    class Managed:
        @classmethod
        @asynccontextmanager
        async def open_for_app(cls, **kwargs: Any) -> AsyncIterator[object]:
            raise ValueError("Permission required")
            yield object()

    monkeypatch.setattr(
        "axio_tui.transport_registry.discover_transports",
        lambda: TransportCatalog({"chatgpt": TransportEntry.from_target("chatgpt", Managed)}),
    )
    monkeypatch.setattr("axio_tui.transport_registry.discover_transport_settings", lambda: {"chatgpt": object})
    registry = TransportRegistry()
    await registry.init(AsyncMock(spec=aiohttp.ClientSession))
    assert not registry.available
    assert "chatgpt" in registry.settings_screens()
    assert registry.errors["chatgpt"] == "Permission required"
    await registry.close()
