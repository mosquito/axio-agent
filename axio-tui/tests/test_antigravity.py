from __future__ import annotations

import time
from pathlib import Path

import aiohttp
import pytest
from axio.models import ModelRegistry, ModelSpec
from axio.transport import TransportCatalog, TransportEntry
from axio_transport_google.antigravity import (
    AntigravityAccount,
    AntigravityFileCredentials,
    AntigravityTransport,
    OAuthConfig,
    Tokens,
    save_account,
)
from axio_transport_google.tui.antigravity import AntigravitySettingsScreen

from axio_tui.plugin import discover_transport_settings, discover_transports
from axio_tui.sqlite_config import ProjectConfig, connect_config
from axio_tui.transport_registry import TransportRegistry


def test_entry_points() -> None:
    assert discover_transports().entries["antigravity"].cls is AntigravityTransport
    assert discover_transport_settings()["antigravity"] is AntigravitySettingsScreen


async def test_registry_account_switch_and_signed_out_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AntigravityFileCredentials(tmp_path / "auth")
    for name in ("one", "two"):
        await save_account(
            store,
            AntigravityAccount(
                name, name, OAuthConfig().client_id, tokens=Tokens(name + "-access", "refresh", time.time() + 3600)
            ),
        )
    monkeypatch.setattr(
        "axio_tui.transport_registry.discover_transports",
        lambda: TransportCatalog({"antigravity": TransportEntry.from_target("antigravity", AntigravityTransport)}),
    )
    monkeypatch.setattr(
        "axio_tui.transport_registry.discover_transport_settings", lambda: {"antigravity": AntigravitySettingsScreen}
    )

    async def fetch(self: AntigravityTransport) -> ModelRegistry:
        assert self.account is not None
        await self.account.access_token()
        self.models = ModelRegistry([ModelSpec("gemini-" + self.account.account_id)])
        self.model = self.models.first()
        return self.models

    monkeypatch.setattr(AntigravityTransport, "fetch_models", fetch)
    db = await connect_config(tmp_path / "config.db")
    config = ProjectConfig(db)
    await config.set("transport.antigravity.credentials", str(store.path))
    await config.set("transport.antigravity.account_id", "one")
    registry = TransportRegistry()
    try:
        async with aiohttp.ClientSession() as http:
            await registry.init(http, config=config)
            source = registry.get_transport("antigravity")
            clone = registry.make_transport("antigravity", source.model)
            assert clone.account is source.account
            await registry.save_settings("antigravity", {"credentials": str(store.path), "account_id": "two"})
            current = registry.get_transport("antigravity")
            assert current.account.account_id == "two" and clone.account.account_id == "one"
            assert await clone.account.access_token() == "one-access"
            assert current.model.id == "gemini-two"
            await current.account.logout()
            await registry.save_settings("antigravity", {"credentials": str(store.path), "account_id": "two"})
            assert "antigravity" not in registry.available
            assert registry.settings_screens()["antigravity"] is AntigravitySettingsScreen
            assert "Sign in" in registry.errors["antigravity"]
            assert await config.get_prefix("transport.antigravity.") == {
                "transport.antigravity.credentials": str(store.path),
                "transport.antigravity.account_id": "two",
            }
    finally:
        await registry.close()
        await db.close()
