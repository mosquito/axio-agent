"""Discovery failures, provider collisions, and transport lifecycle contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from axio.models import ModelSpec
from axio.transport import TransportEntry, TransportError, discover_transports


class EntryPoint:
    def __init__(self, name: str, target: Any = None, error: Exception | None = None) -> None:
        self.name, self.target, self.error = name, target, error
        self.loads = 0

    def load(self) -> Any:
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.target


def test_discovery_preserves_failures_and_first_declaration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first = EntryPoint("ok", SimpleNamespace)
    duplicate = EntryPoint("ok", dict)
    broken = EntryPoint("broken", error=ImportError("missing optional dependency"))
    replacement = EntryPoint("broken", dict)
    invalid = EntryPoint("invalid", object())
    monkeypatch.setattr(
        "importlib.metadata.entry_points", lambda **_: [first, duplicate, broken, replacement, invalid]
    )
    catalog = discover_transports()
    assert catalog.require("ok").cls is SimpleNamespace
    assert duplicate.loads == replacement.loads == 0
    assert "first declaration wins" in caplog.text
    with pytest.raises(TransportError, match="broken.*missing optional dependency"):
        catalog.require("broken")
    with pytest.raises(TransportError, match="invalid.*Expected a transport class"):
        catalog.require("invalid")
    with pytest.raises(TransportError, match="Unknown transport.*Available: ok"):
        catalog.require("absent")


def test_empty_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: [])
    catalog = discover_transports()
    assert not catalog.entries and not catalog.errors


async def test_legacy_catalog_load_and_clone_preserve_settings() -> None:
    calls: list[str] = []

    class Legacy:
        def __init__(self, *, session: Any, temperature: float) -> None:
            self.session = session
            self.temperature = temperature
            self.model = ModelSpec("first")
            self.on_auth_refresh: Any = None

        async def fetch_models(self) -> None:
            assert self.on_auth_refresh is callback
            calls.append("fetch")

    def callback() -> None:
        pass

    entry = TransportEntry.from_target("legacy", Legacy)
    session = object()
    async with entry.open(session=session, temperature=0, on_auth_refresh=callback) as transport:
        selected = entry.select_model(transport, ModelSpec("second"))
        assert selected is not transport and selected.session is session
        assert selected.temperature == 0 and transport.model.id == "first"
    assert calls == ["fetch"]


async def test_plugin_instance_owns_factory_and_cleanup() -> None:
    calls: list[Any] = []

    class Plugin:
        @asynccontextmanager
        async def open(self, **settings: Any) -> AsyncIterator[object]:
            calls.append(settings)
            try:
                yield object()
            finally:
                calls.append("closed")

    entry = TransportEntry.from_target("instance", Plugin())
    with pytest.raises(RuntimeError, match="consumer failed"):
        async with entry.open(session=None, account="chosen", credentials="private-file"):
            raise RuntimeError("consumer failed")
    assert calls == [{"session": None, "account": "chosen", "credentials": "private-file"}, "closed"]


def test_metadata_uses_class_attributes_not_dataclass_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Legacy:
        display_name = "Example"
        env_vars = ("TEST_PLUGIN_KEY",)

    entry = TransportEntry.from_target("legacy", Legacy)
    monkeypatch.delenv("TEST_PLUGIN_KEY", raising=False)
    assert not entry.has_credentials()
    assert entry.has_credentials({"TEST_PLUGIN_KEY": "test"})
    monkeypatch.setenv("TEST_PLUGIN_KEY", "test")
    assert entry.has_credentials()
    assert entry.display_name == "Example"
