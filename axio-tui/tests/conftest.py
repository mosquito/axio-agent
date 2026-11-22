"""Shared test fixtures for axio-tui."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import aiosqlite
import pytest
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.transport import TransportCatalog, TransportEntry
from axio_context_sqlite import connect
from axio_tools_local.paths import LocalPaths as Workspace
from pydantic import BaseModel

from axio_tui.app import AgentApp
from axio_tui.options import StartupOptions


@pytest.fixture
async def db_conn(tmp_path: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Open a fresh context database connection for the test."""
    conn = await connect(tmp_path / "test.db")
    yield conn
    await conn.close()


class Nothing(BaseModel):
    """Empty argument model for MCP fixtures."""


class FakeTransport:
    """A transport with one model and no network access."""

    name = "Fixture"
    api_key = ""
    model = ModelSpec(id="fixture", capabilities=frozenset({Capability.text, Capability.tool_use}))
    models = ModelRegistry([model])

    def __init__(self, **kwargs: object) -> None:
        self.session = kwargs.get("session")

    async def fetch_models(self) -> None:
        pass


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentApp:
    """An application with its own configuration, no plugins and one fake transport."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr("axio_tui.app.DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr("axio_tui.app.discover_tools_plugins", lambda: {})
    monkeypatch.setattr("axio_tui.app.discover_guards", lambda: {})
    monkeypatch.setattr("axio_tui.transport_registry.discover_transport_settings", lambda: {})
    monkeypatch.setattr(
        "axio_tui.transport_registry.discover_transports",
        lambda: TransportCatalog({"fake": TransportEntry.from_target("fake", FakeTransport)}),
    )
    return AgentApp(
        paths=Workspace.discover(tmp_path),
        options=StartupOptions(skill_directories=(tmp_path / "skills",)),
    )
