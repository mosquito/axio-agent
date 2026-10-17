from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Static, Switch

from axio_tui.configuration import load_mcp_config
from axio_tui.mcp import MCPScreen
from axio_tui.mcp_files import MCPDocument
from axio_tui.mcp_manage import MCPFileEditScreen, MCPManagementScreen, destinations
from axio_tui.paths import ConfigPaths


def test_persistence_precedence_conflicts_and_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    paths = ConfigPaths.discover(root)
    targets = destinations(paths)
    home, project, private = [MCPDocument.read(t.path, t.layer) for t in targets]
    home.write("docs", {"url": "https://user.example/mcp", "oauth": {}})
    project.write("docs", {"url": "https://project.example/mcp", "headersFrom": {"Authorization": "TOKEN"}})
    private.write("docs", {"enabled": False})
    loaded = load_mcp_config(paths)[0]
    assert not loaded.effective["docs"].enabled
    assert [server.layer for server in loaded.history("docs")] == ["global", "project", "private"]
    assert home.path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="changed"):
        home.write("other", {"command": "python"})
    fresh = MCPDocument.read(project.path, "project")
    with pytest.raises(ValueError, match="already exists"):
        fresh.write("docs", {"url": "https://other.example/mcp"})
    fresh.write("second", {"command": "python", "args": ["argument,with,commas"]})
    fresh = MCPDocument.read(project.path, "project")
    assert fresh.servers["docs"]["headersFrom"] == {"Authorization": "TOKEN"}
    fresh.write("renamed", fresh.servers["second"], original="second")
    assert set(MCPDocument.read(project.path, "project").servers) == {"docs", "renamed"}
    fresh = MCPDocument.read(private.path, "private")
    fresh.write("docs", None, original="docs")
    assert load_mcp_config(paths)[0].effective["docs"].layer == "project"
    link = tmp_path / "link.json"
    link.symlink_to(project.path)
    with pytest.raises(ValueError, match="symlink"):
        MCPDocument.read(link, "project")


@pytest.mark.parametrize("size", [(120, 45), (60, 24)])
async def test_manage_add_edit_authorization_and_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    targets = destinations(ConfigPaths.discover(root))
    manager = MCPManagementScreen(targets)
    app: App[None] = App()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(manager)
        await pilot.pause()
        assert manager.query_one("#mcp-edit", Button).disabled
        await pilot.click("#mcp-add")
        await pilot.pause()
        editor = app.screen_stack[-1]
        assert isinstance(editor, MCPFileEditScreen)
        editor.query_one("#mcp-edit-name", Input).value = "docs"
        editor.query_one("#mcp-edit-url", Input).value = "https://example.com/mcp"
        editor.query_one("#mcp-destination", Select).value = 0
        editor.query_one("#mcp-auth-mode", Select).value = "oauth"
        await pilot.pause()
        editor.query_one("#mcp-file-save", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is manager
        assert json.loads(targets[0].path.read_text())["mcpServers"]["docs"]["oauth"] == {"redirect_port": 0}
        assert not manager.query_one("#mcp-authorize", Button).disabled
        manager.query_one("#mcp-edit", Button).press()
        await pilot.pause()
        editor = app.screen_stack[-1]
        assert isinstance(editor, MCPFileEditScreen)
        editor.query_one("#mcp-enabled", Switch).value = False
        editor.query_one("#mcp-file-save", Button).press()
        await pilot.pause()
        assert not json.loads(targets[0].path.read_text())["mcpServers"]["docs"]["enabled"]
        manager.query_one("#mcp-edit", Button).press()
        await pilot.pause()
        editor = app.screen_stack[-1]
        editor.query_one("#mcp-file-delete", Button).press()
        await pilot.pause()
        assert app.screen is editor
        editor.query_one("#mcp-file-delete", Button).press()
        await pilot.pause()
        assert app.screen is manager
        assert json.loads(targets[0].path.read_text())["mcpServers"] == {}
        assert manager.changed
        await pilot.press("escape")


async def test_inspector_manage_is_accessible_in_empty_state() -> None:
    called = []

    async def manage() -> None:
        called.append(True)

    app: App[None] = App()
    screen = MCPScreen(lambda: [], manage=manage)
    async with app.run_test(size=(60, 24)) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.click("#mcp-manage")
        assert called == [True]
        assert app.screen is screen


async def test_header_edit_preserves_other_fields_and_reports_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    targets = destinations(ConfigPaths.discover(tmp_path))
    doc = MCPDocument.read(targets[1].path, "project")
    doc.write(
        "docs",
        {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer secret", "X-Mode": "test"},
            "timeout": 42,
            "headersFrom": {"X-Token": "TOKEN"},
        },
    )
    doc = MCPDocument.read(doc.path, doc.layer)
    app: App[None] = App()
    async with app.run_test() as pilot:
        editor = MCPFileEditScreen(targets, document=doc, name="docs")
        await app.push_screen(editor)
        await pilot.pause()
        assert editor.query_one("#mcp-authorization", Input).password
        editor.query_one("#mcp-edit-url", Input).value = "invalid"
        editor.query_one("#mcp-file-save", Button).press()
        await pilot.pause()
        assert app.screen is editor
        assert "HTTP(S)" in str(editor.query_one("#mcp-edit-error", Static).content)
        editor.query_one("#mcp-edit-url", Input).value = "https://example.com/new"
        editor.query_one("#mcp-file-save", Button).press()
        await pilot.pause()
    entry = MCPDocument.read(doc.path, doc.layer).servers["docs"]
    assert entry["headers"] == {"Authorization": "Bearer secret", "X-Mode": "test"}
    assert entry["headersFrom"] == {"X-Token": "TOKEN"} and entry["timeout"] == 42


async def test_authorization_screen_cancels_worker_on_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from collections.abc import Awaitable, Callable

    from axio_tools_mcp.config import MCPServerConfig

    from axio_tui.mcp_manage import MCPAuthorizationScreen

    started, cancelled = asyncio.Event(), asyncio.Event()

    async def login(config: MCPServerConfig, open_browser: Callable[[str], Awaitable[None]]) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("axio_tui.mcp_manage.sign_in", login)
    screen = MCPAuthorizationScreen(MCPServerConfig("docs", url="https://example.com/mcp", oauth={}))
    app: App[None] = App()
    async with app.run_test(size=(60, 24)) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.click("#mcp-sign-in")
        await asyncio.wait_for(started.wait(), 2)
        assert screen.query_one("#mcp-sign-out", Button).disabled
        await pilot.press("escape")
        await asyncio.wait_for(cancelled.wait(), 2)
        assert app.screen is not screen
        assert not screen.changed


async def test_disabled_override_can_be_renamed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    targets = destinations(ConfigPaths.discover(tmp_path))
    doc = MCPDocument.read(targets[1].path, "project")
    doc.write("disabled", {"enabled": False})
    doc = MCPDocument.read(doc.path, doc.layer)
    app: App[None] = App()
    async with app.run_test() as pilot:
        screen = MCPFileEditScreen(targets, document=doc, name="disabled")
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#mcp-edit-name", Input).value = "renamed"
        screen.query_one("#mcp-file-save", Button).press()
        await pilot.pause()
        assert app.screen is not screen
    assert MCPDocument.read(doc.path, doc.layer).servers == {"renamed": {"enabled": False}}
