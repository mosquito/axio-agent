from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.tui.screens import MCPServerEditScreen


@pytest.mark.parametrize("size", [(100, 50), (60, 24)])
async def test_edit_timeout_preserves_connection_settings(size: tuple[int, int]) -> None:
    app: App[None] = App()
    config = MCPServerConfig(name="docs", command="python", timeout=12, cwd="/project", inherit_env=False)
    results: list[MCPServerConfig | None] = []
    screen = MCPServerEditScreen(config)
    async with app.run_test(size=size) as pilot:
        await app.push_screen(screen, results.append)
        await pilot.pause()
        timeout = screen.query_one("#mcp-timeout", Input)
        assert timeout.value == "12"
        timeout.value = "0"
        screen._save()
        assert not results and app.screen is screen
        timeout.value = "not a number"
        screen._save()
        assert not results and app.screen is screen
        timeout.value = "2.5"
        screen._save()
        await pilot.pause()
        assert len(results) == 1
        saved = results[0]
        assert saved is not None and saved.timeout == 2.5
        assert saved.cwd == config.cwd and saved.inherit_env is False
