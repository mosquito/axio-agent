from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from aiohttp_tiny_mcp.protocol.models import ResourceDef, ResourceTemplateDef
from axio.tool import Tool
from axio_tools_mcp.inspection import ServerInspection
from axio_tools_mcp.skills import Skill
from rich.console import Console
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Input, Static, Tree
from textual.widgets.tree import TreeNode

from axio_tui.mcp import MCPItem, MCPScreen


def detail_text(screen: MCPScreen) -> str:
    console = Console(width=100, color_system=None)
    with console.capture() as capture:
        console.print(screen.query_one("#mcp-detail", Static).content)
    return capture.get()


def find_node(tree: Tree[MCPItem], title: str) -> TreeNode[MCPItem]:
    pending = list(tree.root.children)
    while pending:
        node = pending.pop()
        if node.data is not None and node.data.title == title:
            return node
        pending.extend(node.children)
    raise AssertionError(f"Missing tree node: {title}")


def fixture_server() -> ServerInspection:
    async def lookup(query: str) -> str:
        """Find a document [literal]."""
        raise AssertionError("The inspector must not execute tools")

    tool: Tool[Any] = Tool(name="docs__lookup", handler=lookup)
    return ServerInspection(
        name="docs",
        state="connected",
        origin="Automatic · project",
        details=(("Configuration", "/project/.mcp.json"), ("Negotiated protocol", "2025-06-18")),
        tools=(tool,),
        resources=(ResourceDef(name="Guide", uri="notes://guide", description="Complete guide"),),
        templates=(
            ResourceTemplateDef(name="Topic", uri_template="notes://topics/{name}", description="Topic details"),
        ),
        skills=(
            Skill(
                uri="skill://review/SKILL.md",
                frontmatter={"name": "review", "description": "Review changes"},
                resources=[],
            ),
        ),
    )


@pytest.mark.parametrize("size", [(120, 40), (60, 24)])
async def test_filter_schema_metadata_navigation_and_close(size: tuple[int, int]) -> None:
    app: App[None] = App()
    server = fixture_server()
    screen = MCPScreen(lambda: [server, replace(server, origin="Manual · global")])
    async with app.run_test(size=size) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        tree: Tree[MCPItem] = screen.query_one("#mcp-tree", Tree)
        assert len(tree.root.children) == 2
        assert tree.root.children[0].is_expanded
        assert "2025-06-18" in detail_text(screen)
        assert screen.has_class("narrow") == (size[0] < 90)
        await pilot.press("enter", "down", "right", "down")
        assert "Find a document [literal]." in detail_text(screen)
        await pilot.press("ctrl+f")
        await pilot.press("l", "o", "o", "k", "u", "p", "enter")
        assert tree.root.children[0].is_expanded
        tree.select_node(find_node(tree, "docs__lookup"))
        await pilot.pause()
        detail = detail_text(screen)
        assert "Find a document [literal]." in detail
        assert '"query"' in detail and '"string"' in detail
        await pilot.press("tab", "pagedown")
        assert screen.query_one("#mcp-details", VerticalScroll).has_focus
        if size[0] < 90:
            assert screen.query_one("#mcp-details", VerticalScroll).scroll_y > 0
        await pilot.press("ctrl+f")
        screen.query_one("#mcp-filter", Input).value = "notes://topics"
        await pilot.pause()
        tree.select_node(find_node(tree, "Topic"))
        await pilot.pause()
        assert "notes://topics/{name}" in detail_text(screen)
        screen.query_one("#mcp-filter", Input).value = "review"
        await pilot.pause()
        tree.select_node(find_node(tree, "review"))
        await pilot.pause()
        assert "skill://review/SKILL.md" in detail_text(screen)
        for selector in ("#mcp-tree", "#mcp-details", "#mcp-refresh", "#mcp-close"):
            region = screen.query_one(selector).region
            assert region.width > 0 and region.height > 0
            assert 0 <= region.x < region.right <= size[0]
            assert 0 <= region.y < region.bottom <= size[1]
        await pilot.press("escape")
        assert app.screen is not screen


async def test_refresh_and_empty_states() -> None:
    servers: list[ServerInspection] = []
    screen = MCPScreen(lambda: servers)
    app: App[None] = App()
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert "No MCP servers configured" in detail_text(screen)
        servers.append(replace(fixture_server(), state="error", errors=("Connection failed: TimeoutError",)))
        await pilot.click("#mcp-refresh")
        assert "TimeoutError" in detail_text(screen)
        await pilot.press("ctrl+f", "z", "z", "z")
        assert "No MCP entries match" in detail_text(screen)
        screen.query_one("#mcp-filter", Input).value = ""
        servers.clear()
        await pilot.click("#mcp-refresh")
        assert "No MCP servers configured" in detail_text(screen)
        await pilot.click("#mcp-close")
        assert app.screen is not screen


async def test_refresh_failure_is_contained() -> None:
    def inspect() -> list[ServerInspection]:
        raise ValueError("secret connection parameters")

    app: App[None] = App()
    async with app.run_test() as pilot:
        screen = MCPScreen(inspect)
        await app.push_screen(screen)
        await pilot.pause()
        assert "ValueError" in detail_text(screen)
        assert "secret connection parameters" not in detail_text(screen)
        await pilot.press("escape")
