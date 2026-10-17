"""Browse MCP connections and their advertised capabilities."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axio_tools_mcp.inspection import ServerInspection
from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.events import Key, Resize
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, Tree
from textual.widgets.tree import TreeNode


@dataclass(frozen=True, slots=True)
class MCPItem:
    title: str
    description: str = ""
    fields: tuple[tuple[str, str], ...] = ()
    document: str = ""
    children: tuple[MCPItem, ...] = ()

    def filtered(self, query: str) -> MCPItem | None:
        text = f"{self.title}\n{self.description}\n{self.fields}\n{self.document}"
        if query in text.casefold():
            return self
        children = tuple(found for child in self.children if (found := child.filtered(query)) is not None)
        return replace(self, children=children) if children else None


def server_item(server: ServerInspection) -> MCPItem:
    tools = tuple(
        MCPItem(tool.name, tool.description, document=json.dumps(tool.input_schema, indent=2, ensure_ascii=False))
        for tool in sorted(server.tools, key=lambda tool: tool.name)
    )
    resources = tuple(
        MCPItem(
            item.name or item.uri,
            item.description or "",
            (("URI", item.uri),),
            json.dumps(item.model_dump(mode="json", by_alias=True, exclude_none=True), indent=2, ensure_ascii=False),
        )
        for item in server.resources
    )
    templates = tuple(
        MCPItem(
            item.name or item.uri_template,
            item.description or "",
            (("URI template", item.uri_template),),
            json.dumps(item.model_dump(mode="json", by_alias=True, exclude_none=True), indent=2, ensure_ascii=False),
        )
        for item in server.templates
    )
    skills = tuple(
        MCPItem(
            item.name,
            item.description,
            (("URI", item.uri), ("Read instructions", "Open /skills")),
            item.model_dump_json(indent=2),
        )
        for item in server.skills
    )
    groups = tuple(
        MCPItem(f"{name} ({len(items)})", explanation, children=items)
        for name, explanation, items in (
            (
                "Tools",
                "Select a tool to inspect its description and input schema. Tools are not executed here.",
                tools,
            ),
            ("Resources", "Select a resource to inspect its URI and metadata.", resources),
            ("Templates", "Select a template to inspect its URI parameters and metadata.", templates),
            (
                "Skills",
                "Select a skill to inspect its description and file manifest. Read instructions in /skills.",
                skills,
            ),
        )
    )
    fields = (
        ("Source", server.origin),
        ("State", server.state),
        (
            "Catalog",
            f"Tools: {len(tools)} · Resources: {len(resources)} · Templates: {len(templates)} · Skills: {len(skills)}",
        ),
        *server.details,
        *(("Error", error) for error in server.errors),
    )
    return MCPItem(
        f"{server.name} · {server.state} · {server.origin}",
        "Refresh updates this view from current connections. Reconnect all restarts automatic and saved servers. "
        "Use Manage to edit configuration files and authorization.",
        fields,
        children=groups,
    )


class MCPBrowserTree(Tree[MCPItem]):
    BINDINGS = [
        Binding("right", "expand_branch", "Expand", show=False),
        Binding("left", "collapse_branch", "Collapse", show=False),
    ]

    def action_expand_branch(self) -> None:
        node = self.cursor_node
        if node is not None and node.children:
            if node.is_expanded:
                self.move_cursor(node.children[0])
            else:
                node.expand()

    def action_collapse_branch(self) -> None:
        node = self.cursor_node
        if node is not None:
            if node.is_expanded:
                node.collapse()
            elif node.parent is not None and node.parent is not self.root:
                self.move_cursor(node.parent)


class MCPScreen(ModalScreen[None]):
    """Inspect cached MCP catalogs and offer an explicit reconnect action."""

    BINDINGS = [Binding("escape", "close", "Close"), Binding("ctrl+f", "filter", "Filter")]
    CSS = """
    MCPScreen { align: center middle; }
    MCPScreen #mcp-inspector {
        width: 95%; max-width: 160; height: 95%;
        border: heavy $accent; background: $panel; padding: 0 1;
    }
    MCPScreen #mcp-title { height: 1; text-style: bold; }
    MCPScreen #mcp-count { height: 1; color: $text-muted; }
    MCPScreen #mcp-body { height: 1fr; }
    MCPScreen #mcp-tree { width: 2fr; height: 1fr; border: solid $border; }
    MCPScreen #mcp-details { width: 3fr; height: 1fr; padding: 0 1; }
    MCPScreen #mcp-buttons { height: 3; align-horizontal: right; }
    MCPScreen #mcp-buttons Button { margin-left: 1; }
    MCPScreen.narrow #mcp-inspector { width: 100%; height: 100%; }
    MCPScreen.narrow #mcp-body { layout: vertical; }
    MCPScreen.narrow #mcp-tree { width: 1fr; height: 7; }
    MCPScreen.narrow #mcp-details { width: 1fr; height: 1fr; padding: 0; }
    """

    def __init__(
        self,
        inspect_servers: Callable[[], list[ServerInspection]],
        *,
        reconnect: Callable[[], Awaitable[None]] | None = None,
        manage: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._inspect_servers = inspect_servers
        self._reconnect = reconnect
        self._manage = manage
        self._items: tuple[MCPItem, ...] = ()

    def compose(self) -> ComposeResult:
        with Container(id="mcp-inspector"):
            yield Static("MCP Servers", id="mcp-title")
            yield Input(placeholder="Filter servers, tools, resources, or skills", id="mcp-filter")
            yield Static(id="mcp-count")
            with Horizontal(id="mcp-body"):
                yield MCPBrowserTree("MCP", id="mcp-tree")
                with VerticalScroll(id="mcp-details"):
                    yield Static(id="mcp-detail", markup=False)
            if self._manage is not None:
                yield Button("Manage", id="mcp-manage")
            with Horizontal(id="mcp-buttons"):
                yield Button("Refresh", id="mcp-refresh", variant="primary")
                yield Button("Reconnect all", id="mcp-reconnect", disabled=self._reconnect is None)
                yield Button("Close", id="mcp-close")

    def on_mount(self) -> None:
        self.set_class(self.size.width < 90, "narrow")
        self.refresh_servers()
        self.action_filter()

    def on_resize(self, event: Resize) -> None:
        self.set_class(event.size.width < 90, "narrow")

    def refresh_servers(self) -> None:
        try:
            servers = self._inspect_servers()
        except Exception as error:
            self.query_one("#mcp-detail", Static).update(f"Cannot refresh MCP status: {type(error).__name__}")
            return
        self._items = tuple(server_item(server) for server in servers)
        self._filter(self.query_one("#mcp-filter", Input).value)

    def _filter(self, query: str) -> None:
        query = query.casefold().strip()
        items = tuple(found for item in self._items if (found := item.filtered(query)) is not None)
        tree: Tree[MCPItem] = self.query_one("#mcp-tree", Tree)
        tree.clear()
        tree.show_root = False
        tree.root.expand()

        def add(parent: TreeNode[MCPItem], item: MCPItem, *, expand: bool = False) -> TreeNode[MCPItem]:
            node = parent.add(Text(item.title), data=item, expand=expand, allow_expand=bool(item.children))
            for child in item.children:
                add(node, child, expand=bool(query))
            return node

        nodes = [add(tree.root, item, expand=True) for item in items]
        tree.disabled = not nodes
        self.query_one("#mcp-count", Static).update(
            f"{len(items)}/{len(self._items)} servers · ←→ expand · Tab details · Esc close"
        )
        if nodes:
            tree.move_cursor(nodes[0])
            self._show(items[0])
        else:
            text = "No MCP entries match this filter." if self._items else "No MCP servers configured."
            self.query_one("#mcp-detail", Static).update(text)

    def _show(self, item: MCPItem) -> None:
        text = Text()
        text.append(item.title + "\n", style="bold")
        for name, value in item.fields:
            text.append(f"\n{name}: ", style="bold red" if name == "Error" else "bold")
            text.append(value, style="red" if name == "Error" else "")
        if item.description:
            text.append("\n\n" + item.description, style="")
        if item.document:
            content = Group(text, Text(""), Syntax(item.document, "json", word_wrap=True, background_color="default"))
            self.query_one("#mcp-detail", Static).update(content)
        else:
            self.query_one("#mcp-detail", Static).update(text)
        self.query_one("#mcp-details", VerticalScroll).scroll_home(animate=False)

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.query_one("#mcp-tree", Tree).focus()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[MCPItem]) -> None:
        event.stop()
        if event.node.data is not None:
            self._show(event.node.data)

    def on_tree_node_selected(self, event: Tree.NodeSelected[MCPItem]) -> None:
        event.stop()
        if event.node.data is not None:
            self._show(event.node.data)

    def on_key(self, event: Key) -> None:
        if event.key == "down" and self.focused is self.query_one("#mcp-filter", Input):
            self.query_one("#mcp-tree", Tree).focus()
            event.prevent_default()
            event.stop()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "mcp-refresh":
            self.refresh_servers()
        elif event.button.id == "mcp-reconnect" and self._reconnect is not None:
            self.action_close()
            await self._reconnect()
        elif event.button.id == "mcp-manage" and self._manage is not None:
            await self._manage()
        elif event.button.id == "mcp-close":
            self.action_close()

    def action_filter(self) -> None:
        self.query_one("#mcp-filter", Input).focus()

    def action_close(self) -> None:
        self.dismiss(None)
