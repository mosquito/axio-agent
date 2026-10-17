"""Manage MCP files and authorization from the MCP inspector."""

from __future__ import annotations

import asyncio
import json
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.oauth import authorization_status, sign_in, sign_out
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Select, Static, Switch, TextArea

from .configuration import Layer
from .mcp_files import MCPDocument
from .paths import ConfigurationPaths, config_directory


@dataclass(frozen=True, slots=True)
class Destination:
    label: str
    path: Path
    layer: Layer


def destinations(paths: ConfigurationPaths, explicit: tuple[Path, ...] = ()) -> tuple[Destination, ...]:
    result = [Destination("User (~)", config_directory() / "mcp.json", "global")]
    for root in paths.roots:
        label = "Project (.)" if len(paths.roots) == 1 else str(root)
        result.extend(
            [
                Destination(label, root / ".mcp.json", "project"),
                Destination(label + " · private", root / ".axio/mcp.local.json", "private"),
            ]
        )
    result.extend(Destination("Explicit", paths.resolve(path), "explicit") for path in explicit)
    return tuple({item.path: item for item in result}.values())


class MCPFileEditScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    MCPFileEditScreen { align: center middle; }
    #mcp-file-editor { width: 90; max-width: 100%; height: 95%; border: heavy $accent;
        background: $panel; padding: 1 2; }
    #mcp-file-editor Input, #mcp-file-editor Select { margin-bottom: 1; }
    #mcp-extra { height: 10; margin-bottom: 1; }
    #mcp-http-fields, #mcp-stdio-fields { height: auto; }
    #mcp-file-actions { height: auto; }
    #mcp-file-actions Button { min-width: 10; margin-right: 1; }
    #mcp-edit-error { color: $error; height: auto; }
    """

    def __init__(
        self, targets: tuple[Destination, ...], *, document: MCPDocument | None = None, name: str | None = None
    ) -> None:
        super().__init__()
        self.targets = targets
        self.document = document
        self.original = name
        self.entry: dict[str, Any] = dict(document.servers[name]) if document is not None and name else {}
        self.delete_pending = False
        self.snapshots: dict[Path, MCPDocument] = {}

    def compose(self) -> ComposeResult:
        entry = self.entry
        headers = dict(entry.get("headers", {}))
        authorization = next((headers.pop(key) for key in list(headers) if key.lower() == "authorization"), "")
        oauth = entry.get("oauth")
        extra = {
            k: v
            for k, v in entry.items()
            if k not in {"type", "command", "args", "url", "enabled", "headers", "oauth"}
        }
        if headers:
            extra["headers"] = headers
        with VerticalScroll(id="mcp-file-editor"):
            yield Static("Edit MCP server" if self.original else "Add MCP server")
            yield Static("Configuration file")
            selected = next(
                (i for i, target in enumerate(self.targets) if self.document and target.path == self.document.path), 1
            )
            yield Select(
                [(f"{target.label} · {target.path}", i) for i, target in enumerate(self.targets)],
                value=min(selected, len(self.targets) - 1),
                allow_blank=False,
                disabled=self.document is not None,
                id="mcp-destination",
            )
            yield Static("Name")
            yield Input(self.original or "", id="mcp-edit-name")
            yield Static("Enabled")
            yield Switch(entry.get("enabled", True), id="mcp-enabled")
            yield Static("Transport")
            yield Select(
                [("HTTP", "http"), ("stdio", "stdio")],
                allow_blank=False,
                value="stdio" if "command" in entry else "http",
                id="mcp-edit-type",
            )
            with VerticalScroll(id="mcp-http-fields"):
                yield Static("Server URL")
                yield Input(entry.get("url", ""), placeholder="https://example.com/mcp", id="mcp-edit-url")
                yield Static("Authorization")
                yield Select(
                    [
                        ("None / custom headers", "none"),
                        ("Authorization header", "header"),
                        ("OAuth (browser sign-in)", "oauth"),
                    ],
                    allow_blank=False,
                    value="oauth" if oauth is not None else "header" if authorization else "none",
                    id="mcp-auth-mode",
                )
                yield Input(
                    authorization,
                    placeholder="Bearer token or Basic credentials",
                    password=True,
                    id="mcp-authorization",
                )
                yield Static(
                    "OAuth client ID and issuer (optional; blank uses server registration)", id="mcp-oauth-label"
                )
                yield Input(str((oauth or {}).get("client_id", "")), placeholder="Client ID", id="mcp-client-id")
                yield Input(
                    str((oauth or {}).get("issuer", "")), placeholder="https://issuer.example.com", id="mcp-issuer"
                )
                yield Input(
                    str((oauth or {}).get("scope", "")), placeholder="Scopes (space-separated)", id="mcp-oauth-scope"
                )
                yield Input(
                    str((oauth or {}).get("redirect_port", 0)),
                    placeholder="Callback port; 0 selects a free port",
                    id="mcp-oauth-port",
                )
            with VerticalScroll(id="mcp-stdio-fields"):
                yield Static("Command")
                yield Input(entry.get("command", ""), placeholder="uvx", id="mcp-edit-command")
                yield Static("Arguments (JSON array)")
                yield Input(json.dumps(entry.get("args", [])), id="mcp-edit-args")
            yield Static(
                "Additional options (JSON): env, envFrom, headers, headersFrom, cwd, timeout, protocol_version"
            )
            yield TextArea(json.dumps(extra, indent=2), language="json", id="mcp-extra")
            yield Static("OAuth tokens are stored separately. Header values are saved in this configuration file.")
            yield Static("", markup=False, id="mcp-edit-error")
            with Horizontal(id="mcp-file-actions"):
                yield Button("Save", id="mcp-file-save", variant="primary")
                if self.original:
                    yield Button("Delete", id="mcp-file-delete", variant="error")
                yield Button("Cancel", id="mcp-file-cancel")

    def on_mount(self) -> None:
        self.update_fields()
        self.query_one("#mcp-edit-name", Input).focus()
        for target in self.targets:
            try:
                self.snapshots[target.path] = MCPDocument.read(target.path, target.layer)
            except (ValueError, OSError):
                self.query_one("#mcp-edit-error", Static).update(
                    "Some files cannot be edited. Check their format and permissions."
                )
        if self.document:
            self.snapshots[self.document.path] = self.document

    def update_fields(self) -> None:
        http = self.query_one("#mcp-edit-type", Select).value == "http"
        self.query_one("#mcp-http-fields").display = http
        self.query_one("#mcp-stdio-fields").display = not http
        mode = self.query_one("#mcp-auth-mode", Select).value
        self.query_one("#mcp-authorization").display = mode == "header"
        for name in ("client-id", "issuer", "oauth-scope", "oauth-port", "oauth-label"):
            self.query_one(f"#mcp-{name}").display = mode == "oauth"

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        self.update_fields()

    def value(self, selector: str) -> str:
        return self.query_one(selector, Input).value.strip()

    def configuration(self) -> tuple[str, dict[str, Any]]:
        name = self.value("#mcp-edit-name")
        try:
            extra = json.loads(self.query_one("#mcp-extra", TextArea).text)
        except ValueError:
            raise ValueError("Additional options must be valid JSON.") from None
        if not isinstance(extra, dict):
            raise ValueError("Additional options must be a JSON object.")
        if set(extra) & {"type", "command", "args", "url", "enabled", "oauth"}:
            raise ValueError("Use the form fields for transport, endpoint, arguments, enabled state, and OAuth.")
        entry = dict(extra)
        entry["enabled"] = self.query_one("#mcp-enabled", Switch).value
        if not entry["enabled"] and not self.value("#mcp-edit-url") and not self.value("#mcp-edit-command"):
            MCPServerConfig.from_dict(name, entry)
            return name, entry
        entry["type"] = self.query_one("#mcp-edit-type", Select).value
        if entry["type"] == "http":
            entry["url"] = self.value("#mcp-edit-url")
            mode = self.query_one("#mcp-auth-mode", Select).value
            if mode == "header":
                authorization = self.value("#mcp-authorization")
                if not authorization:
                    raise ValueError("Enter an Authorization header value.")
                entry["headers"] = {**entry.get("headers", {}), "Authorization": authorization}
            elif mode == "oauth":
                try:
                    port = int(self.value("#mcp-oauth-port"))
                except ValueError:
                    raise ValueError("Callback port must be an integer.") from None
                entry["oauth"] = {"redirect_port": port}
                for field_name, selector in (
                    ("client_id", "client-id"),
                    ("issuer", "issuer"),
                    ("scope", "oauth-scope"),
                ):
                    if value := self.value(f"#mcp-{selector}"):
                        entry["oauth"][field_name] = value
        else:
            entry["command"] = self.value("#mcp-edit-command")
            try:
                entry["args"] = json.loads(self.value("#mcp-edit-args"))
            except ValueError:
                raise ValueError("Arguments must be a JSON array of strings.") from None
        MCPServerConfig.from_dict(name, entry)
        return name, entry

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "mcp-file-cancel":
            self.dismiss(False)
            return
        try:
            if event.button.id == "mcp-file-delete" and self.document and self.original:
                if not self.delete_pending:
                    self.delete_pending = True
                    event.button.label = "Confirm delete"
                    self.query_one("#mcp-edit-error", Static).update(
                        "Delete this entry? A lower-priority entry with the same name can become active."
                    )
                    return
                self.document.write(self.original, None, original=self.original)
            elif event.button.id == "mcp-file-save":
                target = self.targets[int(str(self.query_one("#mcp-destination", Select).value))]
                document = self.snapshots.get(target.path)
                if document is None:
                    raise ValueError("Cannot edit this configuration. Reopen the editor after fixing the file.")
                name, entry = self.configuration()
                document.write(name, entry, original=self.original)
            else:
                return
        except (ValueError, OSError, TypeError) as error:
            message = str(error) if isinstance(error, ValueError) else "Cannot save this configuration file."
            self.query_one("#mcp-edit-error", Static).update(message)
            return
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class MCPAuthorizationScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "close", "Close")]
    CSS = """
    MCPAuthorizationScreen { align: center middle; }
    #mcp-auth { width: 85; max-width: 100%; height: auto; max-height: 95%;
        border: heavy $accent; background: $panel; padding: 1 2; }
    #mcp-auth-actions { height: auto; }
    #mcp-auth-actions Button { min-width: 10; margin-right: 1; }
    #mcp-auth-url { margin: 1 0; }
    """

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__()
        self.config = config
        self.changed = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="mcp-auth"):
            yield Static(Text(f"Authorize MCP · {self.config.name}"))
            yield Static("", markup=False, id="mcp-auth-status")
            yield Input(placeholder="Browser sign-in URL appears here", id="mcp-auth-url", select_on_focus=True)
            with Horizontal(id="mcp-auth-actions"):
                yield Button("Sign in", id="mcp-sign-in", variant="primary")
                yield Button("Sign out", id="mcp-sign-out")
                yield Button("Close", id="mcp-auth-close")

    async def on_mount(self) -> None:
        try:
            status = await authorization_status(self.config)
        except Exception:
            status = "Cannot read saved credentials."
        self.query_one("#mcp-auth-status", Static).update(status)

    async def open_browser(self, url: str) -> None:
        self.query_one("#mcp-auth-url", Input).value = url
        opened = await asyncio.to_thread(webbrowser.open, url)
        self.query_one("#mcp-auth-status", Static).update(
            "Complete sign-in in your browser." if opened else "Open the URL in a browser on this computer."
        )

    async def authorize(self, *, logout: bool = False) -> None:
        for selector in ("#mcp-sign-in", "#mcp-sign-out"):
            self.query_one(selector, Button).disabled = True
        self.query_one("#mcp-auth-status", Static).update("Signing out…" if logout else "Preparing sign-in…")
        try:
            if logout:
                await sign_out(self.config)
            else:
                await sign_in(self.config, self.open_browser)
            self.changed = True
            self.query_one("#mcp-auth-status", Static).update(
                "Signed out locally." if logout else "Authorization saved."
            )
        except Exception as error:
            # Network exceptions can contain URLs; expose only known sanitized OAuth errors.
            from aiohttp_tiny_mcp import OAuthError

            message = (
                str(error) if isinstance(error, OAuthError) else f"Authorization failed ({type(error).__name__})."
            )
            self.query_one("#mcp-auth-status", Static).update(message)
        finally:
            if self.is_mounted:
                self.query_one("#mcp-auth-url", Input).value = ""
                for selector in ("#mcp-sign-in", "#mcp-sign-out"):
                    self.query_one(selector, Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "mcp-auth-close":
            self.action_close()
        elif event.button.id in {"mcp-sign-in", "mcp-sign-out"}:
            self.run_worker(self.authorize(logout=event.button.id == "mcp-sign-out"), group="oauth", exclusive=True)

    def action_close(self) -> None:
        self.workers.cancel_group(self, "oauth")
        self.dismiss(self.changed)


class MCPManagementScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "close", "Close")]
    CSS = """
    MCPManagementScreen { align: center middle; }
    #mcp-manager { width: 100; max-width: 100%; height: 90%; border: heavy $accent;
        background: $panel; padding: 1 2; }
    #mcp-file-list { height: 1fr; min-height: 5; }
    #mcp-manager-actions { height: auto; }
    #mcp-manager-actions Button { min-width: 10; margin-right: 1; }
    #mcp-manager-message { height: auto; }
    """

    def __init__(
        self,
        targets: tuple[Destination, ...],
        *,
        isolated: bool = False,
        legacy: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self.targets = targets
        self.isolated = isolated
        self.legacy = legacy
        self.entries: list[tuple[MCPDocument, str]] = []
        self.changed = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="mcp-manager"):
            yield Static("Manage MCP servers")
            yield Static("Select a file entry to edit it. Priority: user → project → private → explicit.")
            if self.isolated:
                yield Static("Isolated mode: only explicit configuration files are loaded.")
            yield OptionList(id="mcp-file-list")
            yield Static("", markup=False, id="mcp-manager-message")
            with Horizontal(id="mcp-manager-actions"):
                yield Button("Add", id="mcp-add", variant="primary")
                yield Button("Edit", id="mcp-edit")
                yield Button("Authorize", id="mcp-authorize")
                yield Button("Close", id="mcp-manager-close")
            if self.legacy:
                yield Button("Saved database connections", id="mcp-legacy")

    def on_mount(self) -> None:
        self.refresh_entries()

    def refresh_entries(self) -> None:
        self.entries = []
        listing = self.query_one("#mcp-file-list", OptionList)
        listing.clear_options()
        errors = []
        for target in self.targets:
            try:
                document = MCPDocument.read(target.path, target.layer)
            except (ValueError, OSError):
                errors.append(f"Cannot edit {target.path}. Check its format and permissions.")
                continue
            for name, entry in document.servers.items():
                self.entries.append((document, name))
                enabled = "enabled" if entry.get("enabled", True) else "disabled"
                listing.add_option(Text(f"{name} · {enabled} · {target.label}\n{target.path}"))
        self.query_one("#mcp-manager-message", Static).update(
            "\n".join(errors) or "Changes apply when you close this manager."
        )
        listing.highlighted = 0 if self.entries else None
        self.update_actions()

    def selected(self) -> tuple[MCPDocument, str] | None:
        index = self.query_one("#mcp-file-list", OptionList).highlighted
        return self.entries[index] if index is not None and index < len(self.entries) else None

    def update_actions(self) -> None:
        selected = self.selected()
        self.query_one("#mcp-edit", Button).disabled = selected is None
        oauth = selected is not None and selected[0].servers[selected[1]].get("oauth") is not None
        self.query_one("#mcp-authorize", Button).disabled = not oauth

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        self.update_actions()

    def edited(self, changed: bool | None) -> None:
        self.changed |= bool(changed)
        self.refresh_entries()

    def edit(self) -> None:
        if selected := self.selected():
            self.app.push_screen(MCPFileEditScreen(self.targets, document=selected[0], name=selected[1]), self.edited)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.edit()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        match event.button.id:
            case "mcp-add":
                self.app.push_screen(MCPFileEditScreen(self.targets), self.edited)
            case "mcp-edit":
                self.edit()
            case "mcp-authorize":
                if selected := self.selected():
                    config = MCPServerConfig.from_dict(selected[1], selected[0].servers[selected[1]])
                    self.app.push_screen(MCPAuthorizationScreen(config), self.edited)
            case "mcp-legacy" if self.legacy:
                await self.legacy()
            case "mcp-manager-close":
                self.action_close()

    def action_close(self) -> None:
        self.dismiss(self.changed)
