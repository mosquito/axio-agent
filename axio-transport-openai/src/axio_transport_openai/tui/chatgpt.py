"""ChatGPT account controls backed by the shared credential store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Link, Select, Static

from ..accounts import list_accounts
from ..cli import account_status, error_message
from ..credential_stores import FileCredentialStore
from ..credentials import CredentialError, save_account
from ..oauth import ChatGPTOAuth, launch_browser
from ..session import ChatGPTSession

USAGE_URL = "https://chatgpt.com/settings/usage"


class ChatGPTSettingsScreen(ModalScreen[dict[str, str] | None]):
    """Select registrations, authorize plan access, or sign out without copying tokens into UI config."""

    BINDINGS = [Binding("escape", "cancel", "Close")]
    CSS = """
    ChatGPTSettingsScreen { align: center middle; }
    #chatgpt-settings { width: 86; height: auto; max-height: 95%; border: heavy $accent;
        background: $panel; padding: 1 2; }
    #chatgpt-settings Horizontal { height: auto; }
    #chatgpt-settings Button { margin: 1 1 0 0; }
    #chatgpt-url { height: 3; }
    """

    def __init__(self, settings: dict[str, str]) -> None:
        super().__init__()
        self.settings = {k: v for k, v in settings.items() if k in {"credentials", "client_id"}}
        path = settings.get("credentials")
        self.store = FileCredentialStore(Path(path) if path else None)
        self.operation: Any = None
        self.callbacks: asyncio.Queue[str] = asyncio.Queue()
        self.changed = False
        self.initial_error = settings.get("_error")

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chatgpt-settings"):
            yield Static("ChatGPT subscription", markup=False)
            yield Select[str]([], prompt="Select an account registration", id="chatgpt-account")
            yield Static("Loading accounts…", id="chatgpt-status", markup=False)
            yield Link("ChatGPT Usage", url=USAGE_URL)
            with Horizontal():
                yield Button("Continue with ChatGPT", id="chatgpt-login", variant="primary")
                yield Button("Add account", id="chatgpt-add")
                yield Button("Sign out", id="chatgpt-logout", variant="warning")
            yield Input(placeholder="Authorization URL appears here", id="chatgpt-url", select_on_focus=True)
            yield Input(
                placeholder="Paste full redirect URL here if the browser cannot reach localhost", id="chatgpt-callback"
            )
            with Horizontal():
                yield Button("Use account", id="chatgpt-use", variant="success")
                yield Button("Close", id="chatgpt-close")

    async def on_mount(self) -> None:
        await self.refresh_accounts()
        if self.initial_error:
            self.status(self.initial_error)

    def status(self, text: str) -> None:
        self.query_one("#chatgpt-status", Static).update(text)

    def selected(self) -> str | None:
        value = self.query_one("#chatgpt-account", Select).value
        return value if isinstance(value, str) else None

    async def refresh_accounts(self, selected: str | None = None) -> None:
        try:
            accounts = await list_accounts(self.store)
            widget = self.query_one("#chatgpt-account", Select)
            widget.set_options([(f"{a.email or a.name or 'Account'} · {a.client_id}", a.client_id) for a in accounts])
            current = selected or self.settings.get("client_id") or await self.store.selected_account()
            if current in {a.client_id for a in accounts}:
                widget.value = current
            elif len(accounts) == 1:
                widget.value = accounts[0].client_id
            await self.show_status()
        except Exception as error:
            self.status(error_message(error))

    async def show_status(self) -> None:
        selected = self.selected()
        if selected is None:
            self.status("Continue with ChatGPT to authorize a new account.")
            return
        try:
            row = await account_status(self.store, selected)
            self.status(
                f"State: {row['state']}. Plan permission: {'granted' if row['plan_enabled'] else 'not granted'}."
            )
        except Exception as error:
            self.status(error_message(error))

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "chatgpt-account":
            await self.show_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chatgpt-callback" and event.value.strip():
            self.callbacks.put_nowait(event.value.strip())
            event.input.value = ""
        event.stop()

    async def open_authorization(self, url: str) -> None:
        self.query_one("#chatgpt-url", Input).value = url
        self.status(
            "Approve plan access in your browser. You can also copy the URL and paste the full redirect below."
        )
        await launch_browser(url)

    async def sign_in(self, client_id: str | None) -> None:
        expected = None
        if client_id is not None:
            async with self.store.lock(client_id):
                expected = await self.store.load(client_id)
            if expected is None:
                raise CredentialError("The selected registration was not found.")
        async with aiohttp.ClientSession() as http:
            oauth = ChatGPTOAuth(http, await self.store.host_id())
            account = await oauth.sign_in(
                expected.public_account() if expected else None,
                open_browser=self.open_authorization,
                read_callback=self.callbacks.get,
            )
        await save_account(self.store, account, expected=expected)
        self.settings["client_id"] = account.client_id
        self.changed = True
        await self.refresh_accounts(account.client_id)

    async def sign_out(self, client_id: str) -> None:
        async with aiohttp.ClientSession() as http:
            account = ChatGPTSession(self.store, ChatGPTOAuth(http, await self.store.host_id()), client_id)
            try:
                result = await account.sign_out()
            finally:
                await account.close()
        self.settings["client_id"] = client_id
        self.changed = True
        await self.refresh_accounts(client_id)
        if not result.revocation_confirmed:
            self.status("Signed out locally. Remote revocation could not be confirmed.")

    async def perform(self, action: str, selected: str | None) -> None:
        try:
            for widget in self.query(Button):
                if widget.id != "chatgpt-close":
                    widget.disabled = True
            if action == "chatgpt-logout" and selected is not None:
                await self.sign_out(selected)
            else:
                await self.sign_in(None if action == "chatgpt-add" else selected)
        except Exception as error:
            self.status(error_message(error))
        finally:
            self.query_one("#chatgpt-url", Input).value = ""
            self.query_one("#chatgpt-callback", Input).value = ""
            for widget in self.query(Button):
                widget.disabled = False

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        if action == "chatgpt-close":
            await self.action_cancel()
        elif action == "chatgpt-use":
            selected = self.selected()
            if selected is not None:
                await self.store.select_account(selected)
                self.settings["client_id"] = selected
                self.dismiss(self.settings)
        elif action in {"chatgpt-login", "chatgpt-add", "chatgpt-logout"}:
            self.callbacks = asyncio.Queue()
            self.operation = self.run_worker(self.perform(action, self.selected()), exclusive=True)

    async def action_cancel(self) -> None:
        if self.operation is not None and self.operation.is_running:
            self.operation.cancel()
            try:
                await self.operation.wait()
            except Exception:
                pass
        self.dismiss(self.settings if self.changed else None)
