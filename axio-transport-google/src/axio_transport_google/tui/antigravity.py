"""Antigravity account controls backed by the shared credential store."""

from __future__ import annotations

import asyncio
import os
import time
import webbrowser
from pathlib import Path
from typing import Any

import aiohttp
from axio.credentials import CredentialError
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from ..antigravity.oauth import AntigravityOAuth
from ..antigravity.session import AntigravityFileCredentials, AntigravitySession, list_accounts, save_account


async def launch_browser(url: str) -> bool:
    if os.environ.get("SSH_CONNECTION"):
        return False
    return await asyncio.to_thread(webbrowser.open, url)


def error_message(error: Exception) -> str:
    return str(error) or type(error).__name__


class AntigravitySettingsScreen(ModalScreen[dict[str, str] | None]):
    """Select registrations, authorize Google access, or sign out without copying tokens into UI config."""

    BINDINGS = [Binding("escape", "cancel", "Close")]
    CSS = """
    AntigravitySettingsScreen { align: center middle; }
    #antigravity-settings { width: 86; height: auto; max-height: 95%; border: heavy $accent;
        background: $panel; padding: 1 2; }
    #antigravity-settings Horizontal { height: auto; }
    #antigravity-settings Button { margin: 1 1 0 0; }
    #antigravity-url { height: 3; }
    """

    def __init__(self, settings: dict[str, str]) -> None:
        super().__init__()
        self.settings = {k: v for k, v in settings.items() if k in {"credentials", "account_id"}}
        path = settings.get("credentials")
        self.store = AntigravityFileCredentials(Path(path) if path else None)
        self.operation: Any = None
        self.callbacks: asyncio.Queue[str] = asyncio.Queue()
        self.changed = False
        self.initial_error = settings.get("_error")

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="antigravity-settings"):
            yield Static("Antigravity subscription", markup=False)
            yield Select[str]([], prompt="Select an account registration", id="antigravity-account")
            yield Static("Loading accounts…", id="antigravity-status", markup=False)
            with Horizontal():
                yield Button("Continue with Antigravity", id="antigravity-login", variant="primary")
                yield Button("Add account", id="antigravity-add")
                yield Button("Sign out", id="antigravity-logout", variant="warning")
            yield Input(placeholder="Authorization URL appears here", id="antigravity-url", select_on_focus=True)
            yield Input(
                placeholder="Paste full redirect URL here if the browser cannot reach localhost",
                id="antigravity-callback",
            )
            with Horizontal():
                yield Button("Use account", id="antigravity-use", variant="success")
                yield Button("Close", id="antigravity-close")

    async def on_mount(self) -> None:
        await self.refresh_accounts()
        if self.initial_error:
            self.status(self.initial_error)

    def status(self, text: str) -> None:
        self.query_one("#antigravity-status", Static).update(text)

    def selected(self) -> str | None:
        value = self.query_one("#antigravity-account", Select).value
        return value if isinstance(value, str) else None

    async def refresh_accounts(self, selected: str | None = None) -> None:
        try:
            accounts = await list_accounts(self.store)
            widget = self.query_one("#antigravity-account", Select)
            widget.set_options([(f"{a.email or 'Google account'} · {a.account_id}", a.account_id) for a in accounts])
            current = selected or self.settings.get("account_id") or await self.store.selected_account()
            if current in {a.account_id for a in accounts}:
                widget.value = current
            elif len(accounts) == 1:
                widget.value = accounts[0].account_id
            await self.show_status()
        except Exception as error:
            self.status(error_message(error))

    async def show_status(self) -> None:
        selected = self.selected()
        if selected is None:
            self.status("Continue with Antigravity to authorize a new account.")
            return
        try:
            async with self.store.lock(selected):
                account = await self.store.load(selected)
            if account is None or account.tokens is None:
                self.status("Signed out. Continue with Antigravity to sign in.")
            elif account.tokens.expires_at <= time.time():
                self.status("Saved account. The token will refresh on the next request.")
            else:
                self.status("Signed in. Model discovery checks Antigravity access.")
        except Exception as error:
            self.status(error_message(error))

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "antigravity-account":
            await self.show_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "antigravity-callback" and event.value.strip():
            self.callbacks.put_nowait(event.value.strip())
            event.input.value = ""
        event.stop()

    async def open_authorization(self, url: str) -> None:
        self.query_one("#antigravity-url", Input).value = url
        self.status(
            "Approve Google access in your browser. You can also copy the URL and paste the full redirect below."
        )
        await launch_browser(url)

    async def sign_in(self, account_id: str | None) -> None:
        previous = {account.account_id: account for account in await self.store.list_records()}
        if account_id is not None and account_id not in previous:
            raise CredentialError("The selected account was not found.")
        async with aiohttp.ClientSession() as http:
            account = await AntigravityOAuth(http).sign_in(
                open_browser=self.open_authorization,
                read_callback=self.callbacks.get,
                manual=bool(os.environ.get("SSH_CONNECTION")),
            )
        if account_id is not None and account.account_id != account_id:
            raise CredentialError("Sign-in returned a different Google account. Use Add account to authorize it.")
        await save_account(self.store, account, expected=previous.get(account.account_id))
        self.settings["account_id"] = account.account_id
        self.changed = True
        await self.refresh_accounts(account.account_id)

    async def sign_out(self, account_id: str) -> None:
        async with aiohttp.ClientSession() as http:
            account = AntigravitySession(self.store, AntigravityOAuth(http), account_id)
            await account.logout()
        self.settings["account_id"] = account_id
        self.changed = True
        await self.refresh_accounts(account_id)
        self.status("Signed out locally. The Google grant has not been revoked.")

    async def perform(self, action: str, selected: str | None) -> None:
        try:
            for widget in self.query(Button):
                if widget.id != "antigravity-close":
                    widget.disabled = True
            if action == "antigravity-logout":
                if selected is not None:
                    await self.sign_out(selected)
                else:
                    self.status("Select an account to sign out.")
            else:
                await self.sign_in(None if action == "antigravity-add" else selected)
        except Exception as error:
            self.status(error_message(error))
        finally:
            self.query_one("#antigravity-url", Input).value = ""
            self.query_one("#antigravity-callback", Input).value = ""
            for widget in self.query(Button):
                widget.disabled = False

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        if action == "antigravity-close":
            await self.action_cancel()
        elif action == "antigravity-use":
            selected = self.selected()
            if selected is not None:
                await self.store.select_account(selected)
                self.settings["account_id"] = selected
                self.dismiss(self.settings)
        elif action in {"antigravity-login", "antigravity-add", "antigravity-logout"}:
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
