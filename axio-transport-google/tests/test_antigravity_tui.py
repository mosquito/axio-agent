from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from textual.app import App
from textual.widgets import Input, Select, Static

from axio_transport_google.antigravity import AntigravityAccount, OAuthConfig, Tokens, save_account
from axio_transport_google.tui.antigravity import AntigravitySettingsScreen


def account(account_id: str = "one") -> AntigravityAccount:
    return AntigravityAccount(
        account_id,
        account_id,
        OAuthConfig().client_id,
        "same@example.test",
        Tokens("access-token", "refresh-token", time.time() + 3600),
    )


class ScreenApp(App[None]):
    def __init__(self, screen: AntigravitySettingsScreen) -> None:
        super().__init__()
        self.settings_screen = screen
        self.result: dict[str, str] | None = None

    def on_mount(self) -> None:
        self.push_screen(self.settings_screen, self.completed)

    def completed(self, result: dict[str, str] | None) -> None:
        self.result = result


@pytest.fixture
def screen(tmp_path: Path) -> AntigravitySettingsScreen:
    return AntigravitySettingsScreen({"credentials": str(tmp_path / "auth" / "antigravity.auth")})


async def test_select_same_email_registrations_without_copying_tokens(screen: AntigravitySettingsScreen) -> None:
    await save_account(screen.store, account("one"))
    await save_account(screen.store, account("two"))
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        screen.query_one("#antigravity-account", Select).value = "two"
        await pilot.pause()
        await pilot.click("#antigravity-use")
    assert app.result == {"credentials": str(screen.store.path), "account_id": "two"}
    assert await screen.store.selected_account() == "two"


@pytest.mark.parametrize("returning", [False, True])
async def test_login_persists_account_and_permission(
    screen: AntigravitySettingsScreen,
    monkeypatch: pytest.MonkeyPatch,
    returning: bool,
) -> None:
    expected = account()
    if returning:
        await save_account(screen.store, expected)
    signin = AsyncMock(return_value=account())
    monkeypatch.setattr("axio_transport_google.tui.antigravity.AntigravityOAuth.sign_in", signin)
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        await screen.sign_in("one" if returning else None)
        await pilot.pause()
        state = str(screen.query_one("#antigravity-status", Static).render())
        assert "Signed in" in state
        await screen.action_cancel()
    async with screen.store.lock("one"):
        saved = await screen.store.load("one")
    assert saved is not None and saved.tokens is not None
    assert app.result is not None and app.result["account_id"] == "one"
    assert "access-token" not in str(app.result)


async def test_close_cancels_pending_signin(
    screen: AntigravitySettingsScreen, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, stopped = asyncio.Event(), asyncio.Event()

    async def signin(*args: Any, **kwargs: Any) -> AntigravityAccount:
        started.set()
        try:
            await asyncio.Future[None]()
        finally:
            stopped.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr("axio_transport_google.tui.antigravity.AntigravityOAuth.sign_in", signin)
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        await pilot.click("#antigravity-login")
        await asyncio.wait_for(started.wait(), 2)
        await pilot.press("escape")
        await asyncio.wait_for(stopped.wait(), 2)
    assert app.result is None


async def test_pasted_callback_uses_same_oauth_flow(
    screen: AntigravitySettingsScreen, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    async def signin(
        self: object,
        *,
        open_browser: Callable[[str], Awaitable[None]],
        read_callback: Callable[[], Awaitable[str]],
        manual: bool,
    ) -> AntigravityAccount:
        await open_browser("https://auth.example.test/authorize")
        started.set()
        assert await read_callback() == "http://localhost/auth/callback?code=fixture"
        return account()

    monkeypatch.setattr("axio_transport_google.tui.antigravity.AntigravityOAuth.sign_in", signin)
    monkeypatch.setattr("axio_transport_google.tui.antigravity.launch_browser", AsyncMock(return_value=False))
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        await pilot.click("#antigravity-login")
        await asyncio.wait_for(started.wait(), 2)
        callback = screen.query_one("#antigravity-callback", Input)
        callback.value = "http://localhost/auth/callback?code=fixture"
        callback.focus()
        await pilot.press("enter")
        await screen.operation.wait()
        assert screen.changed
        assert screen.query_one("#antigravity-url", Input).value == ""


async def test_signout_clears_only_selected_account(screen: AntigravitySettingsScreen) -> None:
    await save_account(screen.store, account("one"))
    await save_account(screen.store, account("two"))
    app = ScreenApp(screen)
    async with app.run_test():
        await screen.sign_out("one")
        assert "Signed out locally" in str(screen.query_one("#antigravity-status", Static).render())
    async with screen.store.lock("one"):
        one = await screen.store.load("one")
    async with screen.store.lock("two"):
        two = await screen.store.load("two")
    assert one is not None and one.tokens is None
    assert two is not None and two.tokens is not None


async def test_wrong_identity_does_not_replace_selected_account(
    screen: AntigravitySettingsScreen, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axio.credentials import CredentialError

    await save_account(screen.store, account("one"))
    monkeypatch.setattr(
        "axio_transport_google.tui.antigravity.AntigravityOAuth.sign_in", AsyncMock(return_value=account("other"))
    )
    app = ScreenApp(screen)
    async with app.run_test():
        with pytest.raises(CredentialError, match="different Google account"):
            await screen.sign_in("one")
    assert [a.account_id for a in await screen.store.list_records()] == ["one"]
