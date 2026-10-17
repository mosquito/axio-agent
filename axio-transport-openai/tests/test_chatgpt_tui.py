from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from textual.app import App
from textual.widgets import Input, Select, Static

from axio_transport_openai.credentials import save_account
from axio_transport_openai.oauth import SCOPES, ChatGPTAccount, ChatGPTTokens
from axio_transport_openai.session import SignOutResult
from axio_transport_openai.tui.chatgpt import ChatGPTSettingsScreen


def account(client_id: str = "one", *, consent: bool = True) -> ChatGPTAccount:
    now = time.time()
    scopes = frozenset(SCOPES.split()) if consent else frozenset({"openid"})
    return ChatGPTAccount(
        client_id,
        "host",
        "https://auth.openai.com",
        "subject",
        "same@example.test",
        tokens=ChatGPTTokens(
            "id-token",
            "access-token",
            "refresh-token",
            "Bearer",
            scopes,
            now,
            now + 3600,
        ),
    )


class ScreenApp(App[None]):
    def __init__(self, screen: ChatGPTSettingsScreen) -> None:
        super().__init__()
        self.settings_screen = screen
        self.result: dict[str, str] | None = None

    def on_mount(self) -> None:
        self.push_screen(self.settings_screen, self.completed)

    def completed(self, result: dict[str, str] | None) -> None:
        self.result = result


@pytest.fixture
def screen(tmp_path: Path) -> ChatGPTSettingsScreen:
    return ChatGPTSettingsScreen({"credentials": str(tmp_path / "auth" / "chatgpt.auth")})


async def test_select_same_email_registrations_without_copying_tokens(screen: ChatGPTSettingsScreen) -> None:
    await save_account(screen.store, account("one"))
    await save_account(screen.store, account("two"))
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        screen.query_one("#chatgpt-account", Select).value = "two"
        await pilot.pause()
        await pilot.click("#chatgpt-use")
    assert app.result == {"credentials": str(screen.store.path), "client_id": "two"}
    assert await screen.store.selected_account() == "two"


@pytest.mark.parametrize("returning", [False, True])
@pytest.mark.parametrize("consent", [False, True])
async def test_login_persists_account_and_permission(
    screen: ChatGPTSettingsScreen,
    monkeypatch: pytest.MonkeyPatch,
    returning: bool,
    consent: bool,
) -> None:
    expected = account()
    if returning:
        await save_account(screen.store, expected)
    signin = AsyncMock(return_value=account(consent=consent))
    monkeypatch.setattr("axio_transport_openai.tui.chatgpt.ChatGPTOAuth.sign_in", signin)
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        await screen.sign_in("one" if returning else None)
        await pilot.pause()
        state = str(screen.query_one("#chatgpt-status", Static).render())
        assert ("ready" if consent else "consent_required") in state
        await screen.action_cancel()
    assert signin.call_args.args[0] == (expected if returning else None)
    async with screen.store.lock("one"):
        saved = await screen.store.load("one")
    assert saved is not None and saved.account.tokens is not None
    assert saved.account.tokens.plan_enabled == consent
    assert app.result is not None and app.result["client_id"] == "one"
    assert "access-token" not in str(app.result)


async def test_close_cancels_pending_signin(screen: ChatGPTSettingsScreen, monkeypatch: pytest.MonkeyPatch) -> None:
    started, stopped = asyncio.Event(), asyncio.Event()

    async def signin(*args: Any, **kwargs: Any) -> ChatGPTAccount:
        started.set()
        try:
            await asyncio.Future[None]()
        finally:
            stopped.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr("axio_transport_openai.tui.chatgpt.ChatGPTOAuth.sign_in", signin)
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        await pilot.click("#chatgpt-login")
        await asyncio.wait_for(started.wait(), 2)
        await pilot.press("escape")
        await asyncio.wait_for(stopped.wait(), 2)
    assert app.result is None


async def test_pasted_callback_uses_same_oauth_flow(
    screen: ChatGPTSettingsScreen, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    async def signin(
        self: object,
        previous: ChatGPTAccount | None,
        *,
        open_browser: Callable[[str], Awaitable[None]],
        read_callback: Callable[[], Awaitable[str]],
    ) -> ChatGPTAccount:
        await open_browser("https://auth.example.test/authorize")
        started.set()
        assert await read_callback() == "http://localhost/auth/callback?code=fixture"
        return account()

    monkeypatch.setattr("axio_transport_openai.tui.chatgpt.ChatGPTOAuth.sign_in", signin)
    monkeypatch.setattr("axio_transport_openai.tui.chatgpt.launch_browser", AsyncMock(return_value=False))
    app = ScreenApp(screen)
    async with app.run_test() as pilot:
        await pilot.click("#chatgpt-login")
        await asyncio.wait_for(started.wait(), 2)
        callback = screen.query_one("#chatgpt-callback", Input)
        callback.value = "http://localhost/auth/callback?code=fixture"
        callback.focus()
        await pilot.press("enter")
        await screen.operation.wait()
        assert screen.changed
        assert screen.query_one("#chatgpt-url", Input).value == ""


async def test_logout_reports_unconfirmed_revocation(
    screen: ChatGPTSettingsScreen, monkeypatch: pytest.MonkeyPatch
) -> None:
    await save_account(screen.store, account())
    monkeypatch.setattr(
        "axio_transport_openai.tui.chatgpt.ChatGPTSession.sign_out", AsyncMock(return_value=SignOutResult(False))
    )
    app = ScreenApp(screen)
    async with app.run_test():
        await screen.sign_out("one")
        assert "could not be confirmed" in str(screen.query_one("#chatgpt-status", Static).render())


@pytest.mark.parametrize(
    "code,status,recovery",
    [
        ("subscription_sharing_usage_limit_exceeded", 429, "quota"),
        ("subscription_sharing_user_not_eligible", 403, "permission"),
    ],
)
def test_provider_error_text_preserves_recovery_without_raw_body(code: str, status: int, recovery: str) -> None:
    from axio_responses import ResponsesAPIError

    from axio_transport_openai.chatgpt import ChatGPTTransport

    result = ChatGPTTransport.format_error(ResponsesAPIError("secret response", code=code, status=status))
    assert recovery in result and str(status) in result
    assert "secret response" not in result
    if recovery == "quota":
        assert "https://chatgpt.com/settings/usage" in result


async def test_signout_clears_only_selected_account(
    screen: ChatGPTSettingsScreen, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axio_transport_openai.cli import account_status

    host = await screen.store.host_id()
    await save_account(screen.store, replace(account("one"), host_id=host))
    await save_account(screen.store, replace(account("two"), host_id=host))
    revoke = AsyncMock()
    monkeypatch.setattr("axio_transport_openai.tui.chatgpt.ChatGPTOAuth.revoke", revoke)
    app = ScreenApp(screen)
    async with app.run_test():
        await screen.sign_out("one")
    assert (await account_status(screen.store, "one"))["state"] == "signed_out"
    assert (await account_status(screen.store, "two"))["state"] == "ready"
    assert revoke.call_args.args[0].client_id == "one"
