"""Headless authorization and terminal persistence without real credentials."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import pytest

from axio_transport_openai import auth
from axio_transport_openai.auth import CallbackInput, TerminalAuthorization
from axio_transport_openai.credential_stores import FileCredentialStore, default_credentials_path
from axio_transport_openai.credentials import CredentialError
from axio_transport_openai.oauth import SCOPES, ChatGPTAccount, ChatGPTOAuth, ChatGPTTokens, OAuthError, open_browser


@pytest.fixture(autouse=True)
def local_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)


@pytest.mark.parametrize("xdg", ["absolute", "relative", "empty", "missing"])
def test_default_credentials_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg: str) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    if xdg == "absolute":
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        expected = tmp_path / "config"
    else:
        expected = tmp_path / ".config"
        if xdg == "missing":
            monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        else:
            monkeypatch.setenv("XDG_CONFIG_HOME", "relative" if xdg == "relative" else "")
    assert default_credentials_path() == expected / "axio" / "chatgpt.auth"
    assert FileCredentialStore().path == default_credentials_path()


@pytest.mark.parametrize("failure", ["absent", "error", "oserror", "ssh"])
async def test_browser_fallback_prints_url_without_id_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    calls = []

    def browser(url: str) -> bool:
        calls.append(url)
        if failure == "error":
            raise webbrowser.Error("no browser")
        if failure == "oserror":
            raise OSError("no display")
        return False

    monkeypatch.setattr(webbrowser, "open", browser)
    if failure == "ssh":
        monkeypatch.setenv("SSH_CONNECTION", "test")
    await open_browser("https://auth.example/authorize?state=state&id_token_hint=id-secret")
    captured = capsys.readouterr()
    assert captured.out.strip() == "https://auth.example/authorize?state=state"
    assert "id-secret" not in captured.out + captured.err
    assert len(calls) == (0 if failure == "ssh" else 1)


async def test_available_browser_does_not_print_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(webbrowser, "open", lambda url: True)
    await open_browser("https://auth.example/authorize")
    assert not capsys.readouterr().out


async def test_manual_mode_does_not_launch_browser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected(url: str) -> bool:
        raise AssertionError("Manual mode must not open a browser")

    monkeypatch.setattr(webbrowser, "open", unexpected)
    terminal = TerminalAuthorization(manual=True)
    await terminal.open("https://auth.example/authorize")
    captured = capsys.readouterr()
    assert "https://auth.example/authorize" in captured.out
    assert "redirect URL" in captured.err


async def test_stdin_reads_chunks_and_can_be_cancelled() -> None:
    read_fd, write_fd = os.pipe()
    try:
        reader = CallbackInput(read_fd)
        task = asyncio.create_task(reader.readline())
        os.write(write_fd, b"http://127.0.0.1/callback?")
        await asyncio.sleep(0.01)
        assert not task.done()
        os.write(write_fd, b"code=code&state=state\nsecond\n")
        assert await task == "http://127.0.0.1/callback?code=code&state=state"
        assert await reader.readline() == "second"
        task = asyncio.create_task(reader.readline())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Cancellation removes the event-loop reader; a later read still works.
        os.write(write_fd, b"third\n")
        assert await reader.readline() == "third"
    finally:
        os.close(read_fd)
        os.close(write_fd)


async def test_stdin_eof() -> None:
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        with pytest.raises(OAuthError, match="No callback"):
            await CallbackInput(read_fd).readline()
    finally:
        os.close(read_fd)


async def test_stdin_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_text("http://127.0.0.1/callback?code=test\n")
    with path.open("rb") as stream:
        assert await CallbackInput(stream.fileno()).readline() == "http://127.0.0.1/callback?code=test"


async def test_sign_in_persists_and_reuses_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    calls: list[ChatGPTAccount | None] = []

    async def authorize(self: ChatGPTOAuth, account: ChatGPTAccount | None = None, **kwargs: Any) -> ChatGPTAccount:
        calls.append(account)
        now = time.time()
        return ChatGPTAccount(
            "issued-client",
            self.host_id,
            self.issuer,
            "subject",
            tokens=ChatGPTTokens(
                "id-secret",
                "access-secret",
                "refresh-secret",
                "Bearer",
                frozenset(SCOPES.split()),
                now,
                now + 3600,
            ),
        )

    monkeypatch.setattr(ChatGPTOAuth, "sign_in", authorize)
    first = await auth.sign_in(manual=True)
    store = FileCredentialStore()
    async with store.lock(first.client_id):
        saved = await store.load(first.client_id)
    assert saved is not None and saved.account == first
    assert await store.host_id() == first.host_id
    second = await auth.sign_in(client_id=first.client_id)
    assert calls == [None, first]
    assert second.host_id == first.host_id


async def test_missing_returning_registration(tmp_path: Path) -> None:
    with pytest.raises(CredentialError, match="not found"):
        await auth.sign_in(credentials=tmp_path / "chatgpt.auth", client_id="missing")


@pytest.mark.parametrize("error", [OAuthError("Sign-in failed."), TimeoutError(), KeyboardInterrupt()])
def test_cli_errors_are_concise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
) -> None:
    async def failed(**kwargs: Any) -> ChatGPTAccount:
        raise error

    monkeypatch.setattr(auth, "sign_in", failed)
    monkeypatch.setattr(sys, "argv", ["auth", "--manual"])
    with pytest.raises(SystemExit) as raised:
        auth.main()
    assert raised.value.code == (130 if isinstance(error, KeyboardInterrupt) else 1)
    assert "Traceback" not in capsys.readouterr().err
