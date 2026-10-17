"""The REPL owns one account session across prompts and model changes."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from axio import Agent, MemoryContextStore
from axio.models import ModelSpec
from axio_tools_local.paths import LocalPaths as Workspace
from axio_tools_local.provider import local_tools
from axio_transport_openai import ChatGPTTransport, accounts, app_auth
from axio_transport_openai.accounts import list_accounts
from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.credentials import CredentialError, save_account
from axio_transport_openai.oauth import ChatGPTAccount, ChatGPTOAuth, OAuthDenied
from axio_transport_openai.session import ChatGPTSession
from axio_transport_openai.subscription import ChatGPTResponses

import axio_repl as repl


@pytest.mark.asyncio
async def test_reasoning_reset_and_model_switch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = FileCredentialStore(tmp_path / "credentials")
    async with aiohttp.ClientSession() as http:
        account = ChatGPTSession(store, ChatGPTOAuth(http, "host"), "chosen")
        transport = ChatGPTTransport(account=account, session=http)
        body = {
            "models": [
                {
                    "slug": "reasoning",
                    "display_name": "Reasoning",
                    "visibility": "list",
                    "supported_reasoning_levels": [{"effort": "high"}],
                },
                {
                    "slug": "low-only",
                    "display_name": "Low only",
                    "visibility": "list",
                    "supported_reasoning_levels": [{"effort": "low"}],
                },
                {"slug": "plain", "display_name": "Plain", "visibility": "list"},
            ]
        }
        with patch.object(ChatGPTResponses, "fetch_models", AsyncMock(return_value=body)):
            await transport.fetch_models()
        repl._apply_thinking(transport, "high")
        assert transport.build_payload([], [], "")["reasoning"]["effort"] == "high"
        repl._apply_thinking(transport, "xhigh")
        assert transport.reasoning_effort == "high"
        assert "Valid levels: default, high" in capsys.readouterr().err
        repl._apply_thinking(transport, "default")
        assert transport.reasoning_effort is None
        assert "effort" not in transport.build_payload([], [], "")["reasoning"]
        repl._apply_thinking(transport, "high")
        workspace = Workspace(tmp_path, (tmp_path,))
        state = repl.REPLState(workspace.cwd, transport)
        state.change_model("low-only")
        low = state.transport
        assert low.reasoning_effort is None
        assert "effort" not in low.build_payload([], [], "")["reasoning"]
        repl._apply_thinking(low, "low")
        state.change_model("low-only")
        low = state.transport
        assert low.build_payload([], [], "")["reasoning"]["effort"] == "low"
        state.change_model("plain")
        plain = state.transport
        assert isinstance(plain, ChatGPTTransport) and plain.account is account
        assert plain.reasoning_effort is None and state.agent.transport is plain
        assert "reasoning" not in plain.build_payload([], [], "")
        for setting in ("high", "none"):
            repl._apply_thinking(plain, setting)
            assert plain.reasoning_effort is None
        plain.reasoning_effort = "high"
        repl._apply_thinking(plain, "default")
        assert "reasoning" not in plain.build_payload([], [], "")
        assert "reset to default" in capsys.readouterr().out
        await account.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "levels,argument,expected,invalid",
    [
        (["low", "medium"], None, "medium", False),
        (["low"], None, None, False),
        ([], None, None, False),
        (["low"], "low", "low", False),
        (["low", "medium"], "default", None, False),
        (["low"], "xhigh", None, True),
        (["future-effort"], "future-effort", "future-effort", False),
    ],
)
async def test_startup_effort_respects_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    levels: list[str],
    argument: str | None,
    expected: str | None,
    invalid: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axio-repl",
            "--isolated",
            "--transport",
            "chatgpt",
            "--account",
            "chosen",
            "--credentials",
            str(tmp_path / "credentials"),
            *([] if argument is None else ["--thinking", argument]),
            "hello",
        ],
    )
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    opened: list[ChatGPTTransport] = []

    @asynccontextmanager
    async def open_account(
        store: FileCredentialStore,
        http: aiohttp.ClientSession,
        *,
        client_id: str | None,
    ) -> AsyncIterator[ChatGPTTransport]:
        account = ChatGPTSession(store, ChatGPTOAuth(http, "host"), "chosen")
        transport = ChatGPTTransport(account=account, session=http)
        body = {
            "models": [
                {
                    "slug": "test",
                    "display_name": "Test",
                    "visibility": "list",
                    "supported_reasoning_levels": [{"effort": value} for value in levels],
                }
            ]
        }
        with patch.object(ChatGPTResponses, "fetch_models", AsyncMock(return_value=body)):
            await transport.fetch_models()
        opened.append(transport)
        try:
            yield transport
        finally:
            await account.close()

    monkeypatch.setattr(accounts, "open_chatgpt", open_account)
    prompt = AsyncMock()
    monkeypatch.setattr(repl, "run_prompt", prompt)
    if invalid:
        with pytest.raises(SystemExit) as error:
            await repl.main()
        assert error.value.code == 2
        assert "Valid levels: default, low" in capsys.readouterr().err
        prompt.assert_not_awaited()
    else:
        await repl.main()
        prompt.assert_awaited_once()
        payload = opened[0].build_payload([], [], "")
        assert payload.get("reasoning", {}).get("effort") == expected
    assert opened[0].session.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["quit", "cancel", "error", "unknown_model", "invalid_startup_model"])
@pytest.mark.parametrize("first_login", [False, True])
async def test_account_survives_model_change_and_closes_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], ending: str, first_login: bool
) -> None:
    path = tmp_path / "credentials"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axio-repl",
            "--isolated",
            "--transport",
            "chatgpt",
            *([] if first_login else ["--account", "chosen"]),
            "--credentials",
            str(path),
            "--model",
            "missing-model" if ending == "invalid_startup_model" else "gpt-6-luna",
        ],
    )
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    opened: list[ChatGPTTransport] = []
    closed: list[bool] = []
    login = AsyncMock()
    if first_login:

        async def sign_in(*, credentials: Path) -> ChatGPTAccount:
            assert credentials == path
            result = ChatGPTAccount("chosen", "host", "https://auth.openai.com", "subject")
            await save_account(FileCredentialStore(credentials), result)
            return result

        login.side_effect = sign_in
    monkeypatch.setattr(app_auth, "sign_in", login)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(app_auth, "_confirm_chatgpt_login", AsyncMock(return_value=True))

    @asynccontextmanager
    async def open_account(
        store: FileCredentialStore, http: aiohttp.ClientSession, *, client_id: str | None
    ) -> AsyncIterator[ChatGPTTransport]:
        assert store.path == path
        if first_login:
            assert client_id is None and await store.selected_account() == "chosen"
        else:
            assert client_id == "chosen"
        account = ChatGPTSession(store, ChatGPTOAuth(http, "host"), "chosen")
        transport = ChatGPTTransport(account=account, session=http)
        body = {
            "models": [
                {"slug": name, "display_name": name, "visibility": "list"}
                | (
                    {
                        "input_modalities": ["text", "image"],
                        "supports_parallel_tool_calls": True,
                        "supported_reasoning_levels": [{"effort": "low"}],
                        "context_window": 500_000,
                    }
                    if name != "unverified-model"
                    else {}
                )
                for name in ("gpt-6-luna", "gpt-6.1-sol", "unverified-model")
            ]
        }
        with patch.object(ChatGPTResponses, "fetch_models", AsyncMock(return_value=body)):
            await transport.fetch_models()
        opened.append(transport)
        try:
            yield transport
        finally:
            await account.close()
            closed.append(True)

    model_id = "unverified-model" if ending == "unknown_model" else "gpt-6.1-sol"
    inputs = iter([f"/model {model_id}", "/thinking low", "/help", "hello", "/quit"])
    monkeypatch.setattr(accounts, "open_chatgpt", open_account)
    monkeypatch.setattr(repl, "_read_input", lambda: next(inputs))

    async def prompt(agent: Agent, context: MemoryContextStore, text: str) -> None:
        current = agent.transport
        assert isinstance(current, ChatGPTTransport)
        assert current is not opened[0] and current.account is opened[0].account
        assert current.model is not None and current.model.id == model_id
        assert current.reasoning_effort == (None if ending == "unknown_model" else "low")
        assert text == "hello"
        assert model_id in agent.system
        if ending == "cancel":
            raise asyncio.CancelledError
        if ending == "error":
            raise RuntimeError("failed prompt")

    monkeypatch.setattr(repl, "run_prompt", prompt)
    if ending == "invalid_startup_model":
        with pytest.raises(SystemExit) as raised:
            await repl.main()
        assert raised.value.code == 2
    elif ending == "error":
        with pytest.raises(RuntimeError, match="failed prompt"):
            await repl.main()
    else:
        await repl.main()
    assert closed == [True] and opened[0].session.closed
    assert login.await_count == int(first_login)
    captured = capsys.readouterr()
    assert "Tools are disabled" not in captured.err
    if ending == "invalid_startup_model":
        assert "Unknown model 'missing-model'" in captured.err
        assert "Available models:\n  gpt-6-luna\n  gpt-6.1-sol\n  unverified-model" in captured.err
        assert "REPL ready" not in captured.out
    else:
        assert "shell" in captured.out
    with pytest.raises(RuntimeError, match="closed"):
        await opened[0].account.access_token()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flags",
    [
        ["--account", "id"],
        ["--transport", "chatgpt", "--temperature", "0.5"],
        ["--transport", "chatgpt", "--max-tokens", "50"],
        ["--transport", "chatgpt", "--debug"],
    ],
)
async def test_invalid_account_or_subscription_options_fail_before_network(
    monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["axio-repl", *flags])
    with patch.object(aiohttp, "ClientSession") as http:
        with pytest.raises(SystemExit) as raised:
            await repl.main()
    assert raised.value.code == 2
    http.assert_not_called()


@pytest.mark.asyncio
async def test_declined_login_does_not_authorize_or_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileCredentialStore(tmp_path / "credentials")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with (
        patch.object(app_auth, "_confirm_chatgpt_login", AsyncMock(return_value=False)),
        patch.object(app_auth, "sign_in", AsyncMock()) as login,
    ):
        with pytest.raises(CredentialError, match="cancelled"):
            await app_auth._offer_chatgpt_login(store, None)
    login.assert_not_awaited()
    assert not store.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,expected",
    [(b"yes\n", True), (b" Y \n", True), (b"n\n", False), (b"\n", False), (b"maybe\n", False), (b"", False)],
)
async def test_confirmation_input_and_eof(monkeypatch: pytest.MonkeyPatch, answer: bytes, expected: bool) -> None:
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        os.write(write_fd, answer)
        os.close(write_fd)
        assert await app_auth._confirm_chatgpt_login() is expected


@pytest.mark.asyncio
async def test_confirmation_cancellation_removes_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd) as stream:
            monkeypatch.setattr(sys, "stdin", stream)
            loop = asyncio.get_running_loop()
            with patch.object(loop, "remove_reader", wraps=loop.remove_reader) as remove:
                task = asyncio.create_task(app_auth._confirm_chatgpt_login())
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                remove.assert_called_once_with(read_fd)
    finally:
        os.close(write_fd)


@pytest.mark.asyncio
async def test_noninteractive_empty_store_never_reads_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with (
        patch.object(app_auth, "_confirm_chatgpt_login", AsyncMock()) as prompt,
        patch.object(app_auth, "sign_in", AsyncMock()) as login,
    ):
        with pytest.raises(CredentialError, match="terminal"):
            await app_auth._offer_chatgpt_login(FileCredentialStore(tmp_path / "credentials"), None)
    prompt.assert_not_called()
    login.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [OAuthDenied("Declined."), asyncio.CancelledError()])
async def test_failed_login_keeps_empty_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    store = FileCredentialStore(tmp_path / "credentials")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with (
        patch.object(app_auth, "_confirm_chatgpt_login", AsyncMock(return_value=True)),
        patch.object(app_auth, "sign_in", AsyncMock(side_effect=error)),
    ):
        with pytest.raises(type(error)):
            await app_auth._offer_chatgpt_login(store, None)
    assert await store.selected_account() is None and not await list_accounts(store)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_existing_registrations_never_trigger_first_run_login(tmp_path: Path, count: int) -> None:
    store = FileCredentialStore(tmp_path / "credentials")
    for index in range(count):
        await save_account(store, ChatGPTAccount(f"client-{index}", "host", "https://auth.openai.com", "subject"))
    with (
        patch.object(app_auth, "_confirm_chatgpt_login", AsyncMock()) as prompt,
        patch.object(app_auth, "sign_in", AsyncMock()) as login,
    ):
        await app_auth._offer_chatgpt_login(store, None)
    prompt.assert_not_called()
    login.assert_not_awaited()
    assert await store.selected_account() is None


def test_unknown_tool_support_is_visible_to_user(capsys: pytest.CaptureFixture[str]) -> None:
    repl._warn_missing_tools(ModelSpec("unknown"), local_tools(Path.cwd()))
    assert "Tools are disabled for unknown" in capsys.readouterr().err
