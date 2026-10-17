"""Antigravity startup and model changes keep the selected OAuth session."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import aiohttp
import pytest
from axio import Agent, MemoryContextStore
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.transport import discover_transports
from axio_tools_local.paths import LocalPaths as Workspace
from axio_transport_google.antigravity import (
    AntigravityAccount,
    AntigravityFileCredentials,
    AntigravityOAuth,
    AntigravitySession,
    AntigravityTransport,
    OAuthConfig,
    Tokens,
    save_account,
)

import axio_repl as repl


@pytest.mark.asyncio
async def test_model_change_keeps_account_and_synchronizes_agent(tmp_path: Path) -> None:
    first = ModelSpec("gemini-first", frozenset({Capability.text, Capability.tool_use}))
    second = ModelSpec("gemini-second", first.capabilities)
    store = AntigravityFileCredentials(tmp_path / "auth")
    async with aiohttp.ClientSession() as http:
        account = AntigravitySession(store, AntigravityOAuth(http), "chosen")
        transport = AntigravityTransport(
            session=http, account=account, model=first, models=ModelRegistry([first, second]), thinking_level="HIGH"
        )
        workspace = Workspace(tmp_path, (tmp_path,))
        state = repl.REPLState(workspace.cwd, transport)
        state.change_model("gemini-second")
        assert state.transport is not transport
        assert state.transport.account is account and state.agent.transport is state.transport
        assert isinstance(state.transport, AntigravityTransport)
        assert state.transport.model == second and state.transport.thinking_level is None
        assert "gemini-second" in state.agent.system


@pytest.mark.asyncio
@pytest.mark.parametrize("first_login", [False, True])
async def test_startup_and_saved_account_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_login: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "auth"
    store = AntigravityFileCredentials(path)
    account = AntigravityAccount(
        "chosen", "subject", OAuthConfig().client_id, tokens=Tokens("secret", "refresh", time.time() + 3600)
    )
    login = AsyncMock()
    if not first_login:
        await save_account(store, account)
        await store.select_account(account.account_id)
    else:

        async def save(target: AntigravityFileCredentials) -> AntigravityAccount:
            await save_account(target, account)
            await target.select_account(account.account_id)
            return account

        login.side_effect = save
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("axio_transport_google.antigravity.cli.sign_in", login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axio-repl",
            "--transport",
            "antigravity",
            "--isolated",
            "--credentials",
            str(path),
            "--model",
            "gemini-second",
            "hello",
        ],
    )
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    fetched: list[AntigravityTransport] = []

    async def fetch(self: AntigravityTransport) -> ModelRegistry:
        fetched.append(self)
        assert self.account is not None and self.account.account_id == "chosen"
        self.models = ModelRegistry([ModelSpec("gemini-first"), ModelSpec("gemini-second")])
        self.model = self.models.first()
        return self.models

    monkeypatch.setattr(AntigravityTransport, "fetch_models", fetch)

    async def prompt(agent: Agent, context: MemoryContextStore, text: str) -> None:
        assert isinstance(agent.transport, AntigravityTransport)
        assert agent.transport.model.id == "gemini-second"
        assert agent.transport.account is fetched[0].account

    run_prompt = AsyncMock(side_effect=prompt)
    monkeypatch.setattr(repl, "run_prompt", run_prompt)
    await repl.main()
    run_prompt.assert_awaited_once()
    assert login.await_count == first_login
    assert fetched[0].session is not None and fetched[0].session.closed
    reopened = AntigravityFileCredentials(path)
    assert await reopened.selected_account() == "chosen"


def test_entry_point_is_installed() -> None:
    assert discover_transports().entries["antigravity"].cls is AntigravityTransport


def test_debug_does_not_mutate_oauth_transport(capsys: pytest.CaptureFixture[str]) -> None:
    transport = AntigravityTransport()
    repl._apply_debug(transport, "on")
    assert "not supported" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_account_commands_reset_history_and_keep_model_clones_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "auth"
    store = AntigravityFileCredentials(path)
    for name in ("first", "second"):
        await save_account(
            store,
            AntigravityAccount(
                name, name, OAuthConfig().client_id, tokens=Tokens("secret", "refresh", time.time() + 3600)
            ),
        )
    await store.select_account("first")
    monkeypatch.setattr(
        sys, "argv", ["axio-repl", "--transport", "antigravity", "--isolated", "--credentials", str(path)]
    )
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    commands = iter(["one", "/account second", "two", "/model gemini-other", "three", "/logout", "/quit"])
    monkeypatch.setattr(repl, "_read_input", lambda: next(commands))

    async def fetch(self: AntigravityTransport) -> ModelRegistry:
        self.models = ModelRegistry([ModelSpec("gemini-default"), ModelSpec("gemini-other")])
        self.model = self.models.first()
        return self.models

    monkeypatch.setattr(AntigravityTransport, "fetch_models", fetch)
    calls: list[tuple[str, str, int]] = []

    async def prompt(agent: Agent, context: MemoryContextStore, text: str) -> None:
        from axio import Message, TextBlock

        assert isinstance(agent.transport, AntigravityTransport) and agent.transport.account is not None
        calls.append((agent.transport.account.account_id, agent.transport.model.id, len(await context.get_history())))
        await context.append(Message("user", [TextBlock(text)]))
        # An external process can change the default without changing the active account.
        await store.select_account("first")

    monkeypatch.setattr(repl, "run_prompt", prompt)
    await repl.main()
    assert calls == [("first", "gemini-default", 0), ("second", "gemini-default", 0), ("second", "gemini-other", 1)]
    async with store.lock("first"):
        first = await store.load("first")
    async with store.lock("second"):
        second = await store.load("second")
    assert first is not None and first.tokens is not None
    assert second is not None and second.tokens is None
    assert "History cleared" in capsys.readouterr().out
