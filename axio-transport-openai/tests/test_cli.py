"""Account commands preserve registrations and keep credentials out of output."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
from axio_responses import ResponsesAPIError

from axio_transport_openai import cli
from axio_transport_openai.accounts import resolve_account
from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.credentials import CredentialError, CredentialRecord, save_account
from axio_transport_openai.oauth import PLAN_SCOPE, SCOPES, ChatGPTAccount, ChatGPTTokens, OAuthDenied
from axio_transport_openai.session import ChatGPTSession, SignOutResult


@pytest.fixture(autouse=True)
def isolated_default_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "default-config"))


@pytest.fixture
def store(tmp_path: Path) -> FileCredentialStore:
    return FileCredentialStore(tmp_path / "auth" / "chatgpt.auth")


@pytest.fixture
async def account(store: FileCredentialStore) -> ChatGPTAccount:
    now = time.time()
    account = ChatGPTAccount(
        "client-a",
        await store.host_id(),
        "https://auth.openai.com",
        "subject-a",
        "owl@example.test",
        tokens=ChatGPTTokens(
            "id-secret", "access-secret", "refresh-secret", "Bearer", frozenset(SCOPES.split()), now, now + 3600
        ),
    )
    await save_account(store, account)
    return account


def arguments(store: FileCredentialStore, *args: str) -> argparse.Namespace:
    return cli.parser().parse_args(["--credentials", str(store.path), *args])


async def test_accounts_and_status_are_safe_offline(
    store: FileCredentialStore, account: ChatGPTAccount, capsys: pytest.CaptureFixture[str]
) -> None:
    other = replace(account, client_id="client-b")
    await save_account(store, other)
    await cli.run(arguments(store, "select", "client-b"))
    capsys.readouterr()
    await cli.run(arguments(store, "accounts", "--json"))
    output = capsys.readouterr().out
    rows = json.loads(output)
    assert len(rows) == 2 and rows[1]["selected"] and not rows[0]["selected"]
    assert "secret" not in output
    await cli.run(arguments(store, "status", "--account", account.client_id, "--json"))
    assert json.loads(capsys.readouterr().out)["state"] == "ready"
    assert await store.selected_account() == "client-b"


async def test_selection_requires_explicit_choice_with_multiple_accounts(
    store: FileCredentialStore, account: ChatGPTAccount
) -> None:
    assert await resolve_account(store) == account.client_id
    await save_account(store, replace(account, client_id="client-b"))
    with pytest.raises(CredentialError, match="Select an account"):
        await resolve_account(store)
    await store.select_account("client-b")
    assert await resolve_account(FileCredentialStore(store.path)) == "client-b"
    assert await resolve_account(store, account.client_id) == account.client_id


@pytest.mark.parametrize("state", ["signed_out", "consent_required", "refresh_required", "reauthentication_required"])
async def test_status_distinguishes_local_states(
    store: FileCredentialStore, account: ChatGPTAccount, state: str
) -> None:
    assert account.tokens is not None
    tokens: ChatGPTTokens | None = account.tokens
    operation: Literal["refresh", "logout"] | None = None
    if state == "signed_out":
        tokens = None
    elif state == "consent_required":
        tokens = replace(account.tokens, scopes=account.tokens.scopes - {PLAN_SCOPE})
    elif state == "refresh_required":
        tokens = replace(account.tokens, received_at=time.time() - 3600, expires_at=time.time() - 1)
    else:
        operation = "refresh"
    async with store.lock(account.client_id):
        await store.save(CredentialRecord(replace(account, tokens=tokens), operation=operation))
    assert (await cli.account_status(store, account.client_id))["state"] == state


async def test_login_reuses_auth_and_selects_only_after_success(
    store: FileCredentialStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def sign_in(**kwargs: Any) -> ChatGPTAccount:
        assert kwargs["credentials"] == store.path and kwargs["manual"]
        assert kwargs["client_id"] is None
        result = ChatGPTAccount("new-client", await store.host_id(), "https://auth.openai.com", "new-subject")
        await save_account(store, result)
        return result

    monkeypatch.setattr(cli, "sign_in", sign_in)
    assert await cli.run(arguments(store, "login", "--manual")) == 0
    assert await store.selected_account() == "new-client"
    output = capsys.readouterr()
    assert "Signed in" in output.out and "permission was not granted" in output.err


@pytest.mark.parametrize("error", [OAuthDenied("Authorization was declined."), asyncio.CancelledError()])
async def test_cancelled_login_preserves_selection(
    store: FileCredentialStore, account: ChatGPTAccount, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    await store.select_account(account.client_id)

    async def sign_in(**kwargs: Any) -> ChatGPTAccount:
        raise error

    monkeypatch.setattr(cli, "sign_in", sign_in)
    with pytest.raises(type(error)):
        await cli.run(arguments(store, "login"))
    assert await store.selected_account() == account.client_id


async def test_logout_uses_shared_session_and_keeps_registration(
    store: FileCredentialStore,
    account: ChatGPTAccount,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await store.select_account(account.client_id)

    async def logout(self: ChatGPTSession) -> SignOutResult:
        assert self.client_id == account.client_id
        async with self.store.lock(self.client_id):
            await self.store.save(CredentialRecord(replace(account, tokens=None)))
        return SignOutResult(False)

    monkeypatch.setattr(ChatGPTSession, "sign_out", logout)
    assert await cli.run(arguments(store, "logout")) == 0
    assert await store.selected_account() == account.client_id
    assert (await cli.account_status(store, account.client_id))["state"] == "signed_out"
    output = capsys.readouterr()
    assert "not confirmed" in output.err and "secret" not in output.out + output.err


def test_errors_are_safe_and_exit_nonzero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def run(args: argparse.Namespace) -> int:
        raise ResponsesAPIError("access-secret", status=403, body={"message": "id-secret"})

    monkeypatch.setattr(cli, "run", run)
    with pytest.raises(SystemExit) as raised:
        cli.main(["models"])
    assert raised.value.code == 1
    assert "secret" not in capsys.readouterr().err


def test_cancelled_command_exit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def run(args: argparse.Namespace) -> int:
        raise asyncio.CancelledError

    monkeypatch.setattr(cli, "run", run)
    with pytest.raises(SystemExit) as raised:
        cli.main(["login"])
    assert raised.value.code == 130
    assert "cancelled" in capsys.readouterr().err


def test_credentials_flag_works_before_and_after_command(tmp_path: Path) -> None:
    path = str(tmp_path / "chatgpt.auth")
    for argv in (["--credentials", path, "status"], ["status", "--credentials", path]):
        assert cli.parser().parse_args(argv).credentials == Path(path)


@pytest.mark.parametrize("argv", [["login", "--timeout", "nan"], ["login", "--port", "65536"]])
def test_invalid_login_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert raised.value.code == 2
