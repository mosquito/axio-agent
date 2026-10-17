"""The same persistence contract exercised against both local backends."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import sqlite3
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import aiohttp
import pytest
from aiohttp import web

from axio_transport_openai.accounts import list_accounts
from axio_transport_openai.credential_files import account_key, decode_record, encode_record
from axio_transport_openai.credential_stores import FileCredentialStore, LocalCredentialStore, SQLiteCredentialStore
from axio_transport_openai.credentials import CredentialError, CredentialRecord, save_account
from axio_transport_openai.oauth import SCOPES, ChatGPTAccount, ChatGPTOAuth, ChatGPTTokens
from axio_transport_openai.session import ChatGPTSession, SignInRequired

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Local credential stores require POSIX permissions and flock"
)
type Backend = Literal["file", "sqlite"]


def make_store(backend: Backend, root: Path) -> LocalCredentialStore:
    if backend == "file":
        return FileCredentialStore(root / "files" / "chatgpt.auth")
    return SQLiteCredentialStore(root / "sqlite" / "accounts.db")


@pytest.fixture(params=["file", "sqlite"])
def backend(request: pytest.FixtureRequest) -> Backend:
    value: Backend = request.param
    return value


@pytest.fixture
def store(backend: Backend, tmp_path: Path) -> LocalCredentialStore:
    return make_store(backend, tmp_path)


@pytest.fixture
def record() -> CredentialRecord:
    now = time.time()
    tokens = ChatGPTTokens(
        "id-secret", "access-secret", "refresh-secret", "Bearer", frozenset(SCOPES.split()), now - 100, now - 1
    )
    return CredentialRecord(
        ChatGPTAccount("client-a", "host-a", "https://auth.openai.com", "subject-a", "owl@example.test", tokens=tokens)
    )


async def read(store: LocalCredentialStore, client_id: str = "client-a") -> CredentialRecord | None:
    async with store.lock(client_id):
        return await store.load(client_id)


async def write(store: LocalCredentialStore, record: CredentialRecord) -> None:
    async with store.lock(record.account.client_id):
        await store.save(record)


async def test_round_trip_restart_and_marker(
    store: LocalCredentialStore,
    backend: Backend,
    tmp_path: Path,
    record: CredentialRecord,
) -> None:
    assert await read(store) is None
    marker = replace(record, operation="refresh")
    await write(store, marker)
    restarted = make_store(backend, tmp_path)
    assert await read(restarted) == marker
    assert marker.public_account().tokens is None
    replacement = CredentialRecord(replace(record.account, tokens=None))
    await write(restarted, replacement)
    assert await read(store) == replacement
    assert "secret" not in repr(store)


async def test_reads_and_updates_original_storage_format(store: LocalCredentialStore) -> None:
    # This fixture spells out the old format without calling the new codec.
    data = {
        "version": 1,
        "account": {
            "client_id": "client-a",
            "host_id": "host-a",
            "issuer": "https://auth.openai.com",
            "subject": "subject-a",
            "email": None,
            "name": None,
            "tokens": {
                "id_token": "legacy-id",
                "access_token": "legacy-access",
                "refresh_token": "legacy-refresh",
                "token_type": "Bearer",
                "scopes": SCOPES.split(),
                "received_at": 1000.0,
                "expires_at": 2000.0,
            },
        },
        "revision": "legacy-revision",
        "operation": "refresh",
    }
    values = {
        account_key("client-a"): data,
        "selected": {"version": 1, "client_id": "client-a"},
        "host": {"version": 1, "host_id": "legacy-host"},
    }
    store.directory.prepare()
    if isinstance(store, FileCredentialStore):
        store.directory.write(store.path.name, json.dumps({"version": 1, "values": values}))
    else:
        assert isinstance(store, SQLiteCredentialStore)
        with sqlite3.connect(store.path) as conn:
            conn.execute("CREATE TABLE axio_chatgpt_credentials (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.executemany(
                "INSERT INTO axio_chatgpt_credentials VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in values.items()],
            )
        store.path.chmod(0o600)

    assert await store.host_id() == "legacy-host"
    assert await store.selected_account() == "client-a"
    async with store.lock("client-a"):
        record = await store.load("client-a")
        assert record is not None and record.revision == "legacy-revision" and record.operation == "refresh"
        assert record.account.tokens is not None and record.account.tokens.refresh_token == "legacy-refresh"
        assert record.public_account().tokens is None
        await store.save(replace(record, revision="new-revision", operation=None))
    assert all(account.tokens is None for account in await list_accounts(store))
    if isinstance(store, FileCredentialStore):
        saved = json.loads(store.path.read_text())["values"][account_key("client-a")]
    else:
        assert isinstance(store, SQLiteCredentialStore)
        with sqlite3.connect(store.path) as conn:
            saved = json.loads(
                conn.execute(
                    "SELECT value FROM axio_chatgpt_credentials WHERE key=?", (account_key("client-a"),)
                ).fetchone()[0]
            )
            assert conn.execute("SELECT name FROM sqlite_schema WHERE name='axio_credentials'").fetchone() is None
    assert saved["revision"] == "new-revision" and saved["operation"] is None
    assert saved["account"]["tokens"]["refresh_token"] == "legacy-refresh"


async def test_host_id_is_stable_and_concurrent(store: LocalCredentialStore, backend: Backend, tmp_path: Path) -> None:
    other = make_store(backend, tmp_path)
    host_ids = await asyncio.gather(*(s.host_id() for s in (store, other, store, other)))
    assert len(set(host_ids)) == 1
    assert host_ids[0].startswith("urn:uuid:")
    assert await make_store(backend, tmp_path).host_id() == host_ids[0]


async def test_committed_marker_survives_exception(store: LocalCredentialStore, record: CredentialRecord) -> None:
    marker = replace(record, operation="refresh")
    with pytest.raises(RuntimeError):
        async with store.lock(record.account.client_id):
            await store.save(marker)
            raise RuntimeError("process failed after save")
    assert await read(store) == marker


async def test_account_discovery_and_selection_survive_restart(
    store: LocalCredentialStore, backend: Backend, tmp_path: Path, record: CredentialRecord
) -> None:
    assert await list_accounts(store) == []
    assert await store.selected_account() is None
    await store.host_id()
    await write(store, record)
    other = replace(record, account=replace(record.account, client_id="client-b"))
    await write(store, other)
    await store.select_account("client-b")
    restarted = make_store(backend, tmp_path)
    accounts = await list_accounts(restarted)
    assert [account.client_id for account in accounts] == ["client-a", "client-b"]
    assert all(account.tokens is None for account in accounts)
    assert await restarted.selected_account() == "client-b"
    assert await read(restarted) == record
    assert await read(restarted, "client-b") == other
    with pytest.raises(CredentialError, match="not found"):
        await restarted.select_account("missing")
    assert await restarted.selected_account() == "client-b"


async def test_selection_does_not_lose_concurrent_account_write(
    store: LocalCredentialStore, backend: Backend, tmp_path: Path, record: CredentialRecord
) -> None:
    await write(store, record)
    other = make_store(backend, tmp_path)
    replacement = replace(record, operation="refresh")
    await asyncio.gather(store.select_account("client-a"), write(other, replacement))
    assert await other.selected_account() == "client-a"
    assert await read(other) == replacement


async def test_account_metadata_rejects_mismatched_key(store: LocalCredentialStore, record: CredentialRecord) -> None:
    await store.host_id()
    await asyncio.to_thread(store.write_value, account_key("wrong-id"), json.loads(encode_record(record)))
    with pytest.raises(CredentialError, match="metadata"):
        await list_accounts(store)


@pytest.mark.parametrize("value", ["{}", "[]", '{"version":true,"client_id":"x"}', '{"version":1,"client_id":null}'])
async def test_bad_selection_is_not_replaced(store: LocalCredentialStore, value: str) -> None:
    await store.host_id()
    await asyncio.to_thread(store.write_value, "selected", json.loads(value))
    with pytest.raises(CredentialError):
        await store.selected_account()


async def test_requires_registration_lock(store: LocalCredentialStore, record: CredentialRecord) -> None:
    with pytest.raises(CredentialError, match="lock"):
        await store.load(record.account.client_id)
    with pytest.raises(CredentialError, match="lock"):
        await store.save(record)
    async with store.lock("other"):
        with pytest.raises(CredentialError, match="lock"):
            await store.save(record)


async def test_accounts_remain_independent(store: LocalCredentialStore, record: CredentialRecord) -> None:
    other = CredentialRecord(replace(record.account, client_id="../../other?x=1", subject="subject-b"))
    await asyncio.gather(write(store, record), write(store, other))
    assert await read(store) == record
    assert await read(store, other.account.client_id) == other
    async with store.lock(record.account.client_id):
        # A different registration is not locked across its network operation.
        async with asyncio.timeout(2):
            assert await read(store, other.account.client_id) == other


async def test_cannot_reenter_lock(store: LocalCredentialStore) -> None:
    async with store.lock("client-a"):
        with pytest.raises(CredentialError, match="reentrant"):
            async with store.lock("client-a"):
                pass


async def test_cancelled_waiter_does_not_steal_lock(
    store: LocalCredentialStore, backend: Backend, tmp_path: Path
) -> None:
    other = make_store(backend, tmp_path)
    async with store.lock("client-a"):
        waiter = asyncio.create_task(read(other))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    async with asyncio.timeout(2):
        assert await read(other) is None


async def test_cancelled_save_keeps_lock_until_commit(
    store: LocalCredentialStore,
    record: CredentialRecord,
    backend: Backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, proceed = threading.Event(), threading.Event()
    original = store.write_value

    def delayed(key: str, value: dict[str, Any]) -> None:
        entered.set()
        assert proceed.wait(3)
        original(key, value)

    monkeypatch.setattr(store, "write_value", delayed)
    task = asyncio.create_task(write(store, record))
    assert await asyncio.to_thread(entered.wait, 2)
    waiter = asyncio.create_task(read(make_store(backend, tmp_path)))
    try:
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done() and not waiter.done()
    finally:
        proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await waiter == record


async def test_private_permissions(store: LocalCredentialStore, record: CredentialRecord) -> None:
    await write(store, record)
    await store.host_id()
    assert store.directory.path.stat().st_mode & 0o777 == 0o700
    for path in store.directory.path.rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)


async def test_rejects_insecure_directory(store: LocalCredentialStore) -> None:
    store.directory.path.mkdir(parents=True, mode=0o700)
    store.directory.path.chmod(0o777)
    with pytest.raises(CredentialError):
        await read(store)


async def test_rejects_symlink_directory(store: LocalCredentialStore, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    store.directory.path.parent.mkdir(parents=True, exist_ok=True)
    store.directory.path.symlink_to(target, target_is_directory=True)
    with pytest.raises(CredentialError):
        await read(store)


def data_path(store: LocalCredentialStore) -> Path:
    assert isinstance(store, (FileCredentialStore, SQLiteCredentialStore))
    return store.path


@pytest.mark.parametrize("invalid", ["mode", "symlink", "hardlink"])
async def test_rejects_unsafe_data_file(
    store: LocalCredentialStore,
    record: CredentialRecord,
    invalid: str,
) -> None:
    await write(store, record)
    path = data_path(store)
    if invalid == "mode":
        path.chmod(0o644)
    elif invalid == "hardlink":
        path.with_name("linked").hardlink_to(path)
    else:
        target = path.with_name("original")
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(CredentialError):
        await read(store)
    with pytest.raises(CredentialError):
        await write(store, record)


async def test_failure_preserves_previous_record(
    store: LocalCredentialStore,
    record: CredentialRecord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await write(store, record)
    replacement = CredentialRecord(replace(record.account, tokens=None))
    if isinstance(store, SQLiteCredentialStore):
        # Abort the SQL update inside the real transaction.
        with sqlite3.connect(store.path) as conn:
            conn.execute(
                "CREATE TRIGGER reject_update BEFORE UPDATE ON axio_credentials "
                "BEGIN SELECT RAISE(ABORT, 'secret'); END"
            )
    else:

        def unavailable(*args: Any, **kwargs: Any) -> None:
            raise OSError("secret")

        monkeypatch.setattr(os, "replace", unavailable)
    with pytest.raises(CredentialError) as error:
        await write(store, replacement)
    assert "secret" not in repr(error.value)
    assert await read(store) == record
    assert not list(store.directory.path.glob(".credential-*"))


async def test_corrupt_records_fail_without_exposing_contents(store: LocalCredentialStore) -> None:
    async with store.lock("client-a"):
        await asyncio.to_thread(store.write_value, account_key("client-a"), {"secret": 1})
        with pytest.raises(CredentialError) as error:
            await store.load("client-a")
    assert "secret" not in repr(error.value)


@pytest.mark.parametrize(
    "change",
    ["version", "identity", "scopes", "timestamp", "operation", "token_type", "expiry", "empty_scope", "empty_host"],
)
def test_codec_rejects_invalid_records(record: CredentialRecord, change: str) -> None:
    import json

    data = json.loads(encode_record(record))
    if change == "version":
        data["version"] = 2
    elif change == "identity":
        data["account"]["client_id"] = "other"
    elif change == "scopes":
        data["account"]["tokens"]["scopes"] = "openid"
    elif change == "timestamp":
        data["account"]["tokens"]["expires_at"] = float("inf")
    elif change == "token_type":
        data["account"]["tokens"]["token_type"] = "Basic"
    elif change == "expiry":
        data["account"]["tokens"]["expires_at"] = data["account"]["tokens"]["received_at"]
    elif change == "empty_scope":
        data["account"]["tokens"]["scopes"] = [""]
    elif change == "empty_host":
        data["account"]["host_id"] = ""
    else:
        data["operation"] = "unknown"
    with pytest.raises(CredentialError):
        decode_record(json.dumps(data), "client-a")


async def test_sqlite_uses_fresh_connections_in_one_thread(tmp_path: Path, record: CredentialRecord) -> None:
    events: list[tuple[str, int, bool]] = []

    class TracedConnection(sqlite3.Connection):
        readonly: bool = False

        def execute(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            events.append(("execute", threading.get_ident(), self.readonly))
            return super().execute(*args, **kwargs)

        def close(self) -> None:
            events.append(("close", threading.get_ident(), self.readonly))
            super().close()

    class TracedStore(SQLiteCredentialStore):
        def connect(self, *, readonly: bool) -> sqlite3.Connection:
            events.append(("connect", threading.get_ident(), readonly))
            conn = sqlite3.connect(
                self.path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True, factory=TracedConnection
            )
            conn.readonly = readonly
            if readonly:
                with pytest.raises(sqlite3.OperationalError, match="readonly"):
                    conn.execute("CREATE TABLE forbidden (x)")
            return conn

    store = TracedStore(tmp_path / "private" / "accounts.db")
    await write(store, record)
    assert await read(store) == record
    assert await read(store) == record
    assert [e[2] for e in events if e[0] == "connect"] == [False, True, True]
    assert sum(e[0] == "close" for e in events) == 3
    worker = None
    for operation, thread, readonly in events:
        assert thread != threading.get_ident()
        if operation == "connect":
            worker = thread
        assert thread == worker


def crash_after_marker(backend: Backend, root: str, record: CredentialRecord) -> None:
    async def run() -> None:
        store = make_store(backend, Path(root))
        async with store.lock(record.account.client_id):
            await store.save(replace(record, operation="refresh"))
            os._exit(0)

    asyncio.run(run())


async def test_process_crash_releases_lock_and_retains_marker(
    backend: Backend,
    tmp_path: Path,
    record: CredentialRecord,
) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=crash_after_marker, args=(backend, str(tmp_path), record))
    process.start()
    try:
        await asyncio.to_thread(process.join, 5)
        assert process.exitcode == 0
        async with asyncio.timeout(2):
            stored = await read(make_store(backend, tmp_path))
        assert stored is not None and stored.operation == "refresh"
    finally:
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 2)


@pytest.fixture
async def refresh_server() -> AsyncIterator[tuple[str, list[str]]]:
    calls: list[str] = []
    issuer = ""

    async def discovery(request: web.Request) -> web.Response:
        return web.json_response(
            {"issuer": issuer, "token_endpoint": issuer + "/token", "revocation_endpoint": issuer + "/revoke"}
        )

    async def refresh(request: web.Request) -> web.Response:
        form = await request.post()
        calls.append(str(form["refresh_token"]))
        await asyncio.sleep(0.05)
        return web.json_response(
            {
                "access_token": "rotated-access",
                "refresh_token": "rotated-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        )

    async def revoke(request: web.Request) -> web.Response:
        assert (await request.post())["token"] == "rotated-refresh"
        return web.Response()

    app = web.Application()
    app.router.add_get("/.well-known/openid-configuration", discovery)
    app.router.add_post("/token", refresh)
    app.router.add_post("/revoke", revoke)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    issuer = f"http://127.0.0.1:{runner.addresses[0][1]}"
    try:
        yield issuer, calls
    finally:
        await runner.cleanup()


def refresh_in_process(backend: Backend, root: str, issuer: str) -> None:
    async def run() -> None:
        async with aiohttp.ClientSession() as http:
            store = make_store(backend, Path(root))
            oauth = ChatGPTOAuth(http, "host-a", issuer=issuer)
            assert await ChatGPTSession(store, oauth, "client-a").access_token() == "rotated-access"

    asyncio.run(run())


async def test_processes_refresh_once_and_sign_out(
    store: LocalCredentialStore,
    backend: Backend,
    tmp_path: Path,
    record: CredentialRecord,
    refresh_server: tuple[str, list[str]],
) -> None:
    issuer, calls = refresh_server
    await save_account(store, replace(record.account, issuer=issuer))
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=refresh_in_process, args=(backend, str(tmp_path), issuer)) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        async with asyncio.timeout(10):
            while any(process.is_alive() for process in processes):
                await asyncio.sleep(0.05)
        assert all(process.exitcode == 0 for process in processes)
        assert calls == ["refresh-secret"]
        async with aiohttp.ClientSession() as http:
            oauth = ChatGPTOAuth(http, "host-a", issuer=issuer)
            session = ChatGPTSession(store, oauth, "client-a")
            assert (await session.sign_out()).revocation_confirmed
            with pytest.raises(SignInRequired):
                await ChatGPTSession(make_store(backend, tmp_path), oauth, "client-a").access_token()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 2)


def write_in_process(backend: Backend, root: str, record: CredentialRecord) -> None:
    asyncio.run(write(make_store(backend, Path(root)), record))


async def test_processes_preserve_different_accounts(
    backend: Backend,
    tmp_path: Path,
    record: CredentialRecord,
) -> None:
    records = [
        CredentialRecord(replace(record.account, client_id=f"client-{i}", subject=f"subject-{i}")) for i in range(4)
    ]
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=write_in_process, args=(backend, str(tmp_path), item)) for item in records]
    for process in processes:
        process.start()
    try:
        async with asyncio.timeout(10):
            while any(process.is_alive() for process in processes):
                await asyncio.sleep(0.05)
        assert all(process.exitcode == 0 for process in processes)
        store = make_store(backend, tmp_path)
        for item in records:
            assert await read(store, item.account.client_id) == item
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 2)
