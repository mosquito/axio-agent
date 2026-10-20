"""Archive identity, transaction recovery and indexing journal contracts."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from axio.blocks import TextBlock
from axio.messages import Message

from axio_context_sqlite import ContextConflictError, SQLiteContextDatabase, SQLiteContextStore, connect


def message(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[SQLiteContextDatabase]:
    conn = await connect(tmp_path / "archive.db")
    try:
        yield SQLiteContextDatabase(conn)
    finally:
        await conn.close()


async def test_identical_delivery_is_idempotent_and_updates_keep_the_original_position(
    database: SQLiteContextDatabase,
) -> None:
    store = database.store("s", "home")
    first = await store.upsert(message("Эй, Ричи!"), message_id="audio-1", metadata={"b": 2, "a": 1})
    await store.append(message("Other message"))
    repeated = await store.upsert(message("Эй, Ричи!"), message_id="audio-1", metadata={"a": 1, "b": 2})
    assert repeated == first
    assert len(await database.changes("home")) == 2
    updated = await store.upsert(
        message("Hello, Ричи!"),
        message_id="audio-1",
        status="interrupted",
        metadata={"played_ms": 320},
        expected_revision=1,
    )
    records = await store.get_records()
    assert updated.position == first.position == 0
    assert records[0] == updated and len(records) == 2
    assert updated.revision == 2 and updated.status == "interrupted"
    assert updated.metadata == {"played_ms": 320}
    changes = await database.changes("home")
    assert changes[0].record == changes[-1].record == updated  # current state, not historical snapshots
    with pytest.raises(ContextConflictError):
        await store.upsert(message("stale"), message_id="audio-1", expected_revision=1)
    assert len(await database.changes("home")) == 3


async def test_shared_factory_serializes_full_transactions_and_scopes_identity(
    database: SQLiteContextDatabase,
) -> None:
    first, second = database.store("s1", "home"), database.store("s2", "home")
    await asyncio.gather(*[(first if i % 2 else second).upsert(message("same"), message_id="same") for i in range(40)])
    assert len(await first.get_history()) == len(await second.get_history()) == 1
    assert len(await database.changes("home")) == 2
    elsewhere = database.store("s1", "other-project")
    await elsewhere.upsert(message("private elsewhere"), message_id="same")
    assert (await first.get_history()) == [message("same")]
    assert (await elsewhere.get_history()) == [message("private elsewhere")]
    await first.clear()
    assert (await elsewhere.get_history()) == [message("private elsewhere")]
    assert len(await database.changes("other-project")) == 1


async def test_two_connections_compete_with_compare_and_swap(tmp_path: Path) -> None:
    first = await connect(tmp_path / "shared.db")
    second = await connect(tmp_path / "shared.db")
    try:
        stores = [SQLiteContextDatabase(conn).store("s", "home") for conn in (first, second)]
        results = await asyncio.gather(
            *[
                store.upsert(message(str(index)), message_id="call", expected_revision=0)
                for index, store in enumerate(stores)
            ],
            return_exceptions=True,
        )
        assert sum(isinstance(result, ContextConflictError) for result in results) == 1
        assert len(await stores[0].get_history()) == 1
        assert len(await stores[1].database.changes("home")) == 1
    finally:
        await first.close()
        await second.close()


async def test_fork_copies_metadata_and_clear_generates_text_free_tombstones(database: SQLiteContextDatabase) -> None:
    store = database.store("s", "home")
    await store.upsert(message("delete this text"), message_id="audio", status="cancelled", metadata={"played_ms": 0})
    await store.set_context_tokens(12, 5)
    fork = await store.fork()
    assert fork.session_id != store.session_id
    original, copied = (await store.get_records())[0], (await fork.get_records())[0]
    assert copied.message_id == original.message_id
    assert copied.message == original.message and copied.metadata == original.metadata
    assert copied.status == original.status and copied.revision == original.revision
    assert await fork.get_context_tokens() == (12, 5)
    before = (await database.changes("home"))[-1].sequence
    await store.clear()
    assert await store.get_history() == [] and await store.get_context_tokens() == (0, 0)
    assert await fork.get_history() == [message("delete this text")]
    changes = await database.changes("home", before)
    assert len(changes) == 1 and changes[0].record is None
    assert changes[0].message_id == "audio" and changes[0].session_id == "s"
    async with database.connection.execute("PRAGMA table_info(axio_context_changes)") as cursor:
        assert {row[1] for row in await cursor.fetchall()} == {"sequence", "session_id", "project", "message_id"}


async def test_failed_commit_rolls_back_message_and_journal(
    database: SQLiteContextDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = database.store("s", "home")
    first = await store.upsert(message("before"), message_id="a")

    async def fail() -> None:
        raise RuntimeError("commit failed")

    with monkeypatch.context() as patch:
        patch.setattr(database.connection, "commit", fail)
        with pytest.raises(RuntimeError, match="commit failed"):
            await store.upsert(message("after"), message_id="a")
    assert (await store.get_records()) == [first]
    assert len(await database.changes("home")) == 1
    assert not database.connection.in_transaction


@pytest.mark.parametrize("operation", ["insert", "update", "clear", "fork"])
async def test_journal_failure_rolls_back_the_entire_operation(
    database: SQLiteContextDatabase, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = database.store("s", "home")
    first = await store.upsert(message("before"), message_id="a")
    await store.set_context_tokens(12, 5)
    changes = await database.changes("home")
    execute = database.connection.execute

    def fail(sql: str, parameters: Any = None) -> Any:
        if "INSERT INTO axio_context_changes" in sql:
            raise RuntimeError("journal failed")
        return execute(sql, parameters)

    with monkeypatch.context() as patch:
        patch.setattr(database.connection, "execute", fail)
        with pytest.raises(RuntimeError, match="journal failed"):
            match operation:
                case "insert":
                    await store.upsert(message("new"), message_id="b")
                case "update":
                    await store.upsert(message("changed"), message_id="a")
                case "clear":
                    await store.clear()
                case "fork":
                    await store.fork()
    assert await store.get_records() == [first]
    assert await store.get_context_tokens() == (12, 5)
    assert await database.changes("home") == changes
    assert len(await store.list_sessions()) == 1
    assert not database.connection.in_transaction


async def test_fresh_database_has_no_triggers(database: SQLiteContextDatabase) -> None:
    async with database.connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'") as cursor:
        assert await cursor.fetchall() == []


async def test_cancellation_rolls_back_before_another_writer_can_enter(
    database: SQLiteContextDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = asyncio.Event()

    async def stalled_commit() -> None:
        ready.set()
        await asyncio.Event().wait()

    store = database.store("s", "home")
    with monkeypatch.context() as patch:
        patch.setattr(database.connection, "commit", stalled_commit)
        writer = asyncio.create_task(store.upsert(message("cancel"), message_id="a"))
        await asyncio.wait_for(ready.wait(), 1)
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
    assert not database.connection.in_transaction
    assert await store.get_records() == [] and await database.changes("home") == []
    await store.append(message("next writer"))
    assert await store.get_history() == [message("next writer")]


def make_legacy(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE axio_context_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, project TEXT NOT NULL, position INTEGER NOT NULL, "
            "role TEXT NOT NULL, content TEXT NOT NULL, "
            "created_at TEXT NOT NULL DEFAULT (datetime('now')), UNIQUE(session_id, position))"
        )
        conn.execute("PRAGMA user_version=42")
        conn.execute(
            "INSERT INTO axio_context_messages(session_id,project,position,role,content) "
            "VALUES('s','home',0,'user',?)",
            (json.dumps(message("legacy").to_dict()["content"]),),
        )


async def test_existing_database_is_migrated_once_without_triggers(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    make_legacy(path)
    for _ in range(2):
        conn = await connect(path)
        try:
            database = SQLiteContextDatabase(conn)
            records = await database.store("s", "home").get_records()
            assert records[0].message_id == "legacy:1" and records[0].revision == 1
            assert records[0].metadata == {} and records[0].status == "completed"
            assert len(await database.changes("home")) == 1
            async with conn.execute("PRAGMA user_version") as cursor:
                assert await cursor.fetchone() == (42,)
            async with conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'") as cursor:
                assert await cursor.fetchall() == []
        finally:
            await conn.close()


async def test_migration_failure_does_not_leave_half_installed_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "legacy.db"
    make_legacy(path)
    execute = aiosqlite.Connection.execute

    def fail(conn: aiosqlite.Connection, sql: str, parameters: Any = None) -> Any:
        if "CREATE INDEX IF NOT EXISTS idx_axio_context_changes_project" in sql:
            raise RuntimeError("migration interrupted")
        return execute(conn, sql, parameters)

    with monkeypatch.context() as patch:
        patch.setattr(aiosqlite.Connection, "execute", fail)
        with pytest.raises(RuntimeError, match="migration interrupted"):
            await connect(path)
    with sqlite3.connect(path) as conn:
        assert "message_id" not in {row[1] for row in conn.execute("PRAGMA table_info(axio_context_messages)")}
        assert conn.execute("SELECT count(*) FROM axio_context_messages").fetchone() == (1,)
    reopened = await connect(path)
    await reopened.close()


async def test_legacy_constructors_on_one_connection_share_transaction_coordination(
    database: SQLiteContextDatabase,
) -> None:
    stores = [SQLiteContextStore(database.connection, str(i), "home") for i in range(5)]
    assert all(store.database.lock is database.lock for store in stores)
    await asyncio.gather(*[store.append(message("hello")) for store in stores])
    histories = await asyncio.gather(*[store.get_history() for store in stores])
    assert all(len(history) == 1 for history in histories)
    assert len(await database.changes("home")) == 5


async def test_managed_operations_do_not_commit_or_rollback_a_callers_transaction(
    database: SQLiteContextDatabase,
) -> None:
    await database.connection.execute("CREATE TABLE app_config(value TEXT)")
    await database.connection.execute("INSERT INTO app_config VALUES ('uncommitted')")
    with pytest.raises(RuntimeError, match="unmanaged"):
        await database.store("s", "home").append(message("must not commit"))
    assert database.connection.in_transaction
    await database.connection.rollback()
    async with database.connection.execute("SELECT count(*) FROM app_config") as cursor:
        assert await cursor.fetchone() == (0,)
    assert await database.changes("home") == []


async def test_journal_pages_are_project_scoped_and_sequences_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "pages.db"
    conn = await connect(path)
    try:
        database = SQLiteContextDatabase(conn)
        for index in range(6):
            await database.store("s", "home" if index % 2 else "other").append(message(str(index)))
        first = await database.changes("home", limit=2)
        assert len(first) == 2 and first[0].sequence < first[1].sequence
        last = await database.changes("home", first[-1].sequence, limit=2)
        assert len(last) == 1 and last[0].sequence > first[-1].sequence
        watermark = last[0].sequence
    finally:
        await conn.close()
    reopened = await connect(path)
    try:
        database = SQLiteContextDatabase(reopened)
        await database.store("s", "home").append(message("after restart"))
        replay = await database.changes("home", watermark)
        assert len(replay) == 1 and replay[0].sequence > watermark
    finally:
        await reopened.close()


@pytest.mark.parametrize("metadata", [{"not_json": b"raw"}, {"nan": float("nan")}, {1: "bad key"}, []])
async def test_invalid_metadata_never_writes_a_record_or_journal(
    database: SQLiteContextDatabase, metadata: Any
) -> None:
    with pytest.raises((ValueError, TypeError)):
        await database.store("s", "home").upsert(message("data"), message_id="a", metadata=metadata)
    assert await database.changes("home") == []
    assert await database.store("s", "home").get_history() == []


@pytest.mark.parametrize("stage", ["before_commit", "after_commit"])
async def test_process_exit_preserves_transaction_and_checkpoint_contract(tmp_path: Path, stage: str) -> None:
    path = tmp_path / "crash.db"
    script = """
import asyncio, json, os, sys
from axio_context_sqlite import connect, SQLiteContextDatabase
from axio.messages import Message
from axio.blocks import TextBlock
async def write():
    conn = await connect(sys.argv[1])
    if sys.argv[2] == 'before_commit':
        async with SQLiteContextDatabase(conn).transaction():
            await conn.execute(
                "INSERT INTO axio_context_messages(session_id,project,position,role,content) "
                "VALUES('s','home',0,'user',?)",
                (json.dumps([{'type': 'text', 'text': 'uncommitted'}]),))
            os._exit(0)
    await SQLiteContextDatabase(conn).store('s', 'home').upsert(
        Message(role='user', content=[TextBlock(text='committed')]), message_id='item')
    os._exit(0)
asyncio.run(write())
"""
    await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, str(path), stage], check=True, timeout=10)
    conn = await connect(path)
    try:
        database = SQLiteContextDatabase(conn)
        changes = await database.changes("home", 0)
        if stage == "before_commit":
            assert changes == [] and await database.store("s", "home").get_history() == []
            return
        assert len(changes) == 1 and changes[0].record is not None
        assert changes[0].record.message == message("committed")
        # Consumer applied a result, but crashed before checkpointing: replay is identical.
        assert await database.changes("home", 0) == changes
        assert await database.changes("home", changes[-1].sequence) == []
    finally:
        await conn.close()
