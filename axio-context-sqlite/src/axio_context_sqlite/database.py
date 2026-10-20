"""Shared SQLite writer and session-store factory for archive applications."""

from __future__ import annotations

import asyncio
import json
import threading
from _thread import LockType
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

import aiosqlite
from axio.messages import Message

from .records import RECORD_COLUMNS, ContextChange, ContextConflictError, ContextRecord, decode_record

if TYPE_CHECKING:
    from .store import SQLiteContextStore


@dataclass(slots=True)
class ConnectionLocks:
    """Coordinate legacy constructors too, without retaining closed connections."""

    locks: WeakKeyDictionary[aiosqlite.Connection, asyncio.Lock] = field(default_factory=WeakKeyDictionary)
    mutex: LockType = field(default_factory=threading.Lock, repr=False)

    def get(self, connection: aiosqlite.Connection) -> asyncio.Lock:
        with self.mutex:
            lock = self.locks.get(connection)
            if lock is None:
                lock = self.locks[connection] = asyncio.Lock()
            return lock


CONNECTION_LOCKS = ConnectionLocks()


@dataclass(slots=True)
class SQLiteContextDatabase:
    """Coordinate stores sharing one caller-owned, initialized connection.

    Create session stores through store(). Legacy constructors and independently
    created facades on the same connection share coordination automatically.
    SQLite's worker serializes statements; this lock also serializes complete
    transactions and reads. Use one event loop and do not mix raw transactions
    concurrently with these managed operations.
    Separate connections are coordinated by SQLite's write lock and busy timeout.
    """

    connection: aiosqlite.Connection = field(repr=False)
    lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lock = CONNECTION_LOCKS.get(self.connection)

    def store(self, session_id: str, project: str | None = None) -> SQLiteContextStore:
        from .store import SQLiteContextStore

        return SQLiteContextStore(self.connection, session_id, project, database=self)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Own one transaction; reject an unrelated caller's pending transaction.

        If cancellation races COMMIT, the outcome can be committed: retry using
        the same message ID. SQL work already queued in aiosqlite is not killed
        by cancellation. Rollback is queued before releasing the writer lock.
        Do not nest managed operations inside this context; use raw SQL here.
        """
        async with self.lock:
            if self.connection.in_transaction:
                raise RuntimeError("The context connection already has an unmanaged transaction")
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                yield
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

    async def upsert(
        self,
        session_id: str,
        project: str,
        message: Message,
        *,
        message_id: str,
        status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> ContextRecord:
        """Insert or replace one complete snapshot without changing its position.

        Identical delivery does not create a revision or journal entry. Without
        expected_revision, different snapshots use last-writer-wins semantics.
        Use revision zero to require insertion, or a read revision for compare
        and swap. Metadata is replaced, not shallow-merged.
        """
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("message_id must be nonempty")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be nonempty")
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
            raise ValueError("expected_revision must be a nonnegative integer")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        values = dict(metadata or {})
        if not all(isinstance(key, str) for key in values):
            raise ValueError("Metadata keys must be strings")
        metadata_json = json.dumps(values, sort_keys=True, ensure_ascii=False, allow_nan=False)
        content = json.dumps(message.to_dict()["content"], sort_keys=True, ensure_ascii=False, allow_nan=False)
        async with self.transaction():
            async with self.connection.execute(
                "SELECT revision FROM axio_context_messages WHERE session_id=? AND project=? AND message_id=?",
                (session_id, project, message_id),
            ) as cursor:
                existing = await cursor.fetchone()
            revision = existing[0] if existing else 0
            if expected_revision is not None and revision != expected_revision:
                raise ContextConflictError("The context record revision changed")
            if existing is None and message_id.startswith("legacy:"):
                raise ValueError("The legacy: message ID prefix is reserved for migrated records")
            async with self.connection.execute(
                "INSERT INTO axio_context_messages(session_id, project, message_id, "
                "position, role, content, status, metadata) "
                "VALUES (?, ?, ?, (SELECT COALESCE(MAX(position)+1, 0) "
                "FROM axio_context_messages WHERE session_id=?), "
                "?, compress_payload(?), ?, ?) "
                "ON CONFLICT(project, session_id, message_id) DO UPDATE SET "
                "role=excluded.role, content=excluded.content, status=excluded.status, metadata=excluded.metadata, "
                "revision=axio_context_messages.revision+1 "
                "WHERE axio_context_messages.role != excluded.role OR "
                "decompress_payload(axio_context_messages.content) != "
                "decompress_payload(excluded.content) OR "
                "axio_context_messages.status != excluded.status OR "
                "axio_context_messages.metadata != excluded.metadata",
                (session_id, project, message_id, session_id, message.role, content, status, metadata_json),
            ) as cursor:
                changed = cursor.rowcount > 0
            if changed:
                await self.connection.execute(
                    "INSERT INTO axio_context_changes(session_id, project, message_id) VALUES (?, ?, ?)",
                    (session_id, project, message_id),
                )
            async with self.connection.execute(
                f"SELECT {RECORD_COLUMNS} FROM axio_context_messages m "
                "WHERE m.session_id=? AND m.project=? AND m.message_id=?",
                (session_id, project, message_id),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            return decode_record(row)

    async def records(self, session_id: str, project: str) -> list[ContextRecord]:
        async with (
            self.lock,
            self.connection.execute(
                f"SELECT {RECORD_COLUMNS} FROM axio_context_messages m "
                "WHERE m.session_id=? AND m.project=? ORDER BY m.position",
                (session_id, project),
            ) as cursor,
        ):
            return [decode_record(row) for row in await cursor.fetchall()]

    async def changes(self, project: str, after: int = 0, *, limit: int = 100) -> list[ContextChange]:
        """Read a bounded project journal page joined to current source records.

        Apply record or deletion by (project, session_id, message_id), commit
        your sink, then persist the last sequence. A crash before the checkpoint
        replays invalidations safely. Empty pages do not advance the cursor.
        Neither deleted text nor old audio/image payloads are copied to the journal.
        """
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("after must be nonnegative and limit must be between 1 and 1000")
        async with (
            self.lock,
            self.connection.execute(
                f"SELECT c.sequence, c.session_id, c.project, c.message_id, {RECORD_COLUMNS} "
                "FROM axio_context_changes c LEFT JOIN axio_context_messages m "
                "ON m.project=c.project AND m.session_id=c.session_id AND m.message_id=c.message_id "
                "WHERE c.project=? AND c.sequence>? ORDER BY c.sequence LIMIT ?",
                (project, after, limit),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [
            ContextChange(row[0], row[1], row[2], row[3], None if row[4] is None else decode_record(row[4:]))
            for row in rows
        ]
