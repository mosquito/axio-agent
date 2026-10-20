"""SQLiteContextStore: persistent conversation storage backed by SQLite."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
from axio.context import ContextStore, SessionInfo
from axio.messages import Message

from .database import SQLiteContextDatabase
from .records import ContextRecord
from .schema import migrate_archive

# Compress content payloads above this size (bytes of UTF-8 JSON).
COMPRESS_THRESHOLD = 512


def compress_payload(data: str) -> str:
    raw = data.encode()
    if len(raw) < COMPRESS_THRESHOLD:
        return "plain:" + data
    return "gzip:" + base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode()


def decompress_payload(data: str) -> str:
    if data.startswith("gzip:"):
        return gzip.decompress(base64.b64decode(data[5:])).decode()
    if data.startswith("plain:"):
        return data[6:]
    # raw JSON
    return data


async def connect(db_path: str | Path) -> aiosqlite.Connection:
    """Open (or create) a SQLite database and initialise the schema.

    The caller is responsible for closing the returned connection.
    """
    path = Path(db_path)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    try:
        await conn.create_function("compress_payload", 1, compress_payload, deterministic=True)
        await conn.create_function("decompress_payload", 1, decompress_payload, deterministic=True)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS axio_context_messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id TEXT NOT NULL,"
            "  project TEXT NOT NULL,"
            "  position INTEGER NOT NULL,"
            "  role TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  created_at TEXT NOT NULL DEFAULT (datetime('now')),"
            "  UNIQUE(session_id, position)"
            ")"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_axio_context_messages_session ON axio_context_messages(session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_axio_context_messages_project ON axio_context_messages(project)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS axio_context_tokens ("
            "  session_id TEXT NOT NULL,"
            "  project TEXT NOT NULL,"
            "  input_tokens INTEGER NOT NULL DEFAULT 0,"
            "  output_tokens INTEGER NOT NULL DEFAULT 0,"
            "  PRIMARY KEY(session_id, project)"
            ")"
        )
        await migrate_archive(conn)
        return conn
    except BaseException:
        await conn.close()
        raise


def _extract_preview(content_json: str, max_len: int = 80) -> str:
    """Extract text preview from serialized content JSON."""
    try:
        blocks = json.loads(content_json)
        for b in blocks:
            if b.get("type") == "text":
                text: str = b["text"]
                return text[:max_len] + ("..." if len(text) > max_len else "")
    except (json.JSONDecodeError, KeyError):
        pass
    return "(no preview)"


class SQLiteContextStore(ContextStore):
    """Persistent conversation storage backed by SQLite.

    The caller owns the connection and is responsible for closing it.
    Use :func:`connect` to open a properly initialized connection.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        project: str | None = None,
        db_name: str = "axio_context",
        *,
        database: SQLiteContextDatabase | None = None,
    ) -> None:
        if database is not None and database.connection is not conn:
            raise ValueError("database must coordinate the supplied connection")
        self.database = database or SQLiteContextDatabase(conn)
        self._conn = conn
        self._db_name = db_name
        self._session_id = session_id
        self._project = project or str(Path.cwd().resolve())

    @property
    def session_id(self) -> str:
        return self._session_id

    async def append(self, message: Message) -> None:
        await self.upsert(message, message_id=uuid4().hex)

    async def upsert(
        self,
        message: Message,
        *,
        message_id: str,
        status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> ContextRecord:
        """Persist a stable message snapshot; identical deliveries do not duplicate it."""
        return await self.database.upsert(
            self._session_id,
            self._project,
            message,
            message_id=message_id,
            status=status,
            metadata=metadata,
            expected_revision=expected_revision,
        )

    async def get_records(self) -> list[ContextRecord]:
        """Return current message snapshots in their original conversation order."""
        return await self.database.records(self._session_id, self._project)

    async def get_history(self) -> list[Message]:
        return [record.message for record in await self.get_records()]

    async def clear(self) -> None:
        async with self.database.transaction():
            await self._conn.execute(
                "INSERT INTO axio_context_changes(session_id, project, message_id) "
                "SELECT session_id, project, message_id FROM axio_context_messages "
                "WHERE session_id = ? AND project = ? ORDER BY id",
                (self._session_id, self._project),
            )
            await self._conn.execute(
                "DELETE FROM axio_context_messages WHERE session_id = ? AND project = ?",
                (self._session_id, self._project),
            )
            await self._conn.execute(
                "DELETE FROM axio_context_tokens WHERE session_id = ? AND project = ?",
                (self._session_id, self._project),
            )

    async def fork(self) -> SQLiteContextStore:
        async with self.database.transaction():
            new_id = uuid4().hex
            await self._conn.execute(
                "INSERT INTO axio_context_messages (session_id, project, position, role, content, "
                "message_id, status, metadata, revision)"
                "SELECT ?, project, position, role, content, message_id, status, metadata, revision "
                "FROM axio_context_messages WHERE session_id = ? AND project = ?",
                (new_id, self._session_id, self._project),
            )
            await self._conn.execute(
                "INSERT INTO axio_context_changes(session_id, project, message_id) "
                "SELECT session_id, project, message_id FROM axio_context_messages "
                "WHERE session_id = ? AND project = ? ORDER BY id",
                (new_id, self._project),
            )
            await self._conn.execute(
                "INSERT OR IGNORE INTO axio_context_tokens (session_id, project, input_tokens, output_tokens) "
                "SELECT ?, project, input_tokens, output_tokens FROM axio_context_tokens "
                "WHERE session_id = ? AND project = ?",
                (new_id, self._session_id, self._project),
            )
            return SQLiteContextStore(self._conn, new_id, self._project, self._db_name, database=self.database)

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        async with self.database.transaction():
            await self._conn.execute(
                "INSERT INTO axio_context_tokens (session_id, project, input_tokens, output_tokens)"
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, project) DO UPDATE SET input_tokens=?, output_tokens=?",
                (self._session_id, self._project, input_tokens, output_tokens, input_tokens, output_tokens),
            )

    async def add_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        async with self.database.transaction():
            await self._conn.execute(
                "INSERT INTO axio_context_tokens (session_id, project, input_tokens, output_tokens)"
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, project) DO UPDATE "
                "SET input_tokens = input_tokens + excluded.input_tokens, "
                "    output_tokens = output_tokens + excluded.output_tokens",
                (self._session_id, self._project, input_tokens, output_tokens),
            )

    async def get_context_tokens(self) -> tuple[int, int]:
        async with self.database.lock:
            async with self._conn.execute(
                "SELECT input_tokens, output_tokens FROM axio_context_tokens WHERE session_id = ? AND project = ?",
                (self._session_id, self._project),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return 0, 0
            return int(row[0]), int(row[1])

    async def close(self) -> None:
        """No-op: the caller owns the connection."""

    async def list_sessions(self) -> list[SessionInfo]:
        """List all sessions for a project, newest first."""
        async with self.database.lock:
            async with self._conn.execute(
                "SELECT m.session_id, COUNT(*) as cnt, "
                "(SELECT decompress_payload(content) FROM axio_context_messages WHERE session_id = m.session_id "
                "AND project = m.project AND role = 'user' ORDER BY position LIMIT 1) as first_content, "
                "MIN(m.created_at) as created, "
                "COALESCE(ct.input_tokens, 0), COALESCE(ct.output_tokens, 0) "
                "FROM axio_context_messages m "
                "LEFT JOIN axio_context_tokens ct ON ct.session_id = m.session_id AND ct.project = m.project "
                "WHERE m.project = ? "
                "GROUP BY m.session_id ORDER BY created DESC",
                (self._project,),
            ) as cursor:
                rows = await cursor.fetchall()
            result: list[SessionInfo] = []
            for session_id, count, first_content, created_at, in_tok, out_tok in rows:
                preview = _extract_preview(first_content) if first_content else "(no preview)"
                result.append(
                    SessionInfo(
                        session_id=session_id,
                        message_count=count,
                        preview=preview,
                        created_at=created_at,
                        input_tokens=int(in_tok),
                        output_tokens=int(out_tok),
                    )
                )
            return result
