"""Application context backend with one shared SQLite connection."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import aiosqlite

from .store import SQLiteContextStore, connect


class SQLiteContextBackend:
    def __init__(self) -> None:
        self.connection: aiosqlite.Connection | None = None

    async def init(self, path: Path) -> None:
        self.connection = await connect(path)

    def create(self, session_id: str | None = None) -> SQLiteContextStore:
        if self.connection is None:
            raise RuntimeError("SQLite context backend is not initialized")
        return SQLiteContextStore(self.connection, session_id or uuid4().hex)

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()
            self.connection = None
