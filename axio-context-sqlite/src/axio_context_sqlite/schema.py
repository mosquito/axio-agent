"""Additive archive migration and transactionally committed invalidations."""

from __future__ import annotations

import aiosqlite


async def migrate_archive(conn: aiosqlite.Connection) -> None:
    """Migrate without changing SQLite user_version owned by other applications.

    Identity, metadata and the initial journal are installed in one transaction.
    The legacy INSERT shape remains writable through an identity trigger.
    """
    try:
        await conn.execute("BEGIN IMMEDIATE")
        async with conn.execute("PRAGMA table_info(axio_context_messages)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        additions = (
            ("message_id", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'completed'"),
            ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
            ("revision", "INTEGER NOT NULL DEFAULT 1"),
        )
        for name, declaration in additions:
            if name not in columns:
                await conn.execute(f"ALTER TABLE axio_context_messages ADD COLUMN {name} {declaration}")
        await conn.execute("UPDATE axio_context_messages SET message_id = 'legacy:' || id WHERE message_id IS NULL")
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_axio_context_message_identity "
            "ON axio_context_messages(project, session_id, message_id)"
        )
        async with conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='axio_context_changes'"
        ) as cursor:
            journal_exists = await cursor.fetchone() is not None
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS axio_context_changes ("
            "sequence INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "project TEXT NOT NULL, message_id TEXT NOT NULL)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_axio_context_changes_project ON axio_context_changes(project, sequence)"
        )
        if not journal_exists:
            await conn.execute(
                "INSERT INTO axio_context_changes(session_id, project, message_id) "
                "SELECT session_id, project, message_id FROM axio_context_messages ORDER BY id"
            )
        await conn.execute(
            "CREATE TRIGGER IF NOT EXISTS axio_context_assign_identity AFTER INSERT ON axio_context_messages "
            "WHEN NEW.message_id IS NULL BEGIN "
            "UPDATE axio_context_messages SET message_id = 'legacy:' || NEW.id WHERE id = NEW.id; END"
        )
        for operation, row, condition in (
            ("INSERT", "NEW", "WHEN NEW.message_id IS NOT NULL"),
            ("UPDATE", "NEW", ""),
            ("DELETE", "OLD", ""),
        ):
            await conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS axio_context_journal_{operation.lower()} "
                f"AFTER {operation} ON axio_context_messages {condition} BEGIN "
                "INSERT INTO axio_context_changes(session_id, project, message_id) "
                f"VALUES ({row}.session_id, {row}.project, {row}.message_id); END"
            )
        await conn.commit()
    except BaseException:
        await conn.rollback()
        raise
