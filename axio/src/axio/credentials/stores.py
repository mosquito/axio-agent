"""File and SQLite credential stores for local POSIX filesystems."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, get_args, get_origin

from .codec import CredentialCodec, record_id, record_key
from .files import ProtectedDirectory, check_file, open_file, sync_directory
from .types import CredentialError, complete_operation


class Credentials[T](ABC):
    """Shared process locking and cancellation handling for local backends.

    Override read_value/write_value to change persistence. These synchronous
    hooks run in a worker thread. Each write must commit before returning.
    load/save require lock(key_id) in the same asyncio task. Locks are not
    reentrant. The local filesystem must support flock and durable writes.
    """

    filename: ClassVar[str] = "credentials.auth"

    def __init__(self, directory: Path, locks: Path, *, record_type: type[T] | None = None) -> None:
        self.record_type = record_type
        self.directory = ProtectedDirectory(directory)
        self.locks = ProtectedDirectory(locks)
        self.owners: dict[str, asyncio.Task[Any]] = {}

    @cached_property
    def codec(self) -> CredentialCodec[T]:
        """Resolve T from a typed instance, a typed subclass, or record_type."""
        if self.record_type is not None:
            return CredentialCodec(self.record_type)
        aliases = [getattr(self, "__orig_class__", None)]
        for cls in type(self).__mro__:
            aliases.extend(cls.__dict__.get("__orig_bases__", ()))
        for alias in aliases:
            origin, args = get_origin(alias), get_args(alias)
            if isinstance(origin, type) and issubclass(origin, Credentials) and args and isinstance(args[0], type):
                return CredentialCodec(args[0])
        raise TypeError("Specify the credential dataclass as a type parameter or record_type.")

    @asynccontextmanager
    async def lock(self, key_id: str) -> AsyncIterator[None]:
        key = record_key(key_id)
        task = asyncio.current_task()
        assert task is not None
        if self.owners.get(key_id) is task:
            raise CredentialError("Credential locks are not reentrant.")
        await complete_operation(asyncio.to_thread(self.directory.prepare))
        async with self.locks.lock(key):
            self.owners[key_id] = task
            try:
                yield
            finally:
                del self.owners[key_id]

    def require_lock(self, key_id: str) -> None:
        if self.owners.get(key_id) is not asyncio.current_task():
            raise CredentialError("Acquire this registration's lock before reading or writing credentials.")

    async def load(self, key_id: str) -> T | None:
        self.require_lock(key_id)
        raw = await complete_operation(asyncio.to_thread(self.read_value, record_key(key_id)))
        return None if raw is None else self.codec.from_dict(raw, key_id)

    async def save(self, record: T) -> None:
        key_id = record_id(record)
        self.require_lock(key_id)
        data = self.codec.to_dict(record)
        await complete_operation(asyncio.to_thread(self.write_value, record_key(key_id), data))

    async def host_id(self) -> str:
        """Create or load a stable host ID under an independent process lock."""
        await complete_operation(asyncio.to_thread(self.directory.prepare))
        async with self.locks.lock("host"):
            return await complete_operation(asyncio.to_thread(self.read_or_create_host))

    async def list_records(self) -> list[T]:
        """Read a typed snapshot. Records can contain secrets; do not print them."""
        values = await complete_operation(asyncio.to_thread(self.read_values))
        records = []
        for key, value in values.items():
            if not key.startswith("account-"):
                continue
            record = self.codec.from_dict(value)
            if record_key(record_id(record)) != key:
                raise CredentialError("Invalid credential record metadata.")
            records.append(record)
        return sorted(records, key=record_id)

    async def selected_account(self) -> str | None:
        """Read the selected record key. This does not change active sessions."""
        raw = await complete_operation(asyncio.to_thread(self.read_value, "selected"))
        if raw is None:
            return None
        try:
            data = raw
            if type(data["version"]) is not int or data["version"] != 1:
                raise ValueError
            # Version 1 called the selected record key client_id.
            key_id = data["client_id"]
            if not isinstance(key_id, str) or not key_id:
                raise ValueError
            return key_id
        except (KeyError, TypeError, ValueError):
            raise CredentialError("Invalid selected account reference.") from None

    async def select_account(self, key_id: str) -> None:
        """Select an existing registration for future sessions."""
        async with self.lock(key_id):
            if await self.load(key_id) is None:
                raise CredentialError("The account registration was not found.")
        async with self.locks.lock("selection"):
            raw = {"version": 1, "client_id": key_id}
            await complete_operation(asyncio.to_thread(self.write_value, "selected", raw))

    def read_values(self) -> dict[str, Any]:
        """Read a snapshot for account discovery. Optional for custom adapters."""
        raise NotImplementedError("This store does not support account discovery.")

    def read_or_create_host(self) -> str:
        raw = self.read_value("host")
        if raw is None:
            host_id = uuid.uuid4().urn
            self.write_value("host", {"version": 1, "host_id": host_id})
            return host_id
        try:
            data = raw
            if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
                raise ValueError
            host_id = data["host_id"]
            if not isinstance(host_id, str) or not host_id:
                raise ValueError
            return host_id
        except (ValueError, KeyError, TypeError):
            raise CredentialError("Invalid stored host identity.") from None

    @abstractmethod
    def read_value(self, key: str) -> dict[str, Any] | None:
        """Read the latest committed value. Called under the corresponding lock."""

    @abstractmethod
    def write_value(self, key: str, value: dict[str, Any]) -> None:
        """Commit a whole value durably. Called under the corresponding lock."""


def default_credentials_path(filename: str = "credentials.auth") -> Path:
    """Use XDG_CONFIG_HOME when absolute, otherwise ~/.config."""
    configured = os.environ.get("XDG_CONFIG_HOME", "")
    root = Path(configured) if configured and Path(configured).is_absolute() else Path.home() / ".config"
    return root / "axio" / filename


class FileCredentialStore[T](Credentials[T]):
    """Store registrations in one JSON file with atomic replacement.

    Defaults to $XDG_CONFIG_HOME/axio/<filename> or ~/.config/axio/<filename>.
    Creates directories with mode 0700 and files with mode 0600. Rejects files
    with unsafe permissions, symlinks, and hard links. Keep the adjacent .locks
    directory intact while any process uses the store. Credentials are protected
    by filesystem access controls, not encrypted at rest.
    """

    def __init__(self, path: str | Path | None = None, *, record_type: type[T] | None = None) -> None:
        self.path = (Path(path) if path is not None else default_credentials_path(self.filename)).absolute()
        super().__init__(self.path.parent, self.path.with_name(self.path.name + ".locks"), record_type=record_type)

    def read_values(self) -> dict[str, Any]:
        raw = self.directory.read(self.path.name)
        if raw is None:
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
                raise ValueError
            values = data["values"]
            if not isinstance(values, dict):
                raise ValueError
            return values
        except (ValueError, TypeError, KeyError):
            raise CredentialError("Invalid credential file.") from None

    def read_value(self, key: str) -> dict[str, Any] | None:
        values = self.read_values()
        if key not in values:
            return None
        value = values[key]
        if not isinstance(value, dict):
            raise CredentialError("Invalid credential value.")
        return value

    def write_value(self, key: str, value: dict[str, Any]) -> None:
        # Serialize whole-file replacements across different account locks.
        # This lock spans only local I/O, never an OAuth request.
        with self.locks.blocking_lock("file"):
            values = self.read_values()
            try:
                values[key] = value
                raw = json.dumps({"version": 1, "values": values}, allow_nan=False, separators=(",", ":"))
            except (ValueError, TypeError):
                raise CredentialError("Invalid credential value.") from None
            self.directory.write(self.path.name, raw)


class SQLiteCredentialStore[T](Credentials[T]):
    """Store registrations in a dedicated SQLite database in a private directory.

    Each load opens a mode=ro connection, reads, and closes in one worker-thread
    call. Each write opens its own connection, commits, and closes. The adjacent
    <database>.locks directory coordinates refresh independently of SQL commits.
    Connections never span an HTTP request. No close() call is necessary.
    """

    def __init__(self, path: str | Path, *, record_type: type[T] | None = None) -> None:
        self.path = Path(path).absolute()
        super().__init__(self.path.parent, self.path.with_name(self.path.name + ".locks"), record_type=record_type)

    def check_database(self, *, create: bool = False) -> bool:
        self.directory.prepare()
        try:
            for suffix in ("-journal", "-wal", "-shm"):
                try:
                    check_file(Path(str(self.path) + suffix).lstat())
                except FileNotFoundError:
                    pass
            flags = os.O_RDWR | os.O_CREAT if create else os.O_RDONLY
            fd = open_file(self.path, flags)
            os.close(fd)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            raise CredentialError("Could not open the protected credential database.") from None

    def connect(self, *, readonly: bool) -> sqlite3.Connection:
        """Open one connection. The calling worker thread must close it."""
        mode = "ro" if readonly else "rw"
        conn = sqlite3.connect(self.path.as_uri() + "?mode=" + mode, uri=True, timeout=5)
        try:
            conn.execute("PRAGMA trusted_schema=OFF")
            if not readonly:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("PRAGMA synchronous=EXTRA")
                conn.execute("PRAGMA fullfsync=ON")
                conn.execute("PRAGMA secure_delete=ON")
            return conn
        except BaseException:
            conn.close()
            raise

    def table_name(self, conn: sqlite3.Connection) -> str:
        """Keep using an existing legacy table so old and new processes share writes."""
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' "
                "AND name IN ('axio_credentials', 'axio_chatgpt_credentials')"
            )
        }
        if len(tables) > 1:
            raise CredentialError("Ambiguous credential database tables.")
        return next(iter(tables), "axio_credentials")

    def read_value(self, key: str) -> dict[str, Any] | None:
        if not self.check_database():
            return None
        try:
            with closing(self.connect(readonly=True)) as conn:
                table = self.table_name(conn)
                if (
                    conn.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (table,)).fetchone()
                    is None
                ):
                    return None
                row = conn.execute(f"SELECT value FROM {table} WHERE key = ?", (key,)).fetchone()
                if row is None:
                    return None
                value = json.loads(row[0])
                if not isinstance(value, dict):
                    raise CredentialError("Invalid stored credential value.")
                return value
        except (sqlite3.Error, ValueError, TypeError):
            raise CredentialError("Could not read the credential database.") from None

    def read_values(self) -> dict[str, Any]:
        if not self.check_database():
            return {}
        try:
            with closing(self.connect(readonly=True)) as conn:
                table = self.table_name(conn)
                if (
                    conn.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (table,)).fetchone()
                    is None
                ):
                    return {}
                return {key: json.loads(value) for key, value in conn.execute(f"SELECT key, value FROM {table}")}
        except (sqlite3.Error, ValueError, TypeError):
            raise CredentialError("Could not read credential record metadata.") from None

    def write_value(self, key: str, value: dict[str, Any]) -> None:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":"))
        self.check_database(create=True)
        try:
            with closing(self.connect(readonly=False)) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                table = self.table_name(conn)
                conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute(
                    f"INSERT INTO {table} (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, raw),
                )
            sync_directory(self.directory.path)
        except (sqlite3.Error, OSError):
            raise CredentialError("Could not commit the credential database.") from None
