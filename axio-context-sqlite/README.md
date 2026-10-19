# axio-context-sqlite

[![PyPI](https://img.shields.io/pypi/v/axio-context-sqlite)](https://pypi.org/project/axio-context-sqlite/)
[![Python](https://img.shields.io/pypi/pyversions/axio-context-sqlite)](https://pypi.org/project/axio-context-sqlite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/mosquito/axio-agent/blob/main/LICENSE)

SQLite-backed persistent context store for [axio](https://github.com/mosquito/axio-agent).

## Installation

```bash
pip install axio-context-sqlite
```

## Usage

### `connect(db_path)`

Open (or create) a SQLite database at `db_path` and initialise the schema.
Returns an `aiosqlite.Connection`. The caller is responsible for closing it.

```python
async def main():
    conn = await connect("~/.axio/chat.db")
    # ... use the connection ...
    await conn.close()
```

The helper enables WAL journal mode and a 5-second busy timeout so concurrent
readers and a single writer can safely share the same database file.

### `SQLiteContextStore(conn, session_id, project=None, db_name="axio_context")`

Create a context store bound to one session.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conn` | `aiosqlite.Connection` | - | Open database connection (from `connect()`) |
| `session_id` | `str` | - | Unique identifier for this conversation session |
| `project` | `str \| None` | `str(Path.cwd().resolve())` | Logical project scope used to group and list sessions. Defaults to the current working directory. |
| `db_name` | `str` | `"axio_context"` | Retained for compatibility; table names use the fixed `axio_context_` prefix. |

Open a connection with `connect()`, then create a `SQLiteContextStore` bound to a
session. The caller owns the connection and is responsible for closing it.

<!-- name: test_readme_usage -->
```python
import asyncio
import tempfile
import pathlib
from axio_context_sqlite import connect, SQLiteContextStore
from axio.messages import Message
from axio.blocks import TextBlock


async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        store = SQLiteContextStore(conn, session_id="my-session")
        await store.append(Message(role="user", content=[TextBlock(text="Hello")]))
        history = await store.get_history()
        assert len(history) == 1
    finally:
        await conn.close()


asyncio.run(main())
```

`SQLiteContextStore` implements the `axio.context.ContextStore` ABC and persists
conversation history across process restarts. Multiple sessions can coexist in
the same database file, isolated by `session_id` and `project`.

### Stable message identity and archive consumers

`append()` still adds a new message each time. For externally identified events,
use `upsert(message, message_id=..., status=..., metadata=...)`. The key is scoped
to a session and project. An identical snapshot creates neither another message
nor another journal entry. An update keeps the message's position and creation
time, increments `revision`, and replaces status/metadata. Metadata must be a
JSON object; key ordering does not affect idempotency. Non-finite numbers and
non-JSON values are rejected before writing. Status is an application-defined
nonempty string, defaulting to `"completed"`.

`get_records()` returns `ContextRecord` snapshots with the stable ID, message,
position, status, metadata, revision and creation time. `get_history()` retains
its original `list[Message]` interface. `expected_revision` enables compare and
swap: zero requires insertion, and a read revision requires that version to
still be current. A mismatch raises `ContextConflictError`. Without it, different
snapshots use last-writer-wins semantics. Retrying the same snapshot after an
uncertain commit remains idempotent.

Create one `SQLiteContextDatabase(conn)` client in your application composition
and obtain session stores with `database.store(session_id, project)`. It owns
transaction coordination, not the connection's lifetime. Independently created
clients and legacy `SQLiteContextStore(conn, ...)` constructors on the same
connection automatically share the writer lock. Forked stores share it too.
Use one event loop per connection; separate connections coordinate through
SQLite's write lock and the configured busy timeout.

`database.changes(project, after=0, limit=100)` returns bounded `ContextChange`
pages ordered by a persistent sequence (`limit` is between 1 and 1000). Each
invalidation contains the project/session/message key and its **current** record,
or `None` when deleted. A record can be newer than the invalidation. The journal
supports rebuilding external indexes; it is not a historical snapshot log.

The following example persists final transcript text, consumes it idempotently,
and observes deletion without keeping a second copy of the deleted text:

<!--
name: test_archive_identity_and_journal
-->
```python
import asyncio
import tempfile
from pathlib import Path

from axio import Message, TextBlock
from axio_context_sqlite import SQLiteContextDatabase, connect


async def archive_example():
    with tempfile.TemporaryDirectory() as directory:
        conn = await connect(Path(directory) / "history.db")
        try:
            database = SQLiteContextDatabase(conn)
            store = database.store("conversation-1", "home")
            text = Message(role="user", content=[TextBlock(text="Эй, Ричи!")])
            first = await store.upsert(text, message_id="input-item-1", metadata={"source": "asr"})
            duplicate = await store.upsert(text, message_id="input-item-1", metadata={"source": "asr"})
            assert duplicate == first and len(await store.get_history()) == 1

            # Replace this dictionary with an idempotent persistent search index.
            index = {}
            page = await database.changes("home")
            assert len(page) == 1
            for change in page:
                key = (change.project, change.session_id, change.message_id)
                index[key] = change.record
            # In production, persist the cursor only after committing the sink.
            cursor = page[-1].sequence
            assert await database.changes("home", cursor) == []

            await store.clear()
            deleted = await database.changes("home", cursor)
            assert len(deleted) == 1 and deleted[0].record is None
            key = (deleted[0].project, deleted[0].session_id, deleted[0].message_id)
            index.pop(key, None)
            assert not index
        finally:
            await conn.close()


asyncio.run(archive_example())
```

Database triggers record invalidations in the same transaction as the source
write. Apply each current record or deletion idempotently, commit your index,
then persist the page's last sequence. A crash before the checkpoint replays
the page; a crash after source commit cannot lose its invalidation. Sequence
gaps are normal across projects. Empty pages do not advance the cursor. The
journal stores identifiers only, so clearing history leaves deletion keys
without retaining deleted message content or metadata in the journal.
The journal currently retains these small invalidations indefinitely; consumers
must persist their own checkpoints. New indexes can start at sequence zero.

`connect()` migrates existing databases additively and transactionally, backfills
stable `legacy:<row-id>` IDs and one initial invalidation per existing message,
and leaves the application's `PRAGMA user_version` alone. The `legacy:` prefix
is reserved for migrated/legacy-inserted messages. Existing records under that
prefix can be updated. Reopening does not duplicate the migration journal.
The original insert shape remains writable; compression and old raw JSON remain
readable. Fork preserves message identity/metadata/revision inside its new session;
clear removes only the selected session/project and resets its token counts.

Managed writes use `BEGIN IMMEDIATE`, commit on success and roll back on errors
or cancellation. Cancellation racing a commit can still leave a committed write;
retry by stable ID. Do not run raw SQL concurrently with managed operations or
nest managed calls inside `database.transaction()`. That context is for raw SQL
extensions on the same initialized connection. An already-open unmanaged
transaction is rejected without committing or rolling it back. The caller must
finish all operations before closing the borrowed connection; `store.close()`
continues to leave it open.

#### Storage and compression

Message content is stored as serialized JSON. Payloads larger than 512 bytes
are automatically compressed with gzip (compresslevel 6) and stored
base64-encoded with a `gzip:` prefix. Smaller payloads are stored as-is with a
`plain:` prefix. Decompression happens transparently on read - callers never
see the encoded form.

#### SQLite performance settings

Every connection opened by `connect()` is configured with:

- `PRAGMA journal_mode=WAL` - enables concurrent readers alongside one writer
- `PRAGMA busy_timeout=5000` - waits up to 5 seconds before raising a lock error
- `PRAGMA synchronous=NORMAL` - balances durability and write throughput

### Agent integration

<!--
name: test_readme_agent
```python
from axio.testing import StubTransport, make_text_response

transport = StubTransport([make_text_response("Hi!")])
```
-->
<!-- name: test_readme_agent -->
```python
import asyncio
import tempfile
import pathlib
from axio.agent import Agent
from axio_context_sqlite import connect, SQLiteContextStore


async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        ctx = SQLiteContextStore(conn, session_id="main")
        agent = Agent(system="You are helpful.", tools=[], transport=transport)
        result = await agent.run("Hello!", ctx)
        assert result == "Hi!"
    finally:
        await conn.close()


asyncio.run(main())
```

### Listing sessions

<!-- name: test_readme_list_sessions -->
```python
import asyncio
import tempfile
import pathlib
from axio_context_sqlite import connect, SQLiteContextStore
from axio.messages import Message
from axio.blocks import TextBlock


async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        store = SQLiteContextStore(conn, session_id="main", project="/myproject")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        sessions = await store.list_sessions()
        for s in sessions:
            print(s.session_id, s.preview, s.message_count)
        assert len(sessions) == 1
    finally:
        await conn.close()


asyncio.run(main())
```

### Token accounting

`add_context_tokens(input_tokens, output_tokens)` atomically increments the
stored token counts for the current session and project using a SQL UPSERT
(`INSERT ... ON CONFLICT DO UPDATE SET ... = ... + excluded....`). This is safe
to call concurrently from multiple coroutines without an application-level lock.

`set_context_tokens()` replaces the counts unconditionally, and
`get_context_tokens()` returns a `(input_tokens, output_tokens)` tuple.

### Forking

`fork()` copies the current session's messages into a new session - useful for
branching conversations without affecting the original:

<!-- name: test_readme_fork -->
```python
import asyncio
import tempfile
import pathlib
from axio_context_sqlite import connect, SQLiteContextStore
from axio.messages import Message
from axio.blocks import TextBlock


async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        store = SQLiteContextStore(conn, session_id="main")
        await store.append(Message(role="user", content=[TextBlock(text="original")]))
        branch = await store.fork()
        assert branch.session_id != store.session_id
        assert len(await branch.get_history()) == 1
    finally:
        await conn.close()


asyncio.run(main())
```

## License

MIT
