# Gas Town

This guide walks through the **Gas Town** example: a multi-agent convoy system
modelled on Steve Yegge's [Gas Town](https://github.com/gastownhall/gastown) methodology.

The full example lives in `examples/gas_town/` in the repository.
For a deep dive into the methodology itself, see {doc}`../gas-town-rules`.

## What is Gas Town?

Gas Town is an opinionated orchestration model with a few core ideas:

- **Beads** are the atomic unit of work — a Git-backed (here SQLite-backed) issue
  with id, title, status, assignee, and notes.
- **Convoys** are work orders: a named collection of beads representing one feature
  or task.
- **GUPP** (Gastown Universal Propulsion Principle): *if you find work assigned to
  you, you run it immediately* — no announcement, no waiting for approval.
- **Roles are fixed**: Mayor, Polecat, Witness, Refinery, Crew. Each has a strict
  contract. Polecats are ephemeral (one bead → done means gone); Crew are long-lived.

## Architecture

```{mermaid}
flowchart TD
    User([Overseer / human]) -->|task| M[Mayor]

    M -->|spawn_polecat x N| PC1[Polecat #1]
    M -->|spawn_polecat x N| PC2[Polecat #2]
    M -->|spawn_polecat x N| PC3[Polecat #N]
    M -->|spawn_witness| W[Witness]
    M -->|spawn_refinery| R[Refinery]
    M -->|spawn_crew| C[Crew]

    PC1 -->|closes bead 1| DB[(workspace/.gas-town/beads.db)]
    PC2 -->|closes bead 2| DB
    PC3 -->|closes bead N| DB
    W   -->|reads bead store| DB
    R   -->|notes results| DB
    M   -->|tracks convoy| DB
```

The **Mayor** is the only agent the user interacts with. It decomposes the task
into beads, spawns polecats in parallel to work them, optionally spawns Witness for
health checks, and finally spawns Refinery to integrate the result.

## Project structure

```
examples/gas_town/
├── pyproject.toml       ← standalone package, adds aiosqlite + axio-tools-local
├── __main__.py          ← CLI entry point (axio-gastown), Rich live display
├── beads.py             ← SQLite bead store, ContextVar[Path], BeadTool
├── swarm.py             ← spawn tools, AutoCompactContextStore, run_gastown()
└── roles/
    ├── __init__.py      ← MAYOR, POLECAT, WITNESS, REFINERY, CREW prototypes
    ├── _common.py       ← GUPP_NOTE, WORKSPACE_NOTE, OUTPUT_FORMAT
    ├── mayor.py
    ├── polecat.py
    ├── witness.py
    ├── refinery.py
    └── crew.py
```

## 1. Beads — the data plane

Beads are stored in `workspace/.gas-town/beads.db` (SQLite).
A single `ContextVar[Path]` makes the database available to every agent without
explicit passing — identical to how `asyncio` copies context per task.

```python
# beads.py
store: ContextVar[Path] = ContextVar("bead_store_path")

async def init_store(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS beads (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                title    TEXT NOT NULL,
                status   TEXT NOT NULL DEFAULT 'open',
                assignee TEXT NOT NULL DEFAULT '',
                notes    TEXT NOT NULL DEFAULT ''
            )
        """)
        await db.commit()
```

`BeadTool` is a `ToolHandler` with five actions — `list`, `create`, `update`,
`close`, `note` — that all read `store.get()` to find the database path:

```python
class BeadTool(ToolHandler):
    """Manage the shared bead store (convoy issue tracker). Data is persisted to SQLite."""
    action: Literal["list", "create", "update", "close", "note"]
    id: int = 0
    title: str = ""
    status: BStatus = "open"
    assignee: str = ""
    notes: str = ""

    async def __call__(self) -> str:
        path = store.get()
        async with aiosqlite.connect(path) as db:
            ...
```

Every agent in the convoy gets the same `BEAD_TOOL` instance. Because `store.get()`
resolves per-run via ContextVar, multiple concurrent runs are safely isolated.

The `.gas-town/` directory is reserved for internal orchestration data.
All role prompts explicitly instruct agents **not** to read or write anything
inside it.

## 2. Roles

Each role is an `Agent` prototype with `DummyCompletionTransport`.
Activate a role by calling `proto.copy(transport=real_transport, tools=[...])`.

### Mayor — chief-of-staff

The Mayor is the agent you talk to. It translates a task into a convoy:

1. Creates a `[CONVOY]` bead as the work-order unit.
2. Decomposes the task into small child beads (one per component).
3. Spawns polecats in parallel — one per bead, all in one response.
4. Optionally spawns Witness for a mid-convoy health check.
5. After all polecats finish, spawns Refinery to integrate the work.
6. Closes the convoy bead and reports to the user.

The Mayor never writes code itself. It only spawns workers.

### Polecat — ephemeral worker

A polecat has exactly one job: complete its assigned bead, then close it.

```
GUPP: if you find work, YOU RUN IT.
No announcement. No waiting. No idle state.
```

Lifecycle:

1. Receives bead assignment in its first message.
2. Updates the bead to `in_progress`.
3. Does the work (file tools, shell, run_python, analyze).
4. Closes the bead: `bead(action='close', id=<id>)`.
5. Session ends — done means gone.

If a polecat discovers unrelated work, it creates a new bead and continues.
It never fixes things outside its assigned bead.

### Witness — per-rig monitor

The Witness is a read-only monitor. It checks the bead store, reviews workspace
quality via `analyze`, and produces a health report. It **does not write code**
and has no write tools.

Useful for long convoys (5+ polecats) as a mid-convoy checkpoint.

### Refinery — merge queue processor

The Refinery is the integration engineer. It is **not** a passive reviewer —
it actively integrates polecat work:

- Verifies that all pieces fit together (imports resolve, interfaces match).
- Runs tests and the linter.
- **Fixes integration issues** — broken imports, mismatched signatures, conflicts.
- Escalates to Mayor only when a fix requires re-doing a full bead.

The Refinery has full write tools. "No work can be lost" is its core rule.

### Crew — long-lived agents

Crew members are the agents you interact with for sustained, back-and-forth work —
design sessions, complex investigations, exploratory coding. Unlike polecats, they
are not ephemeral and not managed by the Witness.

## 3. Spawn tools

`swarm.py` creates the Mayor's spawn tools as closures over `workspace`,
`transport`, `on_event`, and `role_models`:

```python
def make_spawn_polecat_tool(workspace, on_event, transport, role_models,
                             analyze_tool, guard_factory=None) -> Tool:
    polecat_counters: dict[int, int] = {}

    class SpawnPolecat(ToolHandler):
        """Spawn a polecat worker to complete a specific bead."""
        bead_id: int
        topic: str   # shown in the status bar as "Polecat [auth middleware]"

        async def __call__(self) -> str:
            path = _bead_store.get()
            row = await _get_bead(path, self.bead_id)
            if row is None:
                return f"Bead {self.bead_id} not found"
            bead_id, title, *_ = row
            await _mark_in_progress(path, bead_id)

            n = polecat_counters.setdefault(self.bead_id, 0) + 1
            polecat_counters[self.bead_id] = n
            agent_id = f"polecat#{n}:{self.topic}"

            polecat = POLECAT.copy(
                transport=transport_for("polecat", transport, role_models),
                tools=[*file_tools(workspace), bead_tool, analyze_tool],
                max_iterations=25,
            )
            stream = polecat.run_stream(
                f"Workspace: {workspace}\n\nBead [{bead_id}]: {title}\n\n"
                f"Close it with bead(action='close', id={bead_id}).",
                AutoCompactContextStore(...),
            )
            parts = []
            async for event in stream:
                await on_event(agent_id, event)
                if isinstance(event, TextDelta):
                    parts.append(event.delta)
            return "".join(parts)
    ...
```

Key points:

- `spawn_polecat` is called multiple times in one Mayor response — they all run
  concurrently because Axio dispatches tool calls with `asyncio.gather()`.
- The `topic` field populates the status bar: **Polecat [auth middleware]**,
  **Polecat [data models]**, etc., making parallel polecats identifiable at a glance.
- `_mark_in_progress` updates the SQLite row before the polecat starts, so Witness
  sees accurate status if it runs mid-convoy.

## 4. GUPP in practice

GUPP is enforced through the system prompt. Every role that can receive a task
includes `GUPP_NOTE` from `_common.py`:

```python
GUPP_NOTE = """
The Propulsion Principle (GUPP)
--------------------------------
If you find assigned work, YOU RUN IT. No announcement, no confirmation, no waiting.
The assignment IS the authorisation. Gas Town is a steam engine — you are a piston.

Failure mode to avoid:
  Agent receives assignment → announces itself → waits for "ok go"
  Human is AFK → work sits idle → the whole convoy stalls.

When assigned, your ONLY next action is to start working.
"""
```

The polecat prompt reinforces this with "The Idle Polecat Heresy":

```
There is no step 5. There is no "wait for approval". There is no idle state.
```

And an explicit close instruction:

```
If your assigned bead has nothing to implement:
- Note the reason: bead(action='note', id=<id>, notes='no-changes: <reason>')
- Close the bead:  bead(action='close', id=<id>)
Never leave without closing your bead.
```

This prevents the most common failure mode: a polecat that finishes work, says
"done!", and then sits idle waiting for acknowledgement — blocking the convoy.

## 5. The bead store ContextVar

`run_gastown` initialises a fresh database for each run and sets the ContextVar:

```python
async def run_gastown(task, workspace, on_event, transport,
                      role_models, guard_factory=None) -> str:
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    db_path = workspace / ".gas-town" / "beads.db"
    await _init_store(db_path)
    bead_token = _bead_store.set(db_path)
    try:
        stream = mayor.run_stream(f"Workspace: {workspace}\n\n{task}", context)
        ...
    finally:
        _bead_store.reset(bead_token)
```

Because `asyncio` copies the context per task, every polecat running concurrently
inherits the same `db_path` without it being passed explicitly. The `finally` block
resets the ContextVar so a second call to `run_gastown` in the same process starts
clean.

The database persists after the run at `workspace/.gas-town/beads.db` — inspect it
with any SQLite tool to see the full convoy history:

```bash
sqlite3 workspace/.gas-town/beads.db "SELECT id, title, status, assignee FROM beads"
```

## 6. Running it

```bash
cd examples
export NEBIUS_API_KEY=...

# Run a convoy
uv run python -m gas_town "Build a Python rate limiter with token-bucket and sliding-window"

# Custom workspace
uv run python -m gas_town --workspace /tmp/my_project "Design a REST API for a blog"
```

Or via the installed console script (from the `examples/` directory):

```bash
axio-gastown "Write an async task queue with priority levels"
```

Default model assignments in `__main__.py`:

```python
role_models = {
    "default":  transport.models["MiniMaxAI/MiniMax-M2.5"],
    "mayor":    transport.models["Qwen/Qwen3-235B-A22B-Instruct-2507"],
    "polecat":  transport.models["Qwen/Qwen3.5-397B-A17B"],
    "witness":  transport.models["openai/gpt-oss-120b"],
    "refinery": transport.models["openai/gpt-oss-120b"],
    "analyst":  transport.models["deepseek-ai/DeepSeek-V3.2"],
}
```

After a run the workspace contains the produced artifacts plus the bead history:

```
workspace/
├── .gas-town/
│   └── beads.db              ← full convoy history (SQLite)
├── reports/
│   └── mayor_analysis.md     ← Mayor's domain analysis
├── solution.py               ← polecat output
├── tests/
│   └── test_solution.py      ← polecat output
└── ...
```

## Gas Town vs Agent Swarm

Both examples implement multi-agent coordination on top of Axio.
The key differences:

| | Agent Swarm | Gas Town |
|---|---|---|
| **Task tracking** | In-memory todo list | SQLite bead store (persists) |
| **Delegation** | `delegate(role, task)` — any role | `spawn_polecat(bead_id)` — per-bead |
| **Worker lifecycle** | Specialist returns result to orchestrator | Polecat closes its bead and disappears |
| **Oversight** | None | Witness monitors; Refinery integrates |
| **Crew** | No equivalent | Long-lived human-facing agents |
| **Work granularity** | One big task per specialist | One small bead per polecat |
| **Propulsion** | Orchestrator drives everything | GUPP: agent drives itself |
| **Workspace state** | Implicit (files only) | Explicit (bead store + files) |

Use **agent_swarm** when you want a straightforward team of specialists with minimal
overhead. Use **gas_town** when you want explicit work tracking, parallel polecat
swarms, integration review, and a workflow model that survives restarts.

```{seealso}
- {doc}`../gas-town-rules` — full Gas Town methodology reference
- {doc}`agent-swarm` — the simpler team-of-specialists pattern
- {doc}`writing-tools` — how to build custom ToolHandlers
```
