# Agent Swarm

This guide walks through building an **agent swarm**: a team of role-specialized AI
agents coordinated by an orchestrator, each tackling a different part of a task —
just like a real engineering team.

The full example lives in `examples/agent_swarm/` in the repository.

## What you'll build

```{mermaid}
flowchart TD
    User([User]) -->|task| O[Orchestrator]

    O -->|delegate| PM[Project Manager]
    O -->|delegate| AR[Architect]
    O -->|delegate| BE[Backend Dev]
    O -->|delegate| FE[Frontend Dev]
    O -->|delegate| QA[QA Engineer]
    O -->|delegate| SEC[Security Engineer]
    O -->|delegate| ETL[ETL Engineer]
    O -->|delegate| DES[Designer]
    O -->|delegate| UX[UX Engineer]

    PM  -->|requirements.md| WS[(workspace/)]
    AR  -->|design.md|       WS
    BE  -->|solution.py|     WS
    FE  -->|index.html|      WS
    QA  -->|test_solution.py| WS
    SEC -->|security_review.md| WS
    ETL -->|pipeline.py|     WS
    DES -->|design_spec.md|  WS
    UX  -->|ux_spec.md|      WS

    WS -->|reads artifacts| BE
    WS -->|reads artifacts| QA
    WS -->|reads artifacts| SEC
```

The **orchestrator** receives a task, decides which specialists to involve, and calls
each one in sequence (or in parallel when independent). Specialists communicate through
a shared **workspace directory**: the architect writes `design.md`, the backend dev reads
it before implementing, the QA engineer reads the implementation before writing tests,
and so on.

## Project structure

```
examples/agent_swarm/
├── pyproject.toml       ← standalone workspace package
├── main.py              ← CLI entry point, Rich live display
├── swarm.py             ← Delegate tool, AutoCompactContextStore, run_swarm()
└── roles/
    ├── __init__.py      ← AGENTS registry + ORCHESTRATOR agent
    ├── _common.py       ← OUTPUT_FORMAT appended to every role prompt
    ├── architect.py
    ├── backend_dev.py
    ├── challenger.py
    ├── designer.py
    ├── etl_engineer.py
    ├── frontend_dev.py
    ├── project_manager.py
    ├── qa.py
    ├── security_engineer.py
    └── ux_engineer.py
```

## 1. Defining roles

A role is just an `Agent` pre-configured with `DummyCompletionTransport`.
The dummy transport exists purely as a "not yet wired up" placeholder — calling it
raises an error and logs a warning, so you never accidentally run an unconfigured agent:

```python
# roles/backend_dev.py
from axio.agent import Agent
from axio.transport import DummyCompletionTransport

DESCRIPTION = "Implements server-side logic, data models, APIs, and business logic in Python"

agent = Agent(
    system="""\
You are a senior backend developer. When given a task you:
1. Read workspace/design.md (if it exists) for the blueprint.
2. Implement the required Python code with full type annotations.
3. Save the result to workspace/solution.py.
...""",
    transport=DummyCompletionTransport(),
)
```

That's it. No custom dataclass, no registry entry point. Each file is
self-contained: `DESCRIPTION` feeds the orchestrator's roster; `agent` is the
prototype that gets activated at runtime via `copy()`.

The `roles/__init__.py` builds the registry and the **orchestrator** agent in one place:

```python
from axio.agent import Agent
from axio.transport import DummyCompletionTransport
from . import architect, backend_dev, qa  # etc.

AGENTS: dict[str, tuple[str, Agent]] = {
    "architect":   (architect.DESCRIPTION,   architect.agent),
    "backend_dev": (backend_dev.DESCRIPTION, backend_dev.agent),
    "qa":          (qa.DESCRIPTION,          qa.agent),
    # ...
}

_roster = "\n".join(f"  {name:20s} — {desc}" for name, (desc, _) in AGENTS.items())

ORCHESTRATOR = Agent(
    system=f"""\
You are a tech lead managing a team of specialist agents.
...
Available team members
----------------------
{_roster}

Use the delegate tool to assign work. Pass the workspace path in every delegation.\
""",
    transport=DummyCompletionTransport(),
)
```

The orchestrator is a regular `Agent` — not a special class. Its system prompt is
built once from `AGENTS` so the roster is always up to date.

**Adding a new role** is a single-file change: create `roles/new_role.py`, add it
to `AGENTS` in `__init__.py`. No changes to `swarm.py` or `main.py`.

## 2. Activating a role with `copy()`

`Agent` is a stateless frozen dataclass. `copy()` is just `dataclasses.replace()`.
To run a specialist with a real transport:

```python
description, proto = AGENTS["backend_dev"]
specialist = proto.copy(
    transport=real_transport,
    tools=file_tools,
    max_iterations=25,
)
result = await specialist.run(task, MemoryContextStore())
```

The same applies to the orchestrator — it gets the real transport and the
`delegate` tool at runtime:

```python
orchestrator = ORCHESTRATOR.copy(transport=transport, tools=[delegate])
```

No factory functions, no dependency injection containers. `copy()` is enough.

## 3. The Delegate tool

The `Delegate` tool is a `ToolHandler` subclass defined inside `make_delegate_tool()`
so it closes over `transport`, `workspace`, `on_event`, `role_models`, and `guard_factory`:

```python
def make_delegate_tool(workspace, on_event, transport, role_models,
                       guard_factory=None) -> Tool:
    role_names = list(AGENTS.keys())

    class Delegate(ToolHandler):
        """Delegate a task to a specialist team member."""
        role: Annotated[str, Field(json_schema_extra={"enum": role_names})]
        task: str

        async def __call__(self) -> str:
            _description, proto = AGENTS[self.role]
            role_transport = _transport_for(self.role, transport, role_models)
            specialist = proto.copy(
                transport=role_transport,
                tools=file_tools(str(workspace), role=self.role,
                                 guard_factory=guard_factory),
                max_iterations=25,
            )
            context = AutoCompactContextStore(transport=role_transport)
            stream = specialist.run_stream(
                f"Workspace: {workspace}\n\n{self.task}",
                context,
            )
            parts: list[str] = []
            async for event in stream:
                await on_event(self.role, event)
                if isinstance(event, TextDelta):
                    parts.append(event.delta)
            return "".join(parts)

    guards = (guard_factory("orchestrator", "delegate"),) if guard_factory else ()
    return Tool(name="delegate", description=Delegate.__doc__ or "",
                handler=Delegate, guards=guards)
```

Key points:

- `json_schema_extra={"enum": role_names}` makes the JSON schema expose a strict enum
  so the LLM cannot hallucinate a role name.
- All parallel delegate calls in a single response run concurrently — no artificial cap.
- Each specialist gets a fresh `AutoCompactContextStore` — clean context per delegation.

## 4. Streaming sub-agent output

Normally `AgentStream.get_final_text()` consumes the stream internally. In the swarm
we need to **both** stream events to the display callback **and** collect the final text,
so we iterate manually:

```python
stream = agent.run_stream(task, context)
parts: list[str] = []
async for event in stream:
    await on_event(self.role, event)   # async: renderer holds a lock
    if isinstance(event, TextDelta):
        parts.append(event.delta)
return "".join(parts)
```

`on_event` is async — the renderer holds an `asyncio.Lock` so concurrent delegates
don't interleave their output. It receives `TextDelta`, `ReasoningDelta`, `ToolUseStart`,
`ToolResult`, `IterationEnd`, `SessionEndEvent`, and `Error`.

## 5. Guards for logging and auditing

`PermissionGuard` is not only for access control. Because it receives the fully-parsed
`ToolHandler` instance (all Pydantic fields already validated), it is also the right
place to log, audit, or display tool calls — **before** the tool executes.

```python
class RoleGuard(PermissionGuard):
    """Logs every tool call to the renderer before execution."""

    def __init__(self, role: str, tool_name: str, renderer: SwarmRenderer) -> None:
        self._role = role
        self._tool_name = tool_name
        self._renderer = renderer

    async def check(self, handler: ToolHandler) -> ToolHandler:
        async with self._renderer._lock:       # same lock as on_event — no interleaving
            self._renderer._print_tool_call(self._role, self._tool_name, handler)
        return handler                         # allow — raise GuardError to deny
```

The guard acquires the same `asyncio.Lock` the event callback uses, so output from
concurrent specialists never interleaves.

`SwarmRenderer` exposes a factory method:

```python
def make_guard(self, role: str, tool_name: str) -> RoleGuard:
    return RoleGuard(role=role, tool_name=tool_name, renderer=self)
```

This factory is passed into `run_swarm()` and attached to every tool — file tools,
`delegate`, and `ask_user`:

```python
await run_swarm(
    task=args.task,
    workspace=workspace,
    on_event=renderer.on_event,
    transport=transport,
    role_models=role_models,
    guard_factory=renderer.make_guard,   # ← attaches RoleGuard to every tool
)
```

This removes all `ToolFieldStart` / `ToolFieldDelta` / `ToolFieldEnd` handling from
`SwarmRenderer._handle()`. The event loop only needs to handle `ToolUseStart` (status
bar update) and `ToolResult` (result content display).

## 6. File tools with a workspace default

`Shell` and `RunPython` accept a `cwd` field. The `file_tools()` helper creates local
subclasses with `cwd` defaulting to the workspace so relative paths work out of the box:

```python
def file_tools(workspace: str, role: str = "",
               guard_factory: GuardFactory | None = None) -> list[Tool]:
    class _Shell(Shell):
        cwd: str = workspace

    class _RunPython(RunPython):
        cwd: str = workspace

    def guards(name: str) -> tuple[PermissionGuard, ...]:
        if guard_factory is None:
            return ()
        return (guard_factory(role, name),)

    return [
        Tool(name="read_file",  description=ReadFile.__doc__ or "",  handler=ReadFile,   guards=guards("read_file")),
        Tool(name="write_file", description=WriteFile.__doc__ or "", handler=WriteFile,  guards=guards("write_file")),
        Tool(name="patch_file", description=PatchFile.__doc__ or "", handler=PatchFile,  guards=guards("patch_file")),
        Tool(name="list_files", description=ListFiles.__doc__ or "", handler=ListFiles,  guards=guards("list_files")),
        Tool(name="shell",      description=Shell.__doc__ or "",     handler=_Shell,     guards=guards("shell")),
        Tool(name="run_python", description=RunPython.__doc__ or "", handler=_RunPython, guards=guards("run_python")),
    ]
```

Specialists always receive the workspace path in their task message and use absolute
paths for file I/O, so `cwd` is mainly a safety net for shell commands.

## 7. Rich output

`main.py` maps each role to a Rich style and handles `StreamEvent` types:

```python
ROLE_STYLES = {
    "orchestrator":    "bold white",
    "architect":       "bold cyan",
    "backend_dev":     "bold green",
    "qa":              "bold magenta",
    # ...
}
```

`SwarmRenderer` accumulates `TextDelta` events in a per-role buffer and renders the
complete text as **markdown** when the role finishes speaking (at `ToolUseStart` or
`SessionEndEvent`). This means headers, code blocks, and lists in agent responses
render with full formatting rather than as raw text.

Tool input display is handled by `RoleGuard` (see section 5 above) — the guard prints
all tool fields before execution. `ToolResult` then shows the outcome: status (✓/✗)
and any output. `ReasoningDelta` events (chain-of-thought tokens from reasoning models)
print in dim italic. `IterationEnd` prints a compact token summary after each LLM call:

```python
elif isinstance(event, IterationEnd):
    u = event.usage
    self._print(
        f"[dim]  iter {event.iteration} · {event.stop_reason} "
        f"· ↑{u.input_tokens} ↓{u.output_tokens}[/dim]"
    )
```

A `StatusBar` renderable (passed once to `Live()` at construction) reads renderer state
at Rich's refresh rate (4 fps) without any manual `update()` calls — avoiding flicker
when dozens of events arrive per second.

After the orchestrator finishes, the workspace is rendered as a `rich.tree.Tree`
so you can see all the artifacts the team produced.

## 8. Running it

Install and run:

```bash
cd examples/agent_swarm
uv sync
export NEBIUS_API_KEY=...   # Nebius AI Studio — TokenFactory

# Full software task — exercises architect + backend_dev + qa + security_engineer
uv run python main.py "Build a Python rate limiter library with token-bucket and sliding-window strategies"

# Data task — exercises etl_engineer + backend_dev + qa
uv run python main.py "Write an ETL pipeline that reads CSV, normalises column names, and writes Parquet"

# UI task — exercises ux_engineer + designer + frontend_dev
uv run python main.py "Build a responsive login form with validation"
```

All roles default to `MiniMaxAI/MiniMax-M2.5` via Nebius TokenFactory.
To assign a different model to a specific role, edit `role_models` in `main.py`:

```python
role_models: dict[str, ModelSpec] = {
    "default":           transport.models["MiniMaxAI/MiniMax-M2.5"],
    # Uncomment to use a reasoning model for roles that need it:
    # "orchestrator":    transport.models.search("DeepSeek-R1").first(),
    # "architect":       transport.models.search("DeepSeek-R1").first(),
    # "security_engineer": transport.models.search("DeepSeek-R1").first(),
}
```

`transport.models` is a `ModelRegistry` populated from the Nebius API after
`fetch_models()`. Use `.search("substring").first()` to find models by name, or
index directly with `transport.models["model-id"]`.

After the run, the `workspace/` directory contains everything the team produced:

```
workspace/
├── requirements.md        ← project_manager
├── design.md              ← architect
├── solution.py            ← backend_dev
├── test_solution.py       ← qa
└── security_review.md     ← security_engineer
```

## 9. Extending the team

To add a new role:

1. Create `roles/data_scientist.py` following the same pattern as the other role files:

   ```python
   from axio.agent import Agent
   from axio.transport import DummyCompletionTransport

   DESCRIPTION = "Explores data, builds models, and produces analytical reports"
   PROMPT = """You are a senior data scientist..."""

   agent = Agent(system=PROMPT, transport=DummyCompletionTransport())
   ```

2. Import it in `roles/__init__.py` and add it to `AGENTS`:

   ```python
   from . import data_scientist
   
   AGENTS: dict[str, tuple[str, Agent]] = {
       # ...existing roles...
       "data_scientist": (data_scientist.DESCRIPTION, data_scientist.agent),
   }
   ```

The orchestrator automatically picks up the new role — its system prompt is rebuilt from
`AGENTS` on startup. No changes to `swarm.py` or `main.py` are needed.

## When to use this pattern

Use agent swarms when:

- The task genuinely benefits from different expertise applied sequentially or in parallel
  (design → implement → test → review).
- You want **isolation**: each agent starts with a clean context and cannot accidentally
  carry state from a previous subtask.
- You need **parallel work**: the orchestrator can delegate to all independent agents
  simultaneously — frontend + backend + security all run at the same time if the LLM
  issues multiple tool calls in one response.

For simpler cases — a single agent that can call tools to break down its own work —
the built-in `subagent` tool from `axio-tui` is sufficient.
The swarm pattern adds structure at the cost of more orchestration prompting and
more LLM calls.

```{seealso}
- {doc}`writing-tools` — how to build custom ToolHandlers
- {doc}`testing` — how to unit-test agents and tools with StubTransport
- {ref}`concepts/agent` — the agent loop in detail
```
