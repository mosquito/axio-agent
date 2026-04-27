"""Agent swarm: orchestrator + specialist team powered by Axio.

Specialist roles are loaded from TOML files at runtime once the shared toolbox
is ready — each role's ``tools`` list is resolved against the toolbox.  The
Orchestrator is the only agent declared in Python (in roles/__init__.py).

Transport setup lives in main.py — swarm.py only uses what it is given.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

import aiosqlite
from axio.agent import Agent
from axio.agent_loader import load_agents
from axio.compaction import AutoCompactStore
from axio.context import MemoryContextStore
from axio.events import StreamEvent, TextDelta
from axio.models import ModelSpec
from axio.permission import PermissionGuard
from axio.tool import Tool, ToolHandler
from axio.transport import CompletionTransport, DummyCompletionTransport
from axio_tools_local.list_files import ListFiles
from axio_tools_local.patch_file import PatchFile
from axio_tools_local.read_file import ReadFile
from axio_tools_local.run_python import RunPython
from axio_tools_local.shell import Shell
from axio_tools_local.write_file import WriteFile
from pydantic import Field

from .ask_user import make_ask_user_tool
from .roles import ROLE_NAMES, ROLES_DIR, make_orchestrator
from .todo import DDL as TODO_DDL
from .todo import make_todo_tool

OnEventCallback = Callable[[str, StreamEvent], Awaitable[None]]
GuardFactory = Callable[[str, str], PermissionGuard]

# ---------------------------------------------------------------------------
# Read-only analyst prototype
# ---------------------------------------------------------------------------

ANALYST = Agent(
    system="""\
You are a read-only analyst. Your only job is to read files in the workspace and
produce a concise, well-structured report answering the question you are given.
You must not create, modify, or delete any files. Never use write_file or patch_file.
Return your findings as plain text — the caller will use them directly.""",
    transport=DummyCompletionTransport(),
)


# ---------------------------------------------------------------------------
# Transport helper
# ---------------------------------------------------------------------------


def transport_for(
    role: str,
    base: CompletionTransport,
    role_models: dict[str, ModelSpec],
) -> CompletionTransport:
    """Return a copy of *base* with the model for *role* (falls back to "default")."""
    model = role_models.get(role) or role_models["default"]
    new_transport = copy.copy(base)
    new_transport.model = model  # type: ignore[attr-defined]
    return new_transport


# ---------------------------------------------------------------------------
# File tools helper
# ---------------------------------------------------------------------------


def file_tools(
    role: str = "",
    guard_factory: GuardFactory | None = None,
) -> list[Tool[Any]]:
    """Standard file + shell tools for specialist agents."""

    def guards(name: str) -> tuple[PermissionGuard, ...]:
        if guard_factory is None:
            return ()
        return (guard_factory(role, name),)

    return [
        Tool(name="read_file", description=ReadFile.__doc__ or "", handler=ReadFile, guards=guards("read_file")),
        Tool(name="write_file", description=WriteFile.__doc__ or "", handler=WriteFile, guards=guards("write_file")),
        Tool(name="patch_file", description=PatchFile.__doc__ or "", handler=PatchFile, guards=guards("patch_file")),
        Tool(name="list_files", description=ListFiles.__doc__ or "", handler=ListFiles, guards=guards("list_files")),
        Tool(name="shell", description=Shell.__doc__ or "", handler=Shell, guards=guards("shell")),
        Tool(name="run_python", description=RunPython.__doc__ or "", handler=RunPython, guards=guards("run_python")),
    ]


# ---------------------------------------------------------------------------
# Analyze tool — top-level handler, context carries all runtime deps
# ---------------------------------------------------------------------------


class AnalyzeContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    guard_factory: GuardFactory | None
    counter: list[int]  # [0] holds the mutable call count


class Analyze(ToolHandler[AnalyzeContext]):
    """Spawn a read-only analyst subagent to investigate a question and return a report.
    The analyst can only read files — it cannot modify anything.
    Safe to call many times in parallel; use one per file or question."""

    task: Annotated[str, Field(description="Question or analysis task for the analyst")]

    async def __call__(self, context: AnalyzeContext) -> str:
        context["counter"][0] += 1
        n = context["counter"][0]
        agent_id = f"analyst#{n}:{self.task[:40]}"

        analyst_transport = transport_for("analyst", context["transport"], context["role_models"])

        def a_guards(name: str) -> tuple[PermissionGuard, ...]:
            gf = context["guard_factory"]
            return (gf("analyst", name),) if gf else ()

        analyst = ANALYST.copy(
            transport=analyst_transport,
            tools=[
                Tool(
                    name="list_files",
                    description=ListFiles.__doc__ or "",
                    handler=ListFiles,
                    guards=a_guards("list_files"),
                ),
                Tool(
                    name="read_file",
                    description=ReadFile.__doc__ or "",
                    handler=ReadFile,
                    guards=a_guards("read_file"),
                ),
            ],
            max_iterations=10,
        )
        stream = analyst.run_stream(f"Workspace: {context['workspace']}\n\n{self.task}", MemoryContextStore())
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)


def make_analyze_tool(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    caller_role: str = "specialist",
    guard_factory: GuardFactory | None = None,
) -> Tool[AnalyzeContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory(caller_role, "analyze"),) if guard_factory else ()
    return Tool(
        name="analyze",
        description=Analyze.__doc__ or "",
        handler=Analyze,
        context=AnalyzeContext(
            workspace=workspace,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            guard_factory=guard_factory,
            counter=[0],
        ),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Delegate tool — top-level handler, context carries all runtime deps
# ---------------------------------------------------------------------------


class DelegateContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    roles: dict[str, tuple[str, Agent]]  # runtime-loaded roles with tools
    guard_factory: GuardFactory | None
    counters: dict[str, int]


class Delegate(ToolHandler[DelegateContext]):
    """Delegate a task to a specialist team member.
    The specialist reads the workspace, does the work, and writes output back.
    Returns the specialist's final response text."""

    role: Annotated[
        str,
        Field(
            description=f"Which specialist to delegate to. One of: {', '.join(ROLE_NAMES)}",
            json_schema_extra={"enum": ROLE_NAMES},
        ),
    ]
    topic: Annotated[
        str,
        Field(description="Short label for this task, e.g. 'auth middleware'. Shown in the status bar."),
    ]
    task: Annotated[str, Field(description="Specific instructions for the specialist")]

    async def __call__(self, context: DelegateContext) -> str:
        context["counters"][self.role] = context["counters"].get(self.role, 0) + 1
        n = context["counters"][self.role]
        base_id = self.role if n == 1 else f"{self.role}#{n}"
        agent_id = f"{base_id}:{self.topic}" if self.topic else base_id

        description, proto = context["roles"][self.role]
        role_transport = transport_for(self.role, context["transport"], context["role_models"])
        specialist = proto.copy(transport=role_transport)
        specialist_ctx = AutoCompactStore(MemoryContextStore(), role_transport, keep_recent=6)
        stream = specialist.run_stream(
            f"Workspace: {context['workspace']}\n\n{self.task}",
            specialist_ctx,
        )
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)


def make_delegate_tool(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    roles: dict[str, tuple[str, Agent]],
    guard_factory: GuardFactory | None = None,
) -> Tool[DelegateContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("orchestrator", "delegate"),) if guard_factory else ()
    return Tool(
        name="delegate",
        description=Delegate.__doc__ or "",
        handler=Delegate,
        context=DelegateContext(
            workspace=workspace,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            roles=roles,
            guard_factory=guard_factory,
            counters={},
        ),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Toolbox — shared tools for all specialist agents
# ---------------------------------------------------------------------------


def build_toolbox(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    guard_factory: GuardFactory | None = None,
) -> dict[str, Tool[Any]]:
    """Build the shared toolbox injected into specialist agents via load_agents().

    The toolbox contains all tools listed in the specialist TOML files.  The
    Orchestrator receives its own separate tool set (delegate + read tools +
    ask_user + todo + analyze).
    """

    def guards(role: str, name: str) -> tuple[PermissionGuard, ...]:
        return (guard_factory(role, name),) if guard_factory else ()

    return {
        "read_file": Tool(
            name="read_file",
            description=ReadFile.__doc__ or "",
            handler=ReadFile,
            guards=guards("specialist", "read_file"),
        ),
        "write_file": Tool(
            name="write_file",
            description=WriteFile.__doc__ or "",
            handler=WriteFile,
            guards=guards("specialist", "write_file"),
        ),
        "patch_file": Tool(
            name="patch_file",
            description=PatchFile.__doc__ or "",
            handler=PatchFile,
            guards=guards("specialist", "patch_file"),
        ),
        "list_files": Tool(
            name="list_files",
            description=ListFiles.__doc__ or "",
            handler=ListFiles,
            guards=guards("specialist", "list_files"),
        ),
        "shell": Tool(
            name="shell",
            description=Shell.__doc__ or "",
            handler=Shell,
            guards=guards("specialist", "shell"),
        ),
        "run_python": Tool(
            name="run_python",
            description=RunPython.__doc__ or "",
            handler=RunPython,
            guards=guards("specialist", "run_python"),
        ),
        "analyze": make_analyze_tool(
            workspace,
            on_event,
            transport,
            role_models,
            caller_role="specialist",
            guard_factory=guard_factory,
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_swarm(
    task: str,
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    guard_factory: GuardFactory | None = None,
    prompt_fn: Callable[[], str] | None = None,
) -> str:
    """Run the agent swarm on *task*.

    Args:
        task:          What to build.
        workspace:     Directory where agents read and write files.
        on_event:      Callback for every StreamEvent emitted by any agent.
        transport:     Base transport — shared session, per-role model applied via copy().
        role_models:   Maps role names (and "default") to ModelSpec.
                       Every role not listed falls back to role_models["default"].
        guard_factory: Optional ``(role, tool_name) -> PermissionGuard`` factory.
        prompt_fn:     Optional callable ``() -> str`` used by ``ask_user`` to read input.
    """
    assert "default" in role_models, "role_models must contain a 'default' key"

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)

    def og(name: str) -> tuple[PermissionGuard, ...]:
        return (guard_factory("orchestrator", name),) if guard_factory else ()

    todo_path = workspace / ".axio-swarm" / "todos.db"
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(todo_path) as todo_db:
        await todo_db.execute(TODO_DDL)
        await todo_db.commit()

        # Build shared toolbox and load specialist roles with tools injected.
        toolbox = build_toolbox(workspace, on_event, transport, role_models, guard_factory)
        roles = load_agents(ROLES_DIR, toolbox=toolbox)

        orch_analyze = make_analyze_tool(
            workspace,
            on_event,
            transport,
            role_models,
            caller_role="orchestrator",
            guard_factory=guard_factory,
        )
        delegate = make_delegate_tool(
            workspace,
            on_event,
            transport,
            role_models,
            roles=roles,
            guard_factory=guard_factory,
        )
        todo_tool = make_todo_tool(todo_db, guards=og("todo"))
        ask_user_tool = make_ask_user_tool(prompt_fn=prompt_fn, guards=og("ask_user"))

        orch_read_tools = [
            Tool(name="list_files", description=ListFiles.__doc__ or "", handler=ListFiles, guards=og("list_files")),
            Tool(name="read_file", description=ReadFile.__doc__ or "", handler=ReadFile, guards=og("read_file")),
        ]

        orch_transport = transport_for("orchestrator", transport, role_models)
        roster = "\n".join(f"  {name:20s} — {desc}" for name, (desc, _) in roles.items())
        orchestrator = make_orchestrator(roster).copy(
            transport=orch_transport,
            tools=[delegate, ask_user_tool, todo_tool, orch_analyze, *orch_read_tools],
        )
        orch_ctx = AutoCompactStore(MemoryContextStore(), orch_transport, keep_recent=10)

        stream = orchestrator.run_stream(task, orch_ctx)
        parts: list[str] = []
        async for event in stream:
            await on_event("orchestrator", event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
    return "".join(parts)
