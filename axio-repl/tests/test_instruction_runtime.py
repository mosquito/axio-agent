"""The model must see current scoped instructions before dependent file operations."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from axio import Agent, MemoryContextStore, Tool
from axio.blocks import ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, StreamEvent, ToolInputDelta, ToolOutputDelta, ToolUseStart
from axio.messages import Message
from axio.testing import StubTransport, make_text_response
from axio.types import StopReason, Usage
from axio_tools_local.paths import LocalPaths as Workspace
from axio_tools_local.provider import LocalBinding, LocalOptions, LocalToolProvider

pytestmark = pytest.mark.asyncio


def batch(*calls: ToolUseBlock) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for index, call in enumerate(calls):
        events += [ToolUseStart(index, call.id, call.name), ToolInputDelta(index, call.id, json.dumps(call.input))]
    events.append(IterationEnd(0, StopReason.tool_use, Usage(1, 1)))
    return events


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def writer(id: str, path: str, content: str = "written") -> ToolUseBlock:
    return ToolUseBlock(id=id, name="write_file", input={"file_path": path, "content": content})


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return Workspace.discover(tmp_path, tmp_path)


class ObservedTransport(StubTransport):
    def __init__(self, responses: list[list[StreamEvent]], observe: Callable[[int, str], None]) -> None:
        super().__init__(responses)
        self.observe = observe
        self.prompts: list[str] = []

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.observe(self._call_count, system)
        self.prompts.append(system)
        return super().stream(messages, tools, system)


async def configured(workspace: Workspace, transport: StubTransport) -> tuple[Agent, LocalBinding]:
    provider = LocalToolProvider()
    await provider.init(LocalOptions(workspace.cwd, workspace.roots))
    agent = Agent(system="", transport=transport, providers=[provider])
    return agent, provider.bind(agent)


async def test_first_write_and_parallel_read_wait_for_scoped_instructions(workspace: Workspace) -> None:
    root = workspace.cwd
    write(root / "api/AGENTS.md", "API_ONLY_RULE")
    write(root / "ui/AGENTS.md", "UI_ONLY_RULE")
    write(root / "unused/AGENTS.md", "DO_NOT_LOAD_NEIGHBOR")
    write(root / "ui/input.txt", "input")
    output = root / "api/output.txt"
    first = [
        writer("write1", "api/output.txt"),
        ToolUseBlock(id="read1", name="read_file", input={"filename": "ui/input.txt"}),
    ]
    second = [
        writer("write2", "api/output.txt"),
        ToolUseBlock(id="read2", name="read_file", input={"filename": "ui/input.txt"}),
    ]

    def observe(turn: int, system: str) -> None:
        assert "DO_NOT_LOAD_NEIGHBOR" not in system
        if turn == 0:
            assert "API_ONLY_RULE" not in system and "UI_ONLY_RULE" not in system
        if turn == 1:
            assert not output.exists()
            assert "API_ONLY_RULE" in system and "UI_ONLY_RULE" in system
            assert f'scope: "{root / "api"}"' in system
            assert f'scope: "{root / "ui"}"' in system

    transport = ObservedTransport([batch(*first), batch(*second), make_text_response()], observe)
    agent, _ = await configured(workspace, transport)
    ctx = MemoryContextStore()
    await agent.run("go", ctx)
    assert output.read_text() == "written"
    results = [b for msg in await ctx.get_history() for b in msg.content if isinstance(b, ToolResultBlock)]
    assert {b.tool_use_id: b.is_error for b in results} == {
        "write1": True,
        "read1": True,
        "write2": False,
        "read2": False,
    }


async def test_instruction_edit_isolated_and_visible_on_next_model_turn(workspace: Workspace) -> None:
    root = workspace.cwd
    write(root / "AGENTS.md", "OLD_RULE")
    output = root / "output.txt"

    def observe(turn: int, system: str) -> None:
        if turn == 1:
            assert not output.exists()
            assert "NEW_RULE" in system and "OLD_RULE" not in system

    transport = ObservedTransport(
        [
            batch(writer("edit", "AGENTS.md", "NEW_RULE"), writer("unsafe", "output.txt")),
            batch(writer("retry", "output.txt")),
            make_text_response(),
        ],
        observe,
    )
    agent, _ = await configured(workspace, transport)
    ctx = MemoryContextStore()
    await agent.run("go", ctx)
    assert output.read_text() == "written"
    results = {
        b.tool_use_id: b for msg in await ctx.get_history() for b in msg.content if isinstance(b, ToolResultBlock)
    }
    assert not results["edit"].is_error and results["unsafe"].is_error and not results["retry"].is_error


@pytest.mark.parametrize("change", ["edit", "delete"])
async def test_external_change_after_model_decision_defers_batch(workspace: Workspace, change: str) -> None:
    root = workspace.cwd
    source = write(root / "AGENTS.md", "OLD_RULE")

    def observe(turn: int, system: str) -> None:
        if turn == 0:
            assert "OLD_RULE" in system
            source.unlink() if change == "delete" else source.write_text("NEW_RULE")
        if turn == 1:
            assert not (root / "output.txt").exists()
            assert "OLD_RULE" not in system
            assert ("NEW_RULE" in system) is (change == "edit")

    transport = ObservedTransport(
        [batch(writer("first", "output.txt")), batch(writer("retry", "output.txt")), make_text_response()], observe
    )
    agent, _ = await configured(workspace, transport)
    await agent.run("go", MemoryContextStore())
    assert (root / "output.txt").exists()


async def test_search_scopes_all_candidates_before_reading_them(workspace: Workspace) -> None:
    root = workspace.cwd
    write(root / "tree/nested/AGENTS.md", "NESTED_SEARCH_RULE")
    write(root / "tree/nested/data.txt", "needle")
    write(root / "tree/.hidden/AGENTS.md", "HIDDEN_RULE")
    write(root / "other/AGENTS.md", "OTHER_RULE")
    call = ToolUseBlock(id="search", name="search_files", input={"query": "needle", "path": "tree"})
    transport = ObservedTransport([batch(call), batch(call), make_text_response()], lambda turn, system: None)
    agent, _ = await configured(workspace, transport)
    ctx = MemoryContextStore()
    await agent.run("go", ctx)
    assert "NESTED_SEARCH_RULE" not in transport.prompts[0]
    assert "NESTED_SEARCH_RULE" in transport.prompts[1]
    assert "HIDDEN_RULE" not in agent.system and "OTHER_RULE" not in agent.system
    results = [b for msg in await ctx.get_history() for b in msg.content if isinstance(b, ToolResultBlock)]
    assert results[0].is_error and "needle" not in str(results[0].content)
    assert not results[1].is_error and "data.txt:1: needle" in str(results[1].content)


async def test_list_does_not_load_child_directory_instructions(workspace: Workspace) -> None:
    root = workspace.cwd
    write(root / "nested/AGENTS.md", "CHILD_RULE")
    agent, _ = await configured(
        workspace, StubTransport([batch(ToolUseBlock(id="list", name="list_files", input={})), make_text_response()])
    )
    await agent.run("go", MemoryContextStore())
    assert "CHILD_RULE" not in agent.system


async def test_streaming_shell_waits_for_cwd_instructions(workspace: Workspace) -> None:
    root = workspace.cwd
    write(root / "nested/AGENTS.md", "SHELL_CWD_RULE")
    call = ToolUseBlock(id="shell", name="shell", input={"command": "printf hello", "cwd": "nested"})
    agent, _ = await configured(workspace, StubTransport([batch(call), batch(call), make_text_response()]))
    events = [event async for event in agent.run_stream("go", MemoryContextStore())]
    output = [event.delta for event in events if isinstance(event, ToolOutputDelta)]
    assert output == ["hello"]
    assert "SHELL_CWD_RULE" in agent._effective_system()


async def test_symlink_uses_other_workspace_and_detects_retargeting(workspace: Workspace) -> None:
    root = workspace.cwd
    other = root.parent / f"{root.name}-other"
    write(other / "AGENTS.md", "OTHER_ROOT_RULE")
    write(other / "file.txt", "other")
    write(root / "AGENTS.md", "PRIMARY_ROOT_RULE")
    link = root / "alias"
    link.symlink_to(other, target_is_directory=True)
    workspace = Workspace.discover(root, root, [other])
    agent, binding = await configured(workspace, StubTransport([]))
    call = ToolUseBlock(id="read", name="read_file", input={"filename": "alias/file.txt"})
    await agent._prepare_instructions()
    await binding.before_tools([call], 1)
    await agent._prepare_instructions()
    assert f'scope: "{other}"' in agent._effective_system()
    assert "OTHER_ROOT_RULE" in agent._effective_system()
    link.unlink()
    write(root / "new/AGENTS.md", "RETARGETED_RULE")
    link.symlink_to(root / "new", target_is_directory=True)
    await binding.before_tools([call], 2)
    await agent._prepare_instructions()
    assert "RETARGETED_RULE" in agent._effective_system()


async def test_combined_limit_blocks_batch_without_partial_snapshot(workspace: Workspace) -> None:
    root = workspace.cwd
    write(root / "a/AGENTS.md", "a" * 6)
    write(root / "b/AGENTS.md", "b" * 6)
    provider = LocalToolProvider()
    await provider.init(LocalOptions(workspace.cwd, workspace.roots, max_instruction_bytes=10))
    agent = Agent(system="", transport=StubTransport([]), providers=[provider])
    binding = provider.bind(agent)
    await agent._prepare_instructions()
    initial = binding.instruction.snapshot
    results = await agent.dispatch_tools([writer("a", "a/file"), writer("b", "b/file")], 1)
    assert len(results) == 2 and all(result.is_error for result in results)
    assert "size limit" in str(results[0].content)
    assert not (root / "a/file").exists() and not (root / "b/file").exists()
    assert binding.instruction.snapshot == initial


async def test_isolated_explicit_source_refresh_and_instruction_alias(workspace: Workspace) -> None:
    root = workspace.cwd
    write(root / "nested/AGENTS.md", "AUTOMATIC_RULE")
    explicit = write(root / "session.txt", "EXPLICIT_RULE")
    alias = root / "alias.txt"
    alias.symlink_to(explicit)
    provider = LocalToolProvider()
    await provider.init(LocalOptions(root, workspace.roots, isolated=True, instruction_files=(explicit,)))
    agent = Agent(system="", transport=StubTransport([]), providers=[provider])
    binding = provider.bind(agent)
    await agent._prepare_instructions()
    assert not await binding.before_tools([writer("plain", "nested/file")], 1)
    results = await binding.before_tools([writer("alias", "alias.txt"), writer("other", "nested/file")], 1)
    assert [result.tool_use_id for result in results] == ["other"]
    explicit.write_text("CHANGED_EXPLICIT")
    await agent._prepare_instructions()
    text = binding.instruction.snapshot.render()
    assert "CHANGED_EXPLICIT" in text and "EXPLICIT_RULE" not in text and "AUTOMATIC_RULE" not in text
