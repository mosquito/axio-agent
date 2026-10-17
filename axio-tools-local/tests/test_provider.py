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
from axio.plugins import ToolProvider, collect_tools
from axio.testing import StubTransport, make_text_response
from axio.tool_access import Access
from axio.types import StopReason, Usage

from axio_tools_local.provider import LocalOptions, LocalToolProvider


def batch(*calls: ToolUseBlock) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for index, call in enumerate(calls):
        events += [ToolUseStart(index, call.id, call.name), ToolInputDelta(index, call.id, json.dumps(call.input))]
    return [*events, IterationEnd(0, StopReason.tool_use, Usage(1, 1))]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def call(name: str, **arguments: Any) -> ToolUseBlock:
    return ToolUseBlock(id=name, name=name, input=arguments)


class ObservedTransport(StubTransport):
    def __init__(self, responses: list[list[StreamEvent]], observe: Callable[[int, str], None] = lambda i, s: None):
        super().__init__(responses)
        self.observe = observe
        self.prompts: list[str] = []
        self.tools: list[list[Tool[Any]]] = []

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.prompts.append(system)
        self.tools.append(tools)
        self.observe(self._call_count, system)
        return super().stream(messages, tools, system)


@pytest.fixture
async def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalToolProvider:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    instance = LocalToolProvider()
    await instance.init(LocalOptions(tmp_path, (tmp_path,), skill_directories=(tmp_path / "skills",)))
    return instance


def agent_for(provider: LocalToolProvider, transport: StubTransport) -> Agent:
    agent = Agent("BASE", transport, tools=provider.all_tools)
    provider.bind(agent)
    return agent


async def test_first_access_from_any_provider_defers_whole_batch(provider: LocalToolProvider) -> None:
    root = provider.paths.cwd
    write(root / "api/AGENTS.md", "API_RULE")
    write(root / "ui/AGENTS.md", "UI_RULE")
    write(root / "unused/AGENTS.md", "NEIGHBOR_RULE")
    seen: list[str] = []

    async def foreign(destination: str) -> str:
        seen.append(destination)
        return "read"

    foreign_tool: Tool[Any] = Tool("foreign", foreign, access=lambda args: (Access("read", args["destination"]),))
    calls = batch(call("write_file", file_path="api/output", content="ok"), call("foreign", destination="ui/input"))

    def observe(turn: int, system: str) -> None:
        assert "NEIGHBOR_RULE" not in system
        if turn == 1:
            assert not (root / "api/output").exists() and not seen
            assert "API_RULE" in system and "UI_RULE" in system

    transport = ObservedTransport([calls, calls, make_text_response()], observe)
    agent = agent_for(provider, transport)
    agent.tools.append(foreign_tool)
    context = MemoryContextStore()
    await agent.run("go", context)
    assert (root / "api/output").read_text() == "ok"
    assert seen == ["ui/input"]
    assert agent.system == "BASE"
    results = [b for msg in await context.get_history() for b in msg.content if isinstance(b, ToolResultBlock)]
    assert [b.is_error for b in results] == [True, True, False, False]


@pytest.mark.parametrize("change", ["edit", "delete"])
async def test_changed_or_deleted_rules_defer_batch(provider: LocalToolProvider, change: str) -> None:
    source = provider.paths.cwd / "AGENTS.md"
    write(source, "OLD_RULE")
    calls = batch(call("write_file", file_path="output", content="ok"))

    def observe(turn: int, system: str) -> None:
        if turn == 0:
            source.unlink() if change == "delete" else source.write_text("NEW_RULE")
        if turn == 1:
            assert not (source.parent / "output").exists()
            assert "OLD_RULE" not in system
            assert ("NEW_RULE" in system) == (change == "edit")

    await agent_for(provider, ObservedTransport([calls, calls, make_text_response()], observe)).run(
        "go", MemoryContextStore()
    )


async def test_instruction_edit_runs_alone(provider: LocalToolProvider) -> None:
    root = provider.paths.cwd
    write(root / "AGENTS.md", "OLD_RULE")
    other = ToolUseBlock("other", "write_file", {"file_path": "output", "content": "ok"})

    def observe(turn: int, system: str) -> None:
        if turn == 1:
            assert "NEW_RULE" in system and "OLD_RULE" not in system
            assert not (root / "output").exists()

    transport = ObservedTransport(
        [
            batch(call("write_file", file_path="AGENTS.md", content="NEW_RULE"), other),
            batch(other),
            make_text_response(),
        ],
        observe,
    )
    await agent_for(provider, transport).run("go", MemoryContextStore())
    assert (root / "output").read_text() == "ok"


async def test_search_and_streaming_paths_use_provider_cwd(provider: LocalToolProvider) -> None:
    root = provider.paths.cwd
    write(root / "tree/nested/AGENTS.md", "SEARCH_RULE")
    write(root / "tree/nested/data", "needle")
    write(root / "shell/AGENTS.md", "SHELL_RULE")
    calls = batch(
        call("search_files", query="needle", path="tree"), call("shell", command="printf hello", cwd="shell")
    )
    transport = ObservedTransport([calls, calls, make_text_response()])
    context = MemoryContextStore()
    events = [event async for event in agent_for(provider, transport).run_stream("go", context)]
    assert "SEARCH_RULE" in transport.prompts[1] and "SHELL_RULE" in transport.prompts[1]
    assert [event.delta for event in events if isinstance(event, ToolOutputDelta)] == ["hello"]
    assert any(
        "data:1: needle" in str(b.content)
        for msg in await context.get_history()
        for b in msg.content
        if isinstance(b, ToolResultBlock)
    )


async def test_skill_load_adds_local_tools_and_reload_replaces_content(provider: LocalToolProvider) -> None:
    source = provider.paths.cwd / "skills/example/SKILL.md"
    write(source, "---\nname: example\ndescription: Example procedure\n---\nLONG_BODY")
    await provider.reload()
    instruction = provider.get_instructions()[0]
    transport = ObservedTransport(
        [
            batch(call("load_instruction", id=instruction.id)),
            batch(call("write_file", file_path="result", content="ok")),
            make_text_response(),
        ]
    )
    agent = Agent("BASE", transport)
    provider.bind(agent)
    await agent.run("go", MemoryContextStore())
    assert "Example procedure" in transport.prompts[0] and "LONG_BODY" not in transport.prompts[0]
    assert "LONG_BODY" in transport.prompts[1]
    assert (provider.paths.cwd / "result").read_text() == "ok"
    assert not agent.tools
    loaded_tools = transport.tools[1]
    assert all(tool.access is not None for tool in loaded_tools if tool.name != "load_instruction")
    source.unlink()
    await provider.reload()
    next_transport = ObservedTransport([make_text_response()])
    agent.transport = next_transport
    await agent.run("again", MemoryContextStore())
    assert "LONG_BODY" not in next_transport.prompts[0]
    assert [tool.name for tool in next_transport.tools[0]] == ["load_instruction"]


async def test_two_agents_and_rebound_copy_have_separate_scopes(provider: LocalToolProvider) -> None:
    write(provider.paths.cwd / "nested/AGENTS.md", "PRIVATE_SCOPE")
    calls = batch(call("read_file", filename="nested/AGENTS.md"))
    first = agent_for(provider, ObservedTransport([calls, calls, make_text_response()]))
    await first.run("go", MemoryContextStore())
    second_transport = ObservedTransport([make_text_response()])
    second = first.copy(transport=second_transport)
    provider.bind(second)
    await second.run("go", MemoryContextStore())
    assert "PRIVATE_SCOPE" not in second_transport.prompts[0]
    assert len(second.before_turn_hooks) == len(first.before_turn_hooks) == 1


async def test_multiple_roots_and_symlink_retargeting(provider: LocalToolProvider, tmp_path: Path) -> None:
    primary, other = tmp_path / "primary", tmp_path / "other"
    write(primary / "AGENTS.md", "PRIMARY_RULE")
    write(other / "AGENTS.md", "OTHER_RULE")
    write(primary / "next/AGENTS.md", "NEXT_RULE")
    link = primary / "alias"
    link.symlink_to(other, target_is_directory=True)
    await provider.init(LocalOptions(primary, (primary, other), isolated=False))
    calls = batch(call("read_file", filename="alias/AGENTS.md"))

    def observe(turn: int, system: str) -> None:
        if turn == 1:
            assert "OTHER_RULE" in system
            link.unlink()
            link.symlink_to(primary / "next", target_is_directory=True)
        if turn == 2:
            assert "NEXT_RULE" in system and "OTHER_RULE" not in system

    transport = ObservedTransport([calls, calls, calls, make_text_response()], observe)
    await agent_for(provider, transport).run("go", MemoryContextStore())


async def test_failed_snapshot_never_runs_dependent_calls(provider: LocalToolProvider) -> None:
    root = provider.paths.cwd
    write(root / "a/AGENTS.md", "a" * 6)
    write(root / "b/AGENTS.md", "b" * 6)
    await provider.init(LocalOptions(root, (root,), max_instruction_bytes=10))
    calls = batch(
        ToolUseBlock("a", "write_file", {"file_path": "a/output", "content": "ok"}),
        ToolUseBlock("b", "write_file", {"file_path": "b/output", "content": "ok"}),
    )
    transport = ObservedTransport([calls, make_text_response()])
    agent = agent_for(provider, transport)
    context = MemoryContextStore()
    await agent.run("go", context)
    assert not (root / "a/output").exists() and not (root / "b/output").exists()
    assert "aaaaaa" not in transport.prompts[1]
    assert any(
        "size limit" in str(b.content)
        for msg in await context.get_history()
        for b in msg.content
        if isinstance(b, ToolResultBlock)
    )


async def test_provider_implements_core_lifecycle(provider: LocalToolProvider) -> None:
    contract: ToolProvider = provider
    assert collect_tools([contract]) == provider.all_tools
    await contract.close()


async def test_skill_loading_declares_read_access_for_policy(provider: LocalToolProvider) -> None:
    from unittest.mock import AsyncMock

    from axio.permission_policy import Approval, PolicyGuard

    source = provider.paths.cwd / "skills/example/SKILL.md"
    write(source, "---\nname: example\ndescription: Example\n---\nREAD_WITH_POLICY")
    await provider.reload()
    instruction = provider.get_instructions()[0]
    prompt = AsyncMock(return_value=Approval())
    policy = PolicyGuard(provider.paths.cwd, prompt)
    transport = ObservedTransport([batch(call("load_instruction", id=instruction.id)), make_text_response()])
    agent = Agent("BASE", transport, providers=[provider], tool_guards=(policy,))
    provider.bind(agent)
    await agent.run("go", MemoryContextStore())
    assert "READ_WITH_POLICY" in transport.prompts[1]
    prompt.assert_not_awaited()
