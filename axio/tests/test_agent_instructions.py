"""Instructions are loaded by providers and applied at agent iteration boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from typing import Any

import pytest

from axio import Agent, MemoryContextStore
from axio.blocks import ToolResultBlock, ToolUseBlock
from axio.events import Error, StreamEvent, ToolOutputDelta, ToolResult
from axio.instructions import Instruction, LoadedInstruction
from axio.messages import Message
from axio.permission import DenyAllGuard
from axio.testing import StubTransport, make_echo_tool, make_text_response, make_tool_use_response
from axio.tool import Tool


class Rules(Instruction):
    def __init__(self, text: str = "Full rules", tools: tuple[Tool[Any], ...] = ()) -> None:
        super().__init__("example:rules", "Short rules", source="example")
        self.text = text
        self.tools = tools
        self.loads = 0

    async def load(self) -> LoadedInstruction:
        self.loads += 1
        return LoadedInstruction(self.text, self.tools)


class RecordingTransport(StubTransport):
    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        super().__init__(responses)
        self.requests: list[tuple[list[Tool[Any]], str]] = []

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.requests.append((list(tools), system))
        return super().stream(messages, tools, system)


def load_response(id: str = "load") -> list[StreamEvent]:
    return make_tool_use_response("load_instruction", id, {"id": "example:rules"})


async def test_load_returns_content_and_tools_for_next_iteration() -> None:
    tool = make_echo_tool()
    rules = Rules(tools=(tool,))
    transport = RecordingTransport([load_response(), make_tool_use_response(), make_text_response()])
    agent = Agent("Base", transport, instructions=[rules])
    events = [event async for event in agent.run_stream("go", MemoryContextStore())]
    assert rules.loads == 1
    assert [t.name for t in transport.requests[0][0]] == ["load_instruction"]
    assert "Short rules" in transport.requests[0][1]
    assert "Full rules" not in transport.requests[0][1]
    assert transport.requests[1][0][-1] is tool
    assert "Full rules" in transport.requests[1][1]
    assert agent.system == "Base" and agent.tools == []
    assert any(
        isinstance(e, ToolResult) and e.name == "load_instruction" and e.content == "Full rules" for e in events
    )


async def test_required_instruction_without_tools_loads_before_request() -> None:
    rules = Rules()
    transport = RecordingTransport([make_text_response()])
    agent = Agent("Base", transport, instructions=[rules], required_instructions=(rules.id,))
    await agent.run("go", MemoryContextStore())
    assert rules.loads == 1
    assert "Full rules" in transport.requests[0][1]


async def test_copy_has_independent_instruction_state() -> None:
    rules = Rules(tools=(make_echo_tool(),))
    agent = Agent("Base", RecordingTransport([load_response(), make_text_response()]), instructions=[rules])
    await agent.run("go", MemoryContextStore())
    transport = RecordingTransport([make_text_response()])
    child = agent.copy(transport=transport)
    await child.run("go", MemoryContextStore())
    assert "Full rules" not in transport.requests[0][1]
    assert [t.name for t in transport.requests[0][0]] == ["load_instruction"]
    child.instructions.clear()
    assert agent.instructions == [rules]


async def test_new_tool_cannot_run_in_the_loading_batch() -> None:
    calls: list[str] = []

    async def write() -> str:
        calls.append("write")
        return "written"

    tool: Tool[Any] = Tool(name="write", handler=write)
    batch = [*load_response()[:-1], *make_tool_use_response("write", "write", {})]
    transport = RecordingTransport([batch, make_text_response()])
    agent = Agent("Base", transport, instructions=[Rules(tools=(tool,))])
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert not calls
    assert any(isinstance(e, ToolResult) and e.name == "write" and e.is_error for e in events)
    assert tool in transport.requests[1][0]


async def test_selector_sees_loaded_tools_and_controls_dispatch() -> None:
    seen: list[list[str]] = []

    class Selector:
        async def select(self, messages: Iterable[Message], tools: Iterable[Tool[Any]]) -> Iterable[Tool[Any]]:
            candidates = list(tools)
            seen.append([t.name for t in candidates])
            return [t for t in candidates if t.name != "echo"]

    transport = RecordingTransport([load_response(), make_tool_use_response(), make_text_response()])
    agent = Agent("Base", transport, instructions=[Rules(tools=(make_echo_tool(),))], selector=Selector())
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert "echo" in seen[1]
    assert "echo" not in [t.name for t in transport.requests[1][0]]
    assert any(isinstance(e, ToolResult) and e.name == "echo" and e.is_error for e in events)


async def test_conflicting_tool_load_does_not_activate_content() -> None:
    transport = RecordingTransport([load_response(), make_text_response()])
    agent = Agent("Base", transport, tools=[make_echo_tool()], instructions=[Rules(tools=(make_echo_tool(),))])
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert "Full rules" not in transport.requests[1][1]
    assert any(isinstance(e, ToolResult) and e.is_error and "Conflicting tool" in e.content for e in events)


async def test_provider_failure_does_not_activate_content() -> None:
    class Broken(Rules):
        async def load(self) -> LoadedInstruction:
            raise ValueError("provider unavailable")

    transport = RecordingTransport([load_response(), make_text_response()])
    agent = Agent("Base", transport, instructions=[Broken()])
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert "Full rules" not in transport.requests[1][1]
    assert any(isinstance(e, ToolResult) and e.is_error and e.content == "provider unavailable" for e in events)


async def test_cancelled_load_preserves_history_and_does_not_activate() -> None:
    started = asyncio.Event()

    class Waiting(Rules):
        async def load(self) -> LoadedInstruction:
            started.set()
            await asyncio.Event().wait()
            return LoadedInstruction("Never loaded")

    agent = Agent("Base", RecordingTransport([load_response()]), instructions=[Waiting()])
    context = MemoryContextStore()
    task = asyncio.create_task(agent.run("go", context))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    history = await context.get_history()
    uses = [b.id for m in history for b in m.content if isinstance(b, ToolUseBlock)]
    results = [b.tool_use_id for m in history for b in m.content if isinstance(b, ToolResultBlock)]
    assert uses == results == ["load"]
    transport = RecordingTransport([make_text_response()])
    agent.transport = transport
    await agent.run("again", context)
    assert "Never loaded" not in transport.requests[0][1]


async def test_reloading_catalog_replaces_and_removes_active_content() -> None:
    rules = Rules("Old rules", (make_echo_tool(),))
    agent = Agent("Base", RecordingTransport([load_response(), make_text_response()]), instructions=[rules])
    await agent.run("go", MemoryContextStore())
    agent.instructions = [Rules("New rules")]
    transport = RecordingTransport([load_response(), make_text_response()])
    agent.transport = transport
    await agent.run("reload", MemoryContextStore())
    assert "Old rules" not in transport.requests[0][1]
    assert "echo" not in [t.name for t in transport.requests[0][0]]
    assert "New rules" in transport.requests[1][1]
    agent.instructions = []
    agent.transport = transport = RecordingTransport([make_text_response()])
    await agent.run("removed", MemoryContextStore())
    assert transport.requests == [([], "Base")]


@pytest.mark.parametrize("streaming", [False, True])
async def test_hooks_compose_and_new_required_content_defers_entire_batch(streaming: bool) -> None:
    order: list[str] = []
    calls: list[str] = []
    rules = Rules()

    async def handler(msg: str) -> str:
        calls.append(msg)
        return msg

    async def stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
        calls.append(msg)
        yield "stdout", msg

    if streaming:
        handler.stream = stream  # type: ignore[attr-defined]

    async def original_turn(iteration: int) -> None:
        order.append("original_turn")

    async def provider_turn(iteration: int) -> None:
        order.append("provider_turn")

    async def original_preflight(blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        order.append("original_preflight")
        return []

    async def provider_preflight(blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        order.append("provider_preflight")
        await agent.load_instruction(rules.id)
        return []

    transport = RecordingTransport([make_tool_use_response(), make_tool_use_response(), make_text_response()])
    agent = Agent(
        "Base",
        transport,
        tools=[Tool(name="echo", handler=handler)],
        instructions=[rules],
        before_turn=original_turn,
        before_turn_hooks=(provider_turn,),
        before_tools=original_preflight,
        before_tools_hooks=(provider_preflight,),
        system_contributions=(lambda: "Contribution",),
    )
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert calls == ["hi"]
    assert order[:4] == ["original_turn", "provider_turn", "original_preflight", "provider_preflight"]
    assert "Full rules" not in transport.requests[0][1]
    assert "Full rules" in transport.requests[1][1]
    assert all(system.startswith("Base\n\nContribution") for _, system in transport.requests)
    assert sum(isinstance(e, ToolResult) and e.is_error for e in events) == 1
    assert sum(isinstance(e, ToolOutputDelta) for e in events) == int(streaming)


async def test_global_guards_apply_to_tools_from_instructions() -> None:
    tool = make_echo_tool()
    rules = Rules(tools=(tool,))
    transport = RecordingTransport([make_tool_use_response(), make_text_response()])
    agent = Agent(
        "Base", transport, instructions=[rules], required_instructions=(rules.id,), tool_guards=(DenyAllGuard(),)
    )
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert any(isinstance(e, ToolResult) and e.name == "echo" and e.is_error and e.content == "denied" for e in events)
    assert transport.requests[0][0][-1] is transport.requests[1][0][-1]


async def test_required_load_failure_prevents_model_request() -> None:
    transport = RecordingTransport([make_text_response()])
    agent = Agent("Base", transport, required_instructions=("missing",))
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert not transport.requests
    assert any(isinstance(e, Error) and "Unknown instruction" in str(e.exception) for e in events)


async def test_two_instructions_can_share_one_tool_without_duplicate_schemas() -> None:
    tool = make_echo_tool()
    first, second = Rules("First", (tool,)), Rules("Second", (tool,))
    second.id = "example:second"
    transport = RecordingTransport([make_text_response()])
    agent = Agent("Base", transport, instructions=[first, second], required_instructions=(first.id, second.id))
    await agent.run("go", MemoryContextStore())
    assert [t.name for t in transport.requests[0][0]].count("echo") == 1
    assert "First" in transport.requests[0][1] and "Second" in transport.requests[0][1]


async def test_failed_required_batch_discards_previously_staged_results() -> None:
    rules = Rules(tools=(make_echo_tool(),))
    transport = RecordingTransport([make_text_response()])
    agent = Agent("Base", transport, instructions=[rules], required_instructions=(rules.id, "missing"))
    events = [e async for e in agent.run_stream("go", MemoryContextStore())]
    assert any(isinstance(e, Error) for e in events)
    assert not transport.requests
    agent.required_instructions = ()
    await agent.run("retry", MemoryContextStore())
    assert "Full rules" not in transport.requests[0][1]
    assert "echo" not in [t.name for t in transport.requests[0][0]]


async def test_application_guards_keep_shared_provider_concurrency() -> None:
    from axio.permission import AllowAllGuard

    active = 0
    peak = 0

    async def limited() -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return "done"
        finally:
            active -= 1

    tool: Tool[Any] = Tool("limited", limited, concurrency=1)
    agents = [Agent("", StubTransport([]), tools=[tool], tool_guards=(AllowAllGuard(),)) for _ in range(2)]
    results = await asyncio.gather(
        *(agent.dispatch_tools([ToolUseBlock(str(index), "limited", {})], 0) for index, agent in enumerate(agents))
    )
    assert peak == 1
    assert all(batch[0].content == "done" and not batch[0].is_error for batch in results)
