"""Preflight runs before every handler in a batch, including streaming handlers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from axio import Agent, MemoryContextStore, Tool
from axio.blocks import ToolResultBlock, ToolUseBlock
from axio.events import Error, ToolOutputDelta, ToolResult
from axio.testing import StubTransport, make_text_response, make_tool_use_response


@pytest.mark.parametrize("streaming", [False, True])
async def test_preflight_defers_before_any_handler_runs(streaming: bool) -> None:
    calls: list[str] = []

    async def handler(msg: str) -> str:
        calls.append(msg)
        return msg

    async def stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
        calls.append(msg)
        yield "stdout", msg

    if streaming:
        handler.stream = stream  # type: ignore[attr-defined]
    tool: Tool[Any] = Tool(name="echo", handler=handler)
    responses = [make_tool_use_response("echo", "first", {"msg": "first"}), make_text_response("replanned")]
    transport = StubTransport(responses)
    turns: list[int] = []

    async def before_turn(iteration: int) -> None:
        turns.append(iteration)

    async def preflight(blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        assert iteration == 1 and not calls
        agent.system = "new instructions"
        return [ToolResultBlock(tool_use_id=block.id, content="replan", is_error=True) for block in blocks]

    agent = Agent(system="old", transport=transport, tools=[tool], before_turn=before_turn, before_tools=preflight)
    ctx = MemoryContextStore()
    events = [event async for event in agent.run_stream("go", ctx)]
    assert not calls
    assert not any(isinstance(event, ToolOutputDelta) for event in events)
    assert any(isinstance(event, ToolResult) and event.is_error for event in events)
    assert turns == [1, 2]
    history = await ctx.get_history()
    results = [block for msg in history for block in msg.content if isinstance(block, ToolResultBlock)]
    assert len(results) == 1 and results[0].tool_use_id == "first"


async def test_partial_preflight_direct_dispatch_and_copy() -> None:
    calls: list[str] = []

    async def handler(msg: str) -> str:
        calls.append(msg)
        return msg

    async def preflight(blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        return [ToolResultBlock(tool_use_id="second", content="deferred", is_error=True)]

    agent = Agent(
        system="", transport=StubTransport([]), tools=[Tool(name="echo", handler=handler)], before_tools=preflight
    )
    copied = agent.copy(system="changed")
    results = await copied.dispatch_tools(
        [
            ToolUseBlock(id="first", name="echo", input={"msg": "run"}),
            ToolUseBlock(id="second", name="echo", input={"msg": "skip"}),
        ],
        1,
    )
    assert calls == ["run"]
    assert {item.tool_use_id for item in results} == {"first", "second"}
    assert copied.before_tools is preflight


@pytest.mark.parametrize("kind", ["raises", "unknown", "duplicate"])
async def test_bad_preflight_fails_closed(kind: str) -> None:
    async def handler() -> str:
        raise AssertionError("handler must not start")

    async def preflight(blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        if kind == "raises":
            raise ValueError("configuration failed")
        item = ToolResultBlock(tool_use_id="unknown" if kind == "unknown" else "first", content="skip")
        return [item, item] if kind == "duplicate" else [item]

    agent = Agent(
        system="", transport=StubTransport([]), tools=[Tool(name="t", handler=handler)], before_tools=preflight
    )
    results = await agent.dispatch_tools([ToolUseBlock(id="first", name="t", input={})], 1)
    assert len(results) == 1 and results[0].tool_use_id == "first" and results[0].is_error
    assert "Tool preflight failed:" in str(results[0].content)


async def test_cancelled_preflight_preserves_call_result_pair() -> None:
    entered = asyncio.Event()
    calls: list[str] = []

    async def handler(msg: str) -> str:
        calls.append(msg)
        return msg

    async def preflight(blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        entered.set()
        await asyncio.Event().wait()
        return []

    transport = StubTransport([make_tool_use_response("echo", "first", {"msg": "never"})])
    agent = Agent(system="", transport=transport, tools=[Tool(name="echo", handler=handler)], before_tools=preflight)
    ctx = MemoryContextStore()
    task = asyncio.create_task(agent.run("go", ctx))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not calls
    history = await ctx.get_history()
    uses = [b.id for msg in history for b in msg.content if isinstance(b, ToolUseBlock)]
    results = [b.tool_use_id for msg in history for b in msg.content if isinstance(b, ToolResultBlock)]
    assert uses == results == ["first"]


async def test_before_turn_failure_never_calls_transport() -> None:
    async def before_turn(iteration: int) -> None:
        raise ValueError("invalid instructions")

    transport = StubTransport([])
    agent = Agent(system="", transport=transport, before_turn=before_turn)
    events = [event async for event in agent.run_stream("go", MemoryContextStore())]
    assert transport._call_count == 0
    assert any(isinstance(event, Error) and str(event.exception) == "invalid instructions" for event in events)
