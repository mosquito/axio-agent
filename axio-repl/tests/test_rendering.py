"""Stream formatting must not depend on provider chunk boundaries."""

from __future__ import annotations

import re

import pytest
from axio import Agent, MemoryContextStore, Tool
from axio.events import (
    BlockEnd,
    IterationEnd,
    ProviderEvent,
    ReasoningDelta,
    ReasoningSignature,
    StreamEvent,
    TextDelta,
)
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.types import StopReason, Usage

from axio_repl import run_prompt


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


THOUGHT = "**Evaluating testing approach**\n\nI'm thinking about tests.\nNext line.\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[THOUGHT], list(THOUGHT), ["", *THOUGHT.splitlines(keepends=True), ""]])
async def test_reasoning_prefixes_lines_not_chunks(chunks: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    events: list[StreamEvent] = []
    for chunk in chunks:
        events.extend(
            [
                ReasoningDelta(index=0, delta=chunk),
                ProviderEvent(provider="test", kind="metadata", data={}),
            ]
        )
    events.extend(make_text_response("Answer."))
    agent = Agent(system="", transport=StubTransport([events]))
    await run_prompt(agent, MemoryContextStore(), "test")
    assert plain(capsys.readouterr().out) == (
        "> **Evaluating testing approach**\n> \n> I'm thinking about tests.\n> Next line.\n"
        "Answer.\n[10in/5out tokens]\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_end", [False, True])
async def test_reasoning_blocks_and_text_have_separate_lines(
    explicit_end: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[StreamEvent] = [
        TextDelta(index=0, delta="Introduction."),
        ReasoningDelta(index=1, delta="First"),
        ReasoningSignature(index=1, signature="opaque"),
        ReasoningDelta(index=1, delta=" thought."),
    ]
    if explicit_end:
        events.append(BlockEnd(index=1))
    events.extend(
        [
            ReasoningDelta(index=2, delta="Second thought."),
            *make_text_response("Answer."),
        ]
    )
    await run_prompt(Agent(system="", transport=StubTransport([events])), MemoryContextStore(), "test")
    assert plain(capsys.readouterr().out) == (
        "Introduction.\n> First thought.\n> Second thought.\nAnswer.\n[10in/5out tokens]\n"
    )


@pytest.mark.asyncio
async def test_reasoning_before_tool_call(capsys: pytest.CaptureFixture[str]) -> None:
    async def echo(msg: str) -> str:
        return msg

    transport = StubTransport(
        [
            [
                ReasoningDelta(index=0, delta="Call"),
                ReasoningDelta(index=0, delta=" tool."),
                *make_tool_use_response(),
            ],
            make_text_response("Done."),
        ]
    )
    agent = Agent(system="", transport=transport, tools=[Tool(name="echo", handler=echo)])
    await run_prompt(agent, MemoryContextStore(), "test")
    output = plain(capsys.readouterr().out)
    assert output.startswith("> Call tool.\n\n▶ echo")
    assert output.count("> ") == 1
    assert "Done.\n[20in/10out tokens]\n" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_reasoning_only_turn_ends_before_usage(failed: bool, capsys: pytest.CaptureFixture[str]) -> None:
    ending: StreamEvent | BaseException = (
        RuntimeError("failed") if failed else IterationEnd(0, StopReason.end_turn, Usage(0, 0))
    )
    transport = StubTransport([[ReasoningDelta(index=0, delta="Thinking."), ending]])
    await run_prompt(Agent(system="", transport=transport), MemoryContextStore(), "test")
    captured = capsys.readouterr()
    assert plain(captured.out) == "> Thinking.\n[0in/0out tokens]\n"
    assert ("Error: failed" in captured.err) is failed
