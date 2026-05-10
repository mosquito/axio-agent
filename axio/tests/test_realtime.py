"""Headless tests for RealtimeAgent against a stub session.

The stub session yields a scripted sequence of provider events; the agent
should pass them through verbatim and (for tool-use turns) dispatch tool
handlers, sending results back to the session.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from axio.blocks import AudioBlock, ContentBlock, TextBlock
from axio.events import (
    AudioOutputDelta,
    Error,
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from axio.realtime import RealtimeAgent
from axio.tool import Tool
from axio.types import StopReason, Usage

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubSession:
    """Yields a scripted sequence of events; records every outbound call."""

    def __init__(self, scripted: list[StreamEvent]) -> None:
        self._scripted: list[StreamEvent] = scripted
        self.sent: list[ContentBlock | list[ContentBlock]] = []
        self.tool_results: list[tuple[str, str, Any]] = []
        self.commits = 0
        self.interrupts = 0
        self.closed = False
        self._extra_events: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        self.sent.append(content)

    async def commit(self) -> None:
        self.commits += 1

    async def interrupt(self) -> None:
        self.interrupts += 1

    async def send_tool_result(self, tool_use_id: str, name: str, content: Any) -> None:
        self.tool_results.append((tool_use_id, name, content))

    def push_event(self, event: StreamEvent | None) -> None:
        """Inject an event from outside (e.g. after a tool result is delivered)."""
        self._extra_events.put_nowait(event)

    async def events(self) -> AsyncIterator[StreamEvent]:
        for ev in self._scripted:
            yield ev
            # Yield to the event loop so background tool tasks can run between
            # scripted events when the test wants ordering.
            await asyncio.sleep(0)
        # Drain any externally-injected follow-ups.
        while True:
            next_event = await self._extra_events.get()
            if next_event is None:
                return
            yield next_event

    async def close(self) -> None:
        self.closed = True


class _StubTransport:
    def __init__(self, session: _StubSession) -> None:
        self._session = session
        self.connect_kwargs: dict[str, Any] | None = None

    async def connect(self, **kwargs: Any) -> _StubSession:
        self.connect_kwargs = kwargs
        return self._session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPassthrough:
    async def test_events_flow_through(self) -> None:
        """Agent yields every event from the session unchanged."""
        events: list[StreamEvent] = [
            SpeechStarted(),
            TranscriptDelta(role="user", delta="hello"),
            SpeechStopped(),
            TextDelta(index=0, delta="hi"),
            AudioOutputDelta(data=b"\x00\x01\x02"),
            TurnComplete(stop_reason=StopReason.end_turn, usage=Usage(10, 5)),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        async with RealtimeAgent(system="test", transport=transport) as agent:
            session.push_event(None)  # close the iterator after scripted events
            received = [ev async for ev in agent.events()]

        assert received == events
        assert session.closed
        assert transport.connect_kwargs is not None
        assert transport.connect_kwargs["system"] == "test"
        assert transport.connect_kwargs["tools"] == []

    async def test_send_routes_to_session(self) -> None:
        session = _StubSession([])
        transport = _StubTransport(session)
        async with RealtimeAgent(system="test", transport=transport) as agent:
            await agent.send(TextBlock(text="hello"))
            await agent.send(AudioBlock(media_type="audio/pcm", data=b"\x01\x02"))
            await agent.commit()

        assert session.sent == [TextBlock(text="hello"), AudioBlock(media_type="audio/pcm", data=b"\x01\x02")]
        assert session.commits == 1


class TestToolDispatch:
    async def test_single_tool_round_trip(self) -> None:
        """Tool-use turn → tool fires → result sent → next turn proceeds."""
        calls: list[dict[str, int]] = []

        async def add(a: int, b: int) -> str:
            calls.append({"a": a, "b": b})
            return str(a + b)

        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="add"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json='{"a": 17, "b": 25}'),
            TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(20, 5)),
            # After the tool result is delivered, the provider would emit the
            # next assistant turn — we script it directly.
            TextDelta(index=0, delta="The answer is 42"),
            TurnComplete(stop_reason=StopReason.end_turn, usage=Usage(25, 8)),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        async with RealtimeAgent(
            system="test",
            transport=transport,
            tools=[Tool(name="add", handler=add)],
        ) as agent:
            session.push_event(None)
            collected = [ev async for ev in agent.events()]

        assert calls == [{"a": 17, "b": 25}]
        assert session.tool_results == [("c1", "add", "42")]
        # Original events still surfaced to the consumer.
        assert any(isinstance(e, TurnComplete) for e in collected)

    async def test_unknown_tool_yields_error_result(self) -> None:
        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="nope"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json="{}"),
            TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(5, 0)),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        async with RealtimeAgent(system="test", transport=transport) as agent:
            session.push_event(None)
            async for _ in agent.events():
                pass

        assert len(session.tool_results) == 1
        tid, name, content = session.tool_results[0]
        assert tid == "c1"
        assert name == "nope"
        assert "Unknown tool" in str(content)

    async def test_handler_exception_surfaces_as_error_result(self) -> None:
        async def boom(**kwargs: Any) -> str:
            raise RuntimeError("kaboom")

        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="boom"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json="{}"),
            TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(5, 0)),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)
        async with RealtimeAgent(
            system="test",
            transport=transport,
            tools=[Tool(name="boom", handler=boom)],
        ) as agent:
            session.push_event(None)
            async for _ in agent.events():
                pass

        assert len(session.tool_results) == 1
        _, _, content = session.tool_results[0]
        assert "kaboom" in str(content)

    async def test_malformed_json_args_treated_as_empty(self) -> None:
        async def echo(msg: str = "default") -> str:
            return msg

        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="echo"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json='{"msg": "trunc'),  # truncated
            TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(5, 0)),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)
        async with RealtimeAgent(
            system="test",
            transport=transport,
            tools=[Tool(name="echo", handler=echo)],
        ) as agent:
            session.push_event(None)
            async for _ in agent.events():
                pass

        # Empty input → handler runs with default value
        assert session.tool_results == [("c1", "echo", "default")]

    async def test_concurrent_tool_dispatch_does_not_block_audio(self) -> None:
        """A slow tool runs in a background task; audio output keeps streaming."""
        gate = asyncio.Event()

        async def slow() -> str:
            await gate.wait()
            return "done"

        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="slow"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json="{}"),
            TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(5, 0)),
            # These audio chunks are emitted before the slow tool finishes,
            # proving dispatch runs concurrently.
            AudioOutputDelta(data=b"\x10"),
            AudioOutputDelta(data=b"\x20"),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        async with RealtimeAgent(
            system="test",
            transport=transport,
            tools=[Tool(name="slow", handler=slow)],
        ) as agent:
            collected: list[StreamEvent] = []

            async def consumer() -> None:
                async for ev in agent.events():
                    collected.append(ev)

            task = asyncio.create_task(consumer())
            # Give the loop a tick to drain scripted events and start the tool.
            for _ in range(5):
                await asyncio.sleep(0)

            # Audio chunks reached the consumer while the tool is still blocked.
            audio_chunks = [e for e in collected if isinstance(e, AudioOutputDelta)]
            assert len(audio_chunks) == 2
            assert session.tool_results == []  # tool hasn't finished yet

            gate.set()
            session.push_event(None)
            await task

        assert session.tool_results == [("c1", "slow", "done")]


class TestInterrupt:
    async def test_interrupt_cancels_pending_tools(self) -> None:
        cancelled = asyncio.Event()

        async def slow() -> str:
            try:
                await asyncio.sleep(60)
                return "should not happen"
            except asyncio.CancelledError:
                cancelled.set()
                raise

        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="slow"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json="{}"),
            TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(5, 0)),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        agent = RealtimeAgent(
            system="test",
            transport=transport,
            tools=[Tool(name="slow", handler=slow)],
        )
        await agent.connect()

        async def consumer() -> None:
            async for _ in agent.events():
                pass

        task = asyncio.create_task(consumer())
        for _ in range(5):
            await asyncio.sleep(0)

        await agent.interrupt()
        assert cancelled.is_set()
        assert session.interrupts == 1

        session.push_event(None)
        await task
        await agent.close()

    async def test_interrupt_drops_pending_tool_fragments(self) -> None:
        """interrupt() must clear half-streamed fragments so a TurnComplete
        arriving afterwards cannot finalize and dispatch stale tool calls."""
        calls: list[dict[str, int]] = []

        async def add(a: int, b: int) -> str:
            calls.append({"a": a, "b": b})
            return str(a + b)

        # Script the fragments without a TurnComplete; we'll inject one *after*
        # interrupt() to simulate the racy ordering.
        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="add"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json='{"a":1,"b":2}'),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        agent = RealtimeAgent(system="test", transport=transport, tools=[Tool(name="add", handler=add)])
        await agent.connect()

        async def consumer() -> None:
            async for _ in agent.events():
                pass

        task = asyncio.create_task(consumer())
        for _ in range(5):
            await asyncio.sleep(0)

        await agent.interrupt()
        # The provider's "tool_use" TurnComplete arrives after interrupt — no
        # dispatch should happen because _pending is empty.
        session.push_event(TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(0, 0)))
        session.push_event(None)
        await task
        await agent.close()

        assert calls == []
        assert session.tool_results == []


class TestErrorPropagation:
    """Server-side error events must reach the consumer — silently swallowing
    them is what made earlier smoke runs hang waiting for events that would
    never arrive."""

    async def test_error_event_raises_by_default(self) -> None:
        boom = RuntimeError("session.update rejected")
        events: list[StreamEvent] = [
            TextDelta(index=0, delta="prefix"),
            Error(exception=boom),
            TextDelta(index=0, delta="never seen"),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        async with RealtimeAgent(system="test", transport=transport) as agent:
            session.push_event(None)
            seen: list[StreamEvent] = []
            with pytest.raises(RuntimeError, match="session.update rejected"):
                async for ev in agent.events():
                    seen.append(ev)

        assert seen == [TextDelta(index=0, delta="prefix")]

    async def test_error_event_yielded_when_raise_on_error_disabled(self) -> None:
        boom = RuntimeError("transient")
        events: list[StreamEvent] = [
            TextDelta(index=0, delta="a"),
            Error(exception=boom),
            TextDelta(index=0, delta="b"),
        ]
        session = _StubSession(events)
        transport = _StubTransport(session)

        async with RealtimeAgent(system="test", transport=transport, raise_on_error=False) as agent:
            session.push_event(None)
            seen = [ev async for ev in agent.events()]

        assert any(isinstance(ev, Error) for ev in seen)
        assert TextDelta(index=0, delta="a") in seen
        assert TextDelta(index=0, delta="b") in seen


class TestSendToolResultFailure:
    async def test_transport_error_logged_not_propagated(self, caplog: pytest.LogCaptureFixture) -> None:
        class FailingSession(_StubSession):
            async def send_tool_result(self, tool_use_id: str, name: str, content: Any) -> None:
                raise RuntimeError("ws disconnected")

        async def add(a: int, b: int) -> str:
            return str(a + b)

        events: list[StreamEvent] = [
            ToolUseStart(index=0, tool_use_id="c1", name="add"),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json='{"a":1,"b":2}'),
            TurnComplete(stop_reason=StopReason.tool_use, usage=Usage(0, 0)),
        ]
        session = FailingSession(events)
        transport = _StubTransport(session)

        with caplog.at_level("ERROR", logger="axio.realtime"):
            async with RealtimeAgent(
                system="test", transport=transport, tools=[Tool(name="add", handler=add)]
            ) as agent:
                session.push_event(None)
                async for _ in agent.events():
                    pass

        assert "Failed to deliver tool result" in caplog.text
        assert "c1" in caplog.text


class TestLifecycle:
    async def test_double_connect_raises(self) -> None:
        session = _StubSession([])
        transport = _StubTransport(session)
        agent = RealtimeAgent(system="test", transport=transport)
        await agent.connect()
        with pytest.raises(RuntimeError):
            await agent.connect()
        await agent.close()

    async def test_use_before_connect_raises(self) -> None:
        session = _StubSession([])
        transport = _StubTransport(session)
        agent = RealtimeAgent(system="test", transport=transport)
        with pytest.raises(RuntimeError):
            await agent.send(TextBlock(text="x"))

    async def test_close_is_idempotent(self) -> None:
        session = _StubSession([])
        transport = _StubTransport(session)
        agent = RealtimeAgent(system="test", transport=transport)
        await agent.connect()
        await agent.close()
        await agent.close()  # no-op
        assert session.closed
