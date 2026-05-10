"""Tests for Agent capability-aware behavior: tool filtering based on model capabilities."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import StreamEvent, ToolResult
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool


async def msg_handler(msg: str) -> str:
    return json.dumps({"msg": msg})


class _ModelTransport(StubTransport):
    model: ModelSpec

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        super().__init__(responses)
        self.tools_received: list[list[Tool[Any]]] = []

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.tools_received.append(tools)
        return super().stream(messages, tools, system)


def _make_transport_with_model(
    responses: list[list[StreamEvent]],
    capabilities: Iterable[Capability],
) -> _ModelTransport:
    """Create a StubTransport with a model attribute that has given capabilities."""
    transport = _ModelTransport(responses)
    transport.model = ModelSpec(id="test-model", capabilities=frozenset(capabilities))
    return transport


class TestToolFiltering:
    async def test_tools_passed_when_model_has_tool_use(self) -> None:
        """When model has tool_use capability, tools are dispatched normally."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = _make_transport_with_model(
            [make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")],
            capabilities=[Capability.text, Capability.tool_use],
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error

    async def test_tools_empty_when_model_lacks_tool_use(self) -> None:
        """When model lacks tool_use capability, no tools are passed to transport."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = _make_transport_with_model(
            [make_text_response("I cannot use tools")],
            capabilities=[Capability.text, Capability.vision],
        )

        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        assert len(transport.tools_received) == 1
        assert transport.tools_received[0] == []

    async def test_tools_passed_when_transport_has_no_model(self) -> None:
        """When transport has no model attribute, tools are passed as-is (backward compat)."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        # StubTransport has no .model attribute by default
        assert not hasattr(transport, "model")

        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error

    async def test_empty_capabilities_filters_tools(self) -> None:
        """When model declares empty capabilities, tools are filtered out."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = _make_transport_with_model(
            [make_text_response("No tools available")],
            capabilities=[],
        )

        agent = Agent(system="test", tools=[tool], transport=transport)
        async for _ in agent.run_stream("go", MemoryContextStore()):
            pass

        assert len(transport.tools_received) >= 1
        assert all(t == [] for t in transport.tools_received)
