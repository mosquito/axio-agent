# Cookbook

Practical recipes for common Axio patterns.

## Agent with memory persistence

Save and restore conversation history:

<!-- name: test_memory_context -->
```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response


transport = StubTransport([make_text_response("Hello!")])


async def main() -> None:
    context = MemoryContextStore()

    agent = Agent(
        system="You are a helpful assistant.",
        tools=[],
        transport=transport,
    )

    reply = await agent.run("Hi", context)
    print(reply)


asyncio.run(main())
```

## Streaming in FastAPI

Build a web API with streaming events:

<!-- name: test_streaming_pattern -->
```python
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio.events import TextDelta


transport = StubTransport([make_text_response("Hello!")])


async def stream_events(message: str, agent: Agent, context: MemoryContextStore):
    """Pattern for streaming - yields events."""
    async for event in agent.run_stream(message, context):
        yield event
```

## RAG with custom tools

Combine retrieval and generation:

<!-- name: test_rag_tools -->
```python
from typing import Any
from axio.tool import Tool, ToolHandler


class RetrieveContext(ToolHandler[Any]):
    """Retrieve relevant context from a knowledge base."""
    query: str

    async def __call__(self, context: Any) -> str:
        return f"Results for: {self.query}"


class GenerateResponse(ToolHandler[Any]):
    """Generate a response using retrieved context."""
    context: str
    question: str

    async def __call__(self, context: Any) -> str:
        return f"Generated: {self.context[:50]}"


# Create tools
retrieve_tool = Tool(
    name="retrieve",
    description="Retrieve relevant context",
    handler=RetrieveContext,
)
generate_tool = Tool(
    name="generate",
    description="Generate response using context",
    handler=GenerateResponse,
)
```

## Multi-agent workflow

Coordinate multiple agents:

<!-- name: test_multi_agent -->
```python
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response


async def main():
    shared_context = MemoryContextStore()

    transport = StubTransport([
        make_text_response("Research result"),
        make_text_response("Final summary"),
    ])

    research_agent = Agent(
        system="Research the topic.",
        tools=[],
        transport=transport,
        context=shared_context,
    )

    writer_agent = Agent(
        system="Write a summary.",
        tools=[],
        transport=transport,
        context=shared_context,
    )

    research_result = await research_agent.run("What is async?")
    final_result = await writer_agent.run(f"Summary: {research_result}")
    print(final_result)
```

## Custom transport with retry

Transport with exponential backoff:

<!-- name: test_retry_transport -->
```python
from typing import AsyncIterator
from axio.messages import Message
from axio.tool import Tool
from axio.events import StreamEvent, TextDelta, IterationEnd
from axio.types import StopReason, Usage


class RetryTransport:
    """Transport with retry logic."""
    max_retries = 3
    base_delay = 0.01

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(index=0, delta="response")
        yield IterationEnd(
            iteration=1,
            stop_reason=StopReason.end_turn,
            usage=Usage(0, 0),
        )
```

## Rate limiting tool

<!-- name: test_rate_limit -->
```python
import asyncio
from typing import Any, ClassVar
from axio.tool import ToolHandler


class RateLimitedTool(ToolHandler[Any]):
    """Tool with rate limiting."""
    rate_limit: ClassVar[int] = 10
    time_window: ClassVar[int] = 60
    data: str

    def __init__(self):
        self._calls = []

    async def __call__(self, *args, **kwargs):
        now = asyncio.get_event_loop().time()
        self._calls = [t for t in self._calls if now - t < self.time_window]

        if len(self._calls) >= self.rate_limit:
            raise RuntimeError(f"Rate limit: {self.rate_limit}/{self.time_window}s")

        self._calls.append(now)
        return "done"
```

## API key guard

Check for required environment variables:

<!-- name: test_api_key_guard -->
```python
import os
from typing import Any
from axio.permission import PermissionGuard
from axio.exceptions import GuardError


class ApiKeyGuard(PermissionGuard):
    """Ensure required environment variables are set."""
    required_keys = ("OPENAI_API_KEY",)

    async def check(self, handler: Any) -> Any:
        missing = [k for k in self.required_keys if not os.environ.get(k)]
        if missing:
            raise GuardError(f"Missing: {', '.join(missing)}")
        return handler
```

## Tool with guards

Apply guards to specific tools:

<!-- name: test_tool_with_guards -->
```python
from typing import Any
from axio.tool import Tool, ToolHandler
from axio.permission import PermissionGuard


class SensitiveTool(ToolHandler[Any]):
    data: str

    async def __call__(self, context: Any) -> str:
        return f"Processed: {self.data}"


class AllowGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        return handler


sensitive_tool = Tool(
    name="sensitive_operation",
    description="Process sensitive data",
    handler=SensitiveTool,
    guards=(AllowGuard(),),
)

assert len(sensitive_tool.guards) == 1
```