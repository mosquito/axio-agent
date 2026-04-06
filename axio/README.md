# axio

[![PyPI](https://img.shields.io/pypi/v/axio)](https://pypi.org/project/axio/)
[![Python](https://img.shields.io/pypi/pyversions/axio)](https://pypi.org/project/axio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Minimal, streaming-first, protocol-driven foundation for LLM-powered agents.

One dependency (`pydantic`). Three protocols. An agent loop that just works.

## Features

- **Streaming agent loop** — `run_stream()` yields typed events as they arrive; no buffering, no polling
- **Three clean protocols** — `CompletionTransport`, `ContextStore`, `PermissionGuard`; swap any piece without touching the rest
- **Concurrent tool dispatch** — all tool calls in a turn run via `asyncio.gather` automatically
- **Context compaction** — `compact_context()` summarises old history to stay within token limits
- **Testing helpers** — `StubTransport`, `make_tool_use_response()`, `make_echo_tool()` ship in `axio.testing`
- **Plugin-ready** — entry-point groups (`axio.tools`, `axio.transport`, `axio.guards`) for drop-in extensions

## Installation

```bash
pip install axio
```

## Quick start

```python
import asyncio
from axio import Agent
from axio.context import MemoryContextStore
from axio.tool import Tool, ToolHandler

# 1. Define a tool
class Greet(ToolHandler):
    """Return a greeting for the given name."""
    name: str

    async def __call__(self) -> str:
        return f"Hello, {self.name}!"

greet_tool = Tool(name="greet", description="Greet someone by name", handler=Greet)

# 2. Wire up the agent (transport comes from an axio-transport-* package)
from axio_transport_openai import OpenAITransport

transport = OpenAITransport(api_key="sk-...", model="gpt-4o-mini")
agent = Agent(system="You are helpful.", tools=[greet_tool], transport=transport)

# 3. Run
async def main() -> None:
    ctx = MemoryContextStore()
    async for event in agent.run_stream("Please greet Alice", ctx):
        print(event)

asyncio.run(main())
```

## Architecture

```
  User message
       │
       ▼
  ┌─────────┐   stream()    ┌─────────────────────┐
  │  Agent  │ ────────────▶ │ CompletionTransport  │
  │  loop   │ ◀──────────── │ (Anthropic, OpenAI, …) │
  └─────────┘  StreamEvent  └─────────────────────┘
       │
       │ tool_use?
       ▼
  ┌──────────┐   check()   ┌─────────────────┐
  │   Tool   │ ──────────▶ │ PermissionGuard │
  │ handler  │             │ (path, LLM, …)  │
  └──────────┘             └─────────────────┘
       │
       ▼
  ┌──────────────┐
  │ ContextStore │  append() / get_history() / fork() / compact()
  └──────────────┘
```

## Protocols

### CompletionTransport

```python
from typing import Protocol, runtime_checkable
from collections.abc import AsyncIterator
from axio.events import StreamEvent
from axio.messages import Message
from axio.tool import Tool

@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self, messages: list[Message], tools: list[Tool], system: str
    ) -> AsyncIterator[StreamEvent]: ...
```

### ContextStore

```python
class ContextStore(ABC):
    async def append(self, message: Message) -> None: ...
    async def get_history(self) -> list[Message]: ...
    async def fork(self) -> ContextStore: ...   # branch conversation
    async def clear(self) -> None: ...
    async def close(self) -> None: ...
```

### PermissionGuard

```python
@runtime_checkable
class PermissionGuard(Protocol):
    async def check(self, handler: ToolHandler) -> ToolHandler: ...
```

## Stream events

| Event | Description |
|---|---|
| `TextDelta` | Incremental assistant text chunk |
| `ToolUseStart` | Tool call begins (name + id) |
| `ToolInputDelta` | Streaming JSON fragment for tool arguments |
| `ToolResult` | Tool execution result |
| `IterationEnd` | One LLM round complete — carries `Usage` + `StopReason` |
| `Error` | Transport or tool exception |
| `SessionEndEvent` | Agent loop finished — carries total `Usage` |

## Tools

```python
from axio.tool import Tool, ToolHandler

class Summarise(ToolHandler):
    """Summarise the given text in one sentence."""
    text: str
    max_words: int = 20

    async def __call__(self) -> str:
        # your implementation
        return "..."

tool = Tool(
    name="summarise",
    description="Summarise text",   # overrides docstring if set
    handler=Summarise,
    concurrency=4,                   # max parallel executions
)
```

## Testing

```python
from axio.testing import (
    StubTransport,
    make_tool_use_response,
    make_text_response,
    make_ephemeral_context,
    make_echo_tool,
)

async def test_agent_calls_tool():
    transport = StubTransport([
        make_tool_use_response("echo", tool_input={"msg": "hi"}),
        make_text_response("Done"),
    ])
    agent = Agent(system="", tools=[make_echo_tool()], transport=transport)
    result = await agent.run("say hi", make_ephemeral_context())
    assert result == "Done"
```

## Plugin entry points

```toml
[project.entry-points."axio.tools"]
my_tool = "my_package:MyHandler"

[project.entry-points."axio.transport"]
my_backend = "my_package:MyTransport"

[project.entry-points."axio.guards"]
my_guard = "my_package:MyGuard"
```

## Ecosystem

| Package | Purpose |
|---|---|
| [axio-transport-anthropic](https://github.com/axio-agent/axio-transport-anthropic) | Anthropic Claude transport |
| [axio-transport-openai](https://github.com/axio-agent/axio-transport-openai) | OpenAI-compatible transport (OpenAI, Nebius, OpenRouter, custom) |
| [axio-transport-codex](https://github.com/axio-agent/axio-transport-codex) | ChatGPT OAuth transport |
| [axio-tools-local](https://github.com/axio-agent/axio-tools-local) | Shell, file, Python tools |
| [axio-tools-mcp](https://github.com/axio-agent/axio-tools-mcp) | MCP server bridge |
| [axio-tools-docker](https://github.com/axio-agent/axio-tools-docker) | Docker sandbox tools |
| [axio-tui](https://github.com/axio-agent/axio-tui) | Textual TUI application |
| [axio-tui-rag](https://github.com/axio-agent/axio-tui-rag) | RAG / semantic search plugin |
| [axio-tui-guards](https://github.com/axio-agent/axio-tui-guards) | Permission guard plugins |

## License

MIT
