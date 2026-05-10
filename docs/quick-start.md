# Quick Start

Write your first agent with the core library.

## Install

**Core library:**

```bash
pip install axio
```

**Transport (pick one or more):**

```bash
pip install axio-transport-openai      # OpenAI, Nebius, OpenRouter, any OpenAI-compatible
pip install axio-transport-anthropic   # Anthropic Claude
pip install axio-transport-google      # Google Gemini + Vertex AI
pip install axio-transport-codex       # ChatGPT via OAuth
```

**Tools (optional):**

```bash
pip install axio-tools-local    # file and shell tools
pip install axio-tools-docker   # isolated Docker sandbox
pip install axio-tools-mcp      # plug any MCP server in as tools
```

## Minimal agent

The smallest possible agent needs a **transport** to talk to an LLM, a
**context store** to hold conversation history, and an **Agent** to tie them
together:

<!-- name: test_minimal_agent -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response


async def main() -> None:
    transport = StubTransport([
        make_text_response("Hello! I'm a stub agent."),
    ])
    context = MemoryContextStore()
    agent = Agent(
        system="You are a helpful assistant.",
        tools=[],
        transport=transport,
    )
    reply = await agent.run("Hi there!", context)
    return reply


assert asyncio.run(main()) == "Hello! I'm a stub agent."
```

Replace `StubTransport` with a real transport to connect to a live LLM:

```python
from axio_transport_openai import OpenAITransport
from axio_transport_anthropic import AnthropicTransport
from axio_transport_google import GoogleTransport
```

The agent loop, tool dispatch, and streaming work the same regardless of
which transport you use.

## Adding tools

Tools are plain `async def` functions. Parameters become the JSON schema
exposed to the LLM; the docstring becomes the description:

<!--
name: test_adding_tools
-->
<!-- name: test_adding_tools -->
```python
from axio import Agent, MemoryContextStore, Tool
from axio.testing import StubTransport, make_text_response

transport = StubTransport([make_text_response("ok")])
context = MemoryContextStore()


async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


agent = Agent(
    system="You are a helpful assistant.",
    tools=[Tool(name="greet", handler=greet)],
    transport=transport,
)
```

## Streaming events

`run_stream()` yields typed `StreamEvent` objects as the agent runs - tokens,
tool calls, and results as they arrive:

<!-- name: test_streaming_example -->
```python
import asyncio
from axio import Agent, MemoryContextStore, TextDelta
from axio.testing import StubTransport, make_text_response
from axio.events import SessionEndEvent


async def main() -> None:
    transport = StubTransport([make_text_response("Streaming works!")])
    context = MemoryContextStore()
    agent = Agent(
        system="You are a helpful assistant.",
        tools=[],
        transport=transport,
    )
    collected = []
    async for event in agent.run_stream("Hello!", context):
        if isinstance(event, TextDelta):
            collected.append(event.delta)
        elif isinstance(event, SessionEndEvent):
            break
    return "".join(collected)


assert asyncio.run(main()) == "Streaming works!"
```

## What's next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Core Concepts
:link: concepts/index
:link-type: doc

Understand the agent loop, protocols, tools, events, and the plugin system.
:::

:::{grid-item-card} How-To Guides
:link: guides/index
:link-type: doc

Writing tools, transports, guards, realtime voice agents, and more.
:::

::::
