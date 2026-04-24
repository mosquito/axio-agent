# Axio

[![GitHub org](https://img.shields.io/badge/github-axio--agent-181717?logo=github&logoColor=white)](https://github.com/axio-agent)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/axio-agent/axio/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Docs](https://img.shields.io/github/actions/workflow/status/axio-agent/docs/docs.yml?label=docs&logo=readthedocs&logoColor=white)](https://github.com/axio-agent/docs/actions)
[![PyPI axio](https://img.shields.io/pypi/v/axio?label=axio&logo=pypi&logoColor=white)](https://pypi.org/project/axio/)
[![PyPI axio-tui](https://img.shields.io/pypi/v/axio-tui?label=axio-tui&logo=pypi&logoColor=white)](https://pypi.org/project/axio-tui/)

**A highly extensible, streaming-first agent framework for Python.**

Axio gives you a minimal but complete foundation for building LLM-powered agents.
Every integration point is a protocol — bring your own transport, context store,
tools, and permission guards.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Quick Start
:link: quick-start
:link-type: doc

Install the TUI and start chatting with an LLM agent in under a minute.
:::

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Write a minimal agent from scratch with the core library.
:::

:::{grid-item-card} Core Concepts
:link: concepts/index
:link-type: doc

Understand the agent loop, protocols, tools, events, and the plugin system.
:::

:::{grid-item-card} How-To Guides
:link: guides/index
:link-type: doc

Step-by-step guides for writing custom tools, transports, and guards.
:::

:::{grid-item-card} Packages
:link: packages
:link-type: doc

Overview of every package in the monorepo and their entry points.
:::

::::

## How to draw an owl

::::{grid} 1 1 2 2
:gutter: 2
:class-row: owl-row

:::{grid-item}
:columns: 12 12 2 2
:class: owl-caption

```{image} _static/axio-circles.svg
:alt: Step 1
:width: 80px
:align: center
```

**Step 1.**
:::

:::{grid-item}
:columns: 12 12 10 10

<!-- name: test_index_example -->
```python
import aiohttp
from axio.tool import Tool, ToolHandler


class Fetch(ToolHandler):
    """Fetch the text content of a URL."""
    url: str

    async def __call__(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as r:
                return (await r.text())[:2000]

fetch = Tool(name="fetch", description=Fetch.__doc__, handler=Fetch)
assert fetch.name == "fetch"
```
:::

::::

::::{grid} 1 1 2 2
:gutter: 2
:class-row: owl-row

:::{grid-item}
:columns: 12 12 2 2
:class: owl-caption

```{image} _static/logo.svg
:alt: Step 2
:width: 80px
:align: center
```

**Step 2.**
:::

:::{grid-item}
:columns: 12 12 10 10

<!--
name: test_index_example
```python
import axio_transport_openai
from axio.testing import StubTransport, make_text_response
axio_transport_openai.OpenAITransport = lambda: StubTransport([make_text_response("The weather is sunny.")])
```
-->
<!-- name: test_index_example -->
```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport


async def main() -> None:
    agent = Agent(
        system="You are a helpful assistant.",
        tools=[fetch],
        transport=OpenAITransport(),
    )
    reply = await agent.run(
        "What's the weather tomorrow? Use geoip for detect my location and wttr.in for weather.",
        MemoryContextStore(),
    )
    assert reply

asyncio.run(main())
```
:::

::::

## Why Axio?

Extensible by design
: Every integration point is a runtime-checkable `Protocol` or ABC.
  Swap transports, context stores, tools, and guards without touching framework code.

Streaming by default
: All LLM I/O flows through typed `StreamEvent` values.
  No hidden buffering — you see every token, tool call, and result as it happens.

Tools are Pydantic models
: Define parameters as fields, get JSON schema for free, override `__call__` for execution.
  Guards gate every tool call through a composable permission chain.

Multi-agent orchestration built-in
: Spawn sub-agents via the `subagent` tool, share context between agents,
  compose complex workflows — all without external dependencies.

### Architecture

```{mermaid}
flowchart TB
    subgraph User["User Code"]
        A[Agent]
    end

    subgraph Core["axio - Core Framework"]
        B[Tool Handler<br/>Pydantic model]
        C[Permission Guard<br/>Protocol]
        D[StreamEvent<br/>Typed events]
    end

    subgraph Transport["Transport (pluggable)"]
        E[OpenAI Transport]
        F[Anthropic Transport]
        G[Custom Transport]
    end

    subgraph Context["Context Store (pluggable)"]
        H[MemoryContextStore]
        I[SQLiteContextStore]
    end

    subgraph LLM["LLM Provider"]
        J[OpenAI]
        K[Claude]
        L[Custom]
    end

    A -->|1. configures| B
    A -->|2. uses| C
    A -->|3. builds| D
    A -->|4. sends to| E
    A -->|5. stores in| H
    E -->|SSE| D
    D -->|tool call| B
    B -->|result| D
    D -->|text| J
    J -->|response| E
    E -->|reply| A
```

The agent loop:
1. You configure the agent with tools (Pydantic models) and guards (permission chain)
2. User sends a message
3. Agent sends to transport → LLM
4. LLM responds with text or tool calls
5. Tools execute → results return → LLM generates final response
6. Events stream back to you (tokens, tool calls, results)

## How does Axio compare?

Here's how Axio compares to other popular Python agent frameworks:

| | Axio | LangChain / LangGraph | AutoGen |
|---|---|---|---|
| **Architecture** | Minimal core + protocols | Heavy abstraction layer | Multi-agent orchestration |
| **Streaming** | Built-in from day one | Added later, inconsistent | Limited |
| **Tool definition** | Pydantic models | Functions + decorators | Class-based agents |
| **Transport** | Pluggable protocol | Built-in, harder to swap | Azure OpenAI focused |
| **Multi-agent** | Built-in (subagent tool) | Via LangGraph | Native |
| **Learning curve** | Low — ~100 lines for agent | Medium — many abstractions | High — complex configs |
| **Scope** | Agent loop + extensions | Full stack (RAG, chains, etc.) | Multi-agent scenarios |

### When to choose each

**Choose Axio if:**
- You want a minimal foundation and full control over integrations
- Streaming and visibility into agent decisions matter
- You prefer explicit patterns over implicit "magic"
- You're building a custom agent UI or need to swap LLM providers

**Choose LangChain if:**
- You need RAG, text splitters, and other built-in utilities
- You want batteries-included with less wiring code
- You're prototyping quickly and can accept abstraction overhead

**Choose AutoGen if:**
- You're building complex multi-agent scenarios with conversation flows
- You need built-in support for human-in-the-loop
- You're okay with Azure OpenAI as the primary backend

**Actually — Axio also supports multi-agent:**
- Sub-agents via the built-in `subagent` tool
- Shared context stores for agent-to-agent communication
- Composable workflows without external dependencies

Axio's philosophy is thin abstraction over the prompt-completion loop,
not a full framework with opinions about how you should structure your application.
If that aligns with your needs — welcome.

```{toctree}
:maxdepth: 2
:hidden:

quick-start
getting-started
concepts/index
guides/index
packages
api
troubleshooting
glossary
```
