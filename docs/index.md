# Axio

[![GitHub](https://img.shields.io/badge/github-mosquito%2Faxio--agent-181717?logo=github&logoColor=white)](https://github.com/mosquito/axio-agent)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/mosquito/axio-agent/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Docs](https://img.shields.io/github/actions/workflow/status/mosquito/axio-agent/docs.yml?label=docs&logo=readthedocs&logoColor=white)](https://github.com/mosquito/axio-agent/actions)
[![PyPI axio](https://img.shields.io/pypi/v/axio?label=axio&logo=pypi&logoColor=white)](https://pypi.org/project/axio/)
[![PyPI axio-tui](https://img.shields.io/pypi/v/axio-tui?label=axio-tui&logo=pypi&logoColor=white)](https://pypi.org/project/axio-tui/)

**Axio** (*Asynchronous eXtensible Intelligent Orchestration*) - a minimal but complete
foundation for building LLM-powered agents in Python.

Every integration point is a protocol - bring your own transport, context store,
tools, and permission guards.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Install the TUI or coding assistant and start chatting in under a minute.
:::

:::{grid-item-card} Quick Start
:link: quick-start
:link-type: doc

Write your first agent in code with the core library.
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
from axio import Tool


async def fetch(url: str) -> str:
    """Fetch the text content of a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            return (await r.text())[:2000]

fetch_tool = Tool(name="fetch", handler=fetch)
assert fetch_tool.name == "fetch"
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
from axio import Agent, MemoryContextStore
from axio_transport_openai import OpenAITransport


async def main() -> None:
    agent = Agent(
        system="You are a helpful assistant.",
        tools=[fetch_tool],
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

The name is a backronym - each letter describes what the framework actually does:

**A - Asynchronous.**
The agent loop is built on `asyncio` end-to-end. Tool calls from a single
LLM response are dispatched concurrently via `asyncio.gather` so results
arrive in parallel, not sequentially. Every transport, tool, and context
store uses `async def` throughout - no thread pools or blocking I/O hidden
beneath the surface.

**X - eXtensible.**
Every integration point is a runtime-checkable `Protocol` or abstract base class.
You can swap the transport (OpenAI, Anthropic, any custom endpoint), the context
store (in-memory, SQLite, your own database), the permission guards, and the tools
without touching a single line of framework code. The plugin system - based on
Python entry points - lets separate packages register transports, tools, and
guards that are discovered automatically at runtime. Extensible is capitalised
because it is the core design decision from which everything else follows.

**I - Intelligent.**
The LLM drives the decision loop. Axio stays out of the way: it presents tools
to the model, faithfully delivers tool results back, and keeps iterating until the
model decides it is done. No hard-coded routing, no fixed decision trees - the
intelligence lives in the model, not the framework.

**O - Orchestration.**
Axio coordinates agents, tools, context, and permissions into a coherent execution
flow. Sub-agents can be spawned and composed via the built-in `subagent` tool;
context stores are shared across agents; permission guards form a composable
chain that gates every tool call. Complex multi-agent workflows emerge from
simple, well-defined primitives.

Extensible by design
: Every integration point is a runtime-checkable `Protocol` or ABC.
  Swap transports, context stores, tools, and guards without touching framework code.

Streaming by default
: All LLM I/O flows through typed `StreamEvent` values.
  No hidden buffering - you see every token, tool call, and result as it happens.

Tools are plain async functions
: Define parameters as function arguments, get JSON schema for free, implement the body for execution.
  Guards gate every tool call through a composable permission chain.

Multi-agent orchestration built-in
: Spawn sub-agents via the `subagent` tool, share context between agents,
  compose complex workflows - all without external dependencies.

### Architecture

```{mermaid}
flowchart TB
    subgraph User["User Code"]
        A[Agent]
    end

    subgraph Core["axio - Core Framework"]
        B[Tool Handler<br/>async function]
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
1. You configure the agent with tools (async functions) and guards (permission chain)
2. User sends a message
3. Agent sends to transport → LLM
4. LLM responds with text or tool calls
5. Tools execute → results return → LLM generates final response
6. Events stream back to you (tokens, tool calls, results)

## How does Axio compare?

Here's how Axio compares to other popular Python agent frameworks:

| | Axio | pydantic-ai | LangChain / LangGraph | AutoGen |
|---|---|---|---|---|
| **Architecture** | Minimal core + protocols | Pydantic-native, validation-centric | Heavy abstraction layer | Multi-agent orchestration |
| **Streaming** | All events typed, full tool visibility | Text streaming works; tool calls and final answer can't stream simultaneously | Added later, inconsistent | Limited |
| **Tool definition** | Plain async function | Decorator + function signature → auto JSON schema | Functions + decorators | Class-based agents |
| **Transport** | Pluggable protocol; OpenAI (+ any OpenAI-compatible endpoint), Anthropic, Google Gemini, Codex, or custom | Built-in 20+ providers, trivial to swap | Built-in, harder to swap | Azure OpenAI focused |
| **Realtime / voice** | First-class via `RealtimeTransport` + `axio-audio` (Gemini Live, Vertex Live) | None | None | None |
| **Multimodal** | text, vision, audio, video input; image + video generation | text + vision | text + vision | text + vision |
| **Multi-agent** | Built-in (`subagent` tool, shared context stores) | Agent-as-tool pattern; not the primary focus | Via LangGraph | Native |
| **Learning curve** | Low - ~100 lines for an agent | Low for Pydantic/FastAPI users; moderate otherwise | Medium - many abstractions | High - complex configs |
| **Scope** | Agent loop + extensions + voice | Agent + structured outputs; no RAG, no built-in memory | Full stack (RAG, chains, etc.) | Multi-agent scenarios |
| **API stability** | Beta (v0.x, breaking changes possible) | Beta (v0.x, breaking changes possible) | Stable | Stable |

### When to choose each

**Choose Axio if:**
- You want full control over every step of the agent cycle - no hidden magic, no framework opinions baked in
- You care about a lean dependency tree: every component is a separate PyPI package - install only what you need, fewer dependencies means fewer supply-chain attack risks
- You need multimodal input - text, images, audio, and video are first-class content blocks throughout the whole stack
- You're building a realtime voice agent - `axio-audio` + `GeminiLiveTransport` give you sample-aligned duplex audio with production-grade echo cancellation, tool dispatch that doesn't block the audio stream, and graceful interruption handling
- You need image or video generation as part of the agent loop - Gemini Nano Banana and Veo are available as ordinary tools
- You need multi-agent orchestration - sub-agents, shared context stores, and composable permission chains are built in
- You need custom sandboxed execution - isolated Docker containers out of the box via [`axio-tools-docker`](guides/docker-sandbox.md)
- You want a ready-made terminal coding assistant - `axio-repl` ships with file/shell tools, streaming output, vision, and transport auto-detection
- You prefer explicit protocols over decorator-driven conventions and need to own the event loop, streaming pipeline, and permission model

**Choose pydantic-ai if:**
- You already use Pydantic/FastAPI and want the same patterns for agents
- Your agent must return strongly typed, validated structured outputs
- You need trivial provider swapping across 20+ LLM backends

**Choose LangChain if:**
- You need RAG, text splitters, and other built-in data utilities
- You want batteries-included with minimal wiring code
- You're prototyping quickly and can accept abstraction overhead

**Choose AutoGen if:**
- You're building complex multi-agent conversation flows with a heavy Azure OpenAI focus
- You need built-in support for human-in-the-loop at the framework level

Axio's philosophy is thin abstraction over the prompt-completion loop,
not a full framework with opinions about how you should structure your application.
If that aligns with your needs - welcome.

## Examples

The [`examples/`](https://github.com/mosquito/axio-agent/tree/master/examples)
directory contains runnable examples:

- [`minimal.py`](https://github.com/mosquito/axio-agent/blob/master/examples/minimal.py) -
  core agent loop with `StubTransport`; no API key needed
- [`stream_tool_args.py`](https://github.com/mosquito/axio-agent/blob/master/examples/stream_tool_args.py) -
  incremental streaming of tool call arguments with partial JSON decoding
- [`codex_chat.py`](https://github.com/mosquito/axio-agent/blob/master/examples/codex_chat.py) -
  CLI chat loop backed by a ChatGPT subscription via OAuth PKCE (no API key required)
- [`agent_swarm/`](https://github.com/mosquito/axio-agent/tree/master/examples/agent_swarm) -
  team of role-specialised agents; see the {doc}`guides/agent-swarm` guide
- [`gas_town/`](https://github.com/mosquito/axio-agent/tree/master/examples/gas_town) -
  multi-agent convoy following the Gas Town methodology; see the {doc}`guides/gas-town` guide
- [`realtime_smoke/`](https://github.com/mosquito/axio-agent/tree/master/examples/realtime_smoke) -
  minimal Gemini Live smoke test: send text, receive and play audio
- [`realtime_chat/`](https://github.com/mosquito/axio-agent/tree/master/examples/realtime_chat) -
  full-featured voice chat with echo cancellation, volume metering, and interruption handling

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
