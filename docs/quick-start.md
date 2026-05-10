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

## MCP server tools

[MCP (Model Context Protocol)](https://modelcontextprotocol.io) servers expose
tools over a standard interface. `load_mcp_tools` connects to one or more
servers and returns regular `Tool` objects ready to pass to `Agent`:

<!--
name: test_mcp_example
```python
import axio_tools_mcp
import axio_transport_anthropic
from axio import Tool
from axio.testing import StubTransport, make_text_response

async def _list_files() -> str:
    """List files in the current directory."""
    return "README.md\nsrc/"

async def _fake_load(servers):
    return [Tool(name="fs__list_files", handler=_list_files)], []

axio_tools_mcp.load_mcp_tools = _fake_load
axio_transport_anthropic.AnthropicTransport = lambda: StubTransport([make_text_response("README.md and src/")])
```
-->
<!-- name: test_mcp_example -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio_transport_anthropic import AnthropicTransport
from axio_tools_mcp import load_mcp_tools, MCPServerConfig


async def main() -> None:
    servers = [
        MCPServerConfig(name="fs", command="mcp-server-filesystem", args=["--root", "."]),
        MCPServerConfig(name="web", url="http://localhost:3000/mcp"),
    ]
    tools, sessions = await load_mcp_tools(servers)
    try:
        agent = Agent(
            system="You are a helpful assistant.",
            tools=tools,
            transport=AnthropicTransport(),
        )
        reply = await agent.run("List files in the current directory.", MemoryContextStore())
        print(reply)
    finally:
        for session in sessions:
            await session.close()


asyncio.run(main())
```

Tool names are prefixed with the server name: `fs__read_file`, `web__search`,
etc. Sessions must be closed when done - use `try/finally` or an
`AsyncExitStack`.

See the {doc}`guides/mcp-tools` guide for configuration options and the TUI
integration.

## Multimodal input

Send images, audio, or video by appending a `Message` with the appropriate
content blocks to the context before calling `agent.run()`:

<!--
name: test_multimodal_example
```python
import builtins as _b
import io
import axio_transport_anthropic
from axio.testing import StubTransport, make_text_response

_real_open = _b.open
_b.open = lambda p, m="r", **kw: (
    io.BytesIO(b"fake_png") if "screenshot.png" in str(p) and "b" in m
    else _real_open(p, m, **kw)
)
axio_transport_anthropic.AnthropicTransport = lambda: StubTransport([make_text_response("A terminal window.")])
```
-->
<!-- name: test_multimodal_example -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio.messages import Message
from axio.blocks import TextBlock, ImageBlock
from axio_transport_anthropic import AnthropicTransport


async def main() -> None:
    image_data = open("screenshot.png", "rb").read()

    context = MemoryContextStore()
    await context.append(Message(
        role="user",
        content=[
            TextBlock(text="What is shown in this screenshot?"),
            ImageBlock(media_type="image/png", data=image_data),
        ],
    ))

    agent = Agent(
        system="You are a helpful visual assistant.",
        tools=[],
        transport=AnthropicTransport(),
    )
    reply = await agent.run("Describe it in detail.", context)
    print(reply)


asyncio.run(main())
```

Tools can also return multimodal blocks - `read_file` from `axio-tools-local`
does this automatically for image, audio, and video files.

See the {doc}`guides/multimodal` guide for all supported formats and patterns.

## Docker sandbox

`DockerSandbox` spins up an isolated container and exposes the same
`shell`, `write_file`, `read_file`, `list_files`, `run_python`, and
`patch_file` tools as `axio-tools-local` - but every operation runs inside
the container, not on the host:

<!--
name: test_docker_example
```python
import axio_transport_anthropic
from axio.testing import StubTransport, make_text_response
axio_transport_anthropic.AnthropicTransport = lambda: StubTransport([make_text_response("Done.")])
```
-->
<!-- name: test_docker_example; fixtures: docker -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio_transport_anthropic import AnthropicTransport
from axio_tools_docker import DockerSandbox


async def main() -> None:
    async with DockerSandbox(image="python:3.12-slim") as sandbox:
        agent = Agent(
            system="You are a coding assistant. Use the sandbox tools.",
            tools=sandbox.tools,
            transport=AnthropicTransport(),
        )
        reply = await agent.run(
            "Write and run a Python script that prints the Fibonacci sequence.",
            MemoryContextStore(),
        )
        print(reply)


asyncio.run(main())
```

The container is created on entry and force-removed on exit - even if the body
raises. Switching from local to sandboxed execution requires changing only the
`tools=` argument.

See the {doc}`guides/docker-sandbox` guide for isolation options, named volumes,
and resource limits.

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
