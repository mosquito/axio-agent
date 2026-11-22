# Axio

[![PyPI](https://img.shields.io/pypi/v/axio)](https://pypi.org/project/axio/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-mosquito%2Faxio--agent-181717?logo=github&logoColor=white)](https://github.com/mosquito/axio-agent)

**Composable building blocks for building AI agents in Python.**

Most agent frameworks are monolithic: they wrap providers in heavy abstractions, inject hidden prompt templates, and dictate how your application loop must run.

**Axio takes the Unix approach.** It provides small, unopinionated, protocol-driven primitives that you assemble into your own agent harness:

- **Pluggable Transports**: Stream typed events from OpenAI, Anthropic, Gemini, or local models using a clean sans-I/O SSE engine.
- **Protocol-driven Storage**: Ephemeral memory, SQLite with transparent gzip compression, or your own backend.
- **Tool Dispatch & Argument Streaming**: Plain async Python functions become tools with automatic JSON Schema generation, concurrent execution, and streaming arguments.
- **Permission Guards**: Middleware pipeline for tool approval, path restriction, and Docker sandboxing.
- **Progressive Guidance**: Load instructions and skills (`AGENTS.md`, MCP) on demand instead of cluttering system prompts.
- **Full-Duplex Audio**: Primitives for real-time voice agents (Gemini Live, OpenAI Realtime) with echo cancellation.

---

## Quickstart: Assemble an Agent in 15 Lines

Install the core building blocks and a transport:

```bash
pip install axio axio-transport-openai
```

Combine a tool, a transport, and a context store into an agent:

```python
import asyncio
from axio import Agent, MemoryContextStore, TextDelta, Tool, ToolResult
from axio_transport_openai import OpenAITransport


# 1. Any async function is a tool
async def search_database(query: str) -> str:
    """Search records in the database."""
    return f"Results for '{query}': 3 records found."


# 2. Compose the agent
agent = Agent(
    system="You are an autonomous researcher.",
    transport=OpenAITransport(model="gpt-4o-mini"),
    tools=[Tool(name="search_database", handler=search_database)],
)


# 3. Stream typed events
async def main() -> None:
    context = MemoryContextStore()
    async for event in agent.run_stream("Find recent sales records", context):
        match event:
            case TextDelta(delta=chunk):
                print(chunk, end="", flush=True)
            case ToolResult(name=name, content=output):
                print(f"\n[{name} executed]: {output}\n")


asyncio.run(main())
```

---

## Composing the Building Blocks

Axio components snap together through standard runtime protocols:

### Persistent History with SQLite
Swap `MemoryContextStore()` with `SQLiteContextStore` to persist sessions and token usage across process restarts:

```python
from axio_context_sqlite import SQLiteContextStore, connect

conn = await connect("chat.db")
context = SQLiteContextStore(conn, session_id="user-session-42")
```

### Tool Permissions & Sandboxing
Protect sensitive tools with guards or run them entirely inside an isolated Docker container:

```python
from axio.permission_policy import PolicyGuard
from axio_tools_docker import DockerSandbox

# Guard: automatically allow project reads, prompt for writes or shell commands
guard = PolicyGuard(root="/my/project", prompt_fn=my_ui_approval_callback)

# Sandbox: execute all tool calls inside a throwaway container
async with DockerSandbox(image="python:3.12-alpine") as sandbox:
    agent = Agent(system="...", transport=transport, tools=sandbox.tools)
```

### Connect External Tools via MCP
Bridge any Model Context Protocol server over stdio or HTTP without changing your agent code:

```python
from axio_tools_mcp.provider import MCPToolProvider

provider = MCPToolProvider()
# Tools and instruction catalogs load dynamically from connected MCP servers
agent = Agent(system="...", transport=transport, providers=[provider])
```

### Realtime Voice Duplex
Build voice agents with bidirectional streaming:

```python
from axio.realtime import RealtimeAgent
from axio_audio import DuplexAudio
from axio_transport_google.realtime import GeminiLiveTransport

async with DuplexAudio() as audio, RealtimeAgent(transport=GeminiLiveTransport()) as voice_agent:
    # Concurrent mic stream, speaker playback, and background tool execution
    ...
```

---

## The Building Blocks Catalog

Every package in the workspace is an independent, focused building block:

| Category | Package | Description |
|---|---|---|
| **Core & Engine** | [`axio`](axio/) | Agent loop, protocol definitions, content blocks, and typed stream events |
| | [`axio-sse`](axio-sse/) | Zero-dependency sans-I/O `text/event-stream` state machine and reader |
| | [`axio-responses`](axio-responses/) | OpenAI Responses API request builders and event decoder |
| **Transports** | [`axio-transport-openai`](axio-transport-openai/) | OpenAI (`/v1/responses`, `/v1/chat`), ChatGPT OAuth, Nebius, OpenRouter, Speech & Realtime |
| | [`axio-transport-anthropic`](axio-transport-anthropic/) | Anthropic Claude (Direct API & Google Cloud Vertex AI) with prompt caching |
| | [`axio-transport-google`](axio-transport-google/) | Gemini Developer API, Vertex AI, Antigravity OAuth, image/video gen & Gemini Live |
| | [`axio-transport-codex`](axio-transport-codex/) | ChatGPT desktop OAuth transport over Responses API |
| **Tools & Sandboxes** | [`axio-tools-local`](axio-tools-local/) | Filesystem, shell execution, and Python execution tools |
| | [`axio-tools-mcp`](axio-tools-mcp/) | Model Context Protocol bridge (stdio and HTTP transports) |
| | [`axio-tools-docker`](axio-tools-docker/) | Isolated Docker container tool provider |
| **Context & Storage** | [`axio-context-sqlite`](axio-context-sqlite/) | Persistent SQLite context store with transparent gzip compression |
| **Media & Audio** | [`axio-audio`](axio-audio/) | Microphone, speaker, and duplex audio helpers for realtime sessions |

---

## Reference Applications

To demonstrate how these building blocks assemble into production tools, Axio ships two reference applications:

- **`axio-repl`**: An opinionated interactive terminal coding assistant that verifies its work by running tests and capturing multi-viewport screenshots.
  ```bash
  uvx --from "axio-repl[all]" axio-repl
  ```
- **`axio-tui`**: A full-screen Textual application with plugin discovery, role management, and SQLite history.
  ```bash
  uv tool install "axio-tui[all]"
  axio
  ```

Both applications support subscription-based OAuth (ChatGPT and Google Antigravity) without needing paid API keys.

---

## Documentation

Full documentation, architecture guides, and tutorials are available at [docs.axio-agent.com](https://docs.axio-agent.com):

- **[Tutorial: Build an Agent Harness](https://docs.axio-agent.com/agent-harness/from-chat-to-agent.html)** — Step-by-step path from an empty directory to a complete production harness.
- **[Concepts](https://docs.axio-agent.com/concepts/index.html)** — Protocols, messages, event streams, permissions, and compaction.
- **[Guides](https://docs.axio-agent.com/guides/index.html)** — Custom tools, transports, MCP integrations, Docker sandboxes, and realtime voice.
- **[Cookbook & Examples](https://docs.axio-agent.com/guides/cookbook.html)** — Multi-agent swarms (`examples/agent_swarm`), autonomous environments (`examples/gas_town`), and voice chat (`examples/realtime_chat`).

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, `uv` workflows, make targets, and code style guidelines.

---

## License

[MIT](LICENSE) © Dmitry Orlov
