# MCP Server Tools

`axio-tools-mcp` lets you plug any
[MCP (Model Context Protocol)](https://modelcontextprotocol.io) server into an
Axio agent. Tools are loaded at runtime from the server and wrapped as regular
`Tool` objects. The agent has no idea they came from MCP.

## Install

```bash
pip install axio-tools-mcp
```

## Keep tools, instructions, and sessions together

Use `MCPToolProvider` when an agent should receive both server tools and server
guidance, skills, or resources. A provider keeps its connections alive and
supplies the current catalogs on every agent iteration. Instructions initially
expose summaries; their full content and associated tools load when needed.
See {doc}`../concepts/instructions` for how the model uses that catalog.

Applications choose configuration files and decode JSON. Pass validated server
entries to the provider, then keep it open while the agent runs:

<!-- name: test_mcp_provider_agent -->
```python
from axio import Agent, CompletionTransport, MemoryContextStore
from axio.plugins import open_tool_providers
from axio_tools_mcp import MCPServerConfig
from axio_tools_mcp.provider import MCPToolProvider


async def ask_with_servers(
    transport: CompletionTransport,
    servers: tuple[MCPServerConfig, ...],
    prompt: str,
) -> str:
    provider = MCPToolProvider()
    async with open_tool_providers([provider], config=servers):
        agent = Agent(
            system="Help using the available server tools and guidance.",
            transport=transport,
            providers=[provider],
        )
        return await agent.run(prompt, MemoryContextStore())
```

<!--
name: test_mcp_provider_agent
```python
import asyncio
from axio.testing import StubTransport, make_text_response

assert asyncio.run(ask_with_servers(StubTransport([make_text_response("Ready")]), (), "Hello")) == "Ready"
```
-->

Create entries with `MCPServerConfig.from_dict(name, decoded_entry)` or the
constructor examples below. Supply an open completion transport as shown in
{doc}`choosing-transports`. MCP does not search for project configuration files;
that policy belongs to the application. REPL and TUI provide their own layered
configuration, described in {doc}`axio-repl`.

After configuration changes, `await provider.reload(servers)` replaces the server
sessions and catalogs. Stop active runs before reloading so they cannot use
closed sessions. Connection failures are available through
`provider.inspect_servers()`; healthy servers remain available. Keep shared
providers open until every agent using their tools has finished.

## Load a tool snapshot directly

`load_mcp_tools` connects to one or more servers and returns a flat list of
tools plus the open sessions:

<!--
name: test_mcp_load_tools
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
axio_transport_anthropic.AnthropicTransport = lambda: StubTransport(
    [make_text_response("README.md and src/")]
)
```
-->
<!-- name: test_mcp_load_tools -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio_transport_anthropic import AnthropicTransport
from axio_tools_mcp import load_mcp_tools, MCPServerConfig


async def main() -> None:
    servers = [
        MCPServerConfig(
            name="fs", command="mcp-server-filesystem", args=["--root", "."]
        ),
    ]
    tools, sessions = await load_mcp_tools(servers)
    try:
        agent = Agent(
            system="You are a helpful assistant.",
            tools=tools,
            transport=AnthropicTransport(),
        )
        reply = await agent.run("List the files here.", MemoryContextStore())
        print(reply)
    finally:
        for session in sessions:
            await session.close()


asyncio.run(main())
```

Sessions must be closed when you're done. Use `try/finally` as above, or an
`AsyncExitStack` for cleaner lifecycle management.

## Server configuration

`MCPServerConfig` supports two transport types: **stdio** (local subprocess)
and **HTTP**.

### Stdio (local subprocess)

```python
MCPServerConfig(
    name="filesystem",
    command="mcp-server-filesystem",
    args=["--root", "/home/user/project"],
    env={"MY_VAR": "value"},  # optional extra environment variables
)
```

| Field | Description |
|---|---|
| `name` | Server identifier - used as tool name prefix |
| `command` | Executable to run |
| `args` | Arguments passed to the command |
| `env` | Extra environment variables (merged with the current environment) |
| `timeout` | Handshake timeout in seconds (default: `30.0`) |
| `protocol_version` | Pin one MCP revision (default: negotiated, see below) |

### HTTP

```python
MCPServerConfig(
    name="remote",
    url="http://mcp-server.internal:3000/mcp",
    headers={"Authorization": "Bearer my-token"},
)
```

| Field | Description |
|---|---|
| `name` | Server identifier - used as tool name prefix |
| `url` | HTTP endpoint URL |
| `headers` | HTTP headers sent with every request |
| `timeout` | Connect and read timeout in seconds (default: `30.0`) |
| `protocol_version` | Pin one MCP revision (default: negotiated, see below) |

## Protocol revisions

The transport is [aiohttp-tiny-mcp](https://github.com/mosquito/aiohttp-tiny-mcp),
which speaks `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25` and
`2026-07-28`.

The handshake starts on `2025-06-18`, because every current server answers it.
A server that answers with another revision selects it, and a server that
refuses and names what it supports gets a second handshake on the newest
revision both sides speak. Set `protocol_version` to pin one revision instead;
a pinned revision is never replaced, and a server that refuses it fails to
connect.

A server that pages its tool list is read to the end, so every page becomes
tools.

## Tool naming

Tools are prefixed with the server name and a double underscore:

```
{server_name}__{tool_name}
```

For example, a `read_file` tool from a server named `fs` becomes `fs__read_file`.
This prevents name collisions when multiple servers expose tools with the same name.

## Multiple servers

Pass multiple configs to `load_mcp_tools` - tools from all servers are merged
into a single flat list:

```python
tools, sessions = await load_mcp_tools(
    [
        MCPServerConfig(
            name="fs", command="mcp-server-filesystem", args=["--root", "."]
        ),
        MCPServerConfig(name="git", command="mcp-server-git"),
        MCPServerConfig(name="web", url="http://localhost:4000/mcp"),
    ]
)
```

## Error handling

`load_mcp_tools` logs a server that fails to connect and skips it, so the other
servers still load. Connect sessions individually to see the failure itself:

```python
from axio_tools_mcp import MCPSession, MCPServerConfig

session = MCPSession(MCPServerConfig(name="fs", command="mcp-server-filesystem"))
try:
    await session.connect()
    tools = await session.list_tools()
except Exception as exc:
    print(f"Server unavailable: {exc}")
    tools = []
```

A tool that fails on the server returns an error result, and the Axio tool
raises `RuntimeError` with the text the server sent.
