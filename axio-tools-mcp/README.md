# axio-tools-mcp

[![PyPI](https://img.shields.io/pypi/v/axio-tools-mcp)](https://pypi.org/project/axio-tools-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-mcp)](https://pypi.org/project/axio-tools-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/mosquito/axio-agent/blob/main/LICENSE)

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) bridge for [axio](https://github.com/mosquito/axio-agent).

Connect any MCP server to your axio agent. Tools exposed by MCP servers are discovered at runtime and become first-class axio `Tool` instances - no manual wiring required.

## Features

- **Dynamic tool discovery** - connects to MCP servers and converts their tools into axio `Tool` instances automatically
- **Five protocol revisions** - the revision is negotiated with the server, or pinned per server
- **Multiple servers** - configure and run several MCP servers simultaneously
- **Lifecycle management** - handles server startup, shutdown, and reconnection
- **TUI integration** - ships a settings screen for managing MCP server configuration from within `axio-tui`

This release uses the tested `aiohttp-tiny-mcp` 0.3 API.

## Installation

```bash
pip install axio-tools-mcp
```

## Usage

### With axio-tui (recommended)

```bash
pip install "axio-tui[mcp]"
uv run axio   # MCP Servers section appears in settings
```

### Standalone

<!--
name: test_readme_standalone
```python
from axio.testing import StubTransport, make_text_response

transport = StubTransport([make_text_response("done")])
```
-->
<!-- name: test_readme_standalone -->
```python
import asyncio
from axio_tools_mcp.registry import MCPRegistry
from axio.agent import Agent
from axio.context import MemoryContextStore


async def main() -> None:
    registry = MCPRegistry()
    await registry.init(config=None)

    tools = registry.all_tools  # list[axio.Tool]
    print(f"Loaded {len(tools)} tools from MCP servers")

    # Pass any CompletionTransport - e.g. OpenAITransport, AnthropicTransport
    agent = Agent(
        system="You are a helpful assistant.",
        tools=tools,
        transport=transport,
    )
    result = await agent.run("Use the available tools to help me", MemoryContextStore())
    print(result)

    await registry.close()


asyncio.run(main())
```

## Transport types

Two transport types are supported, selected by which field is set in the server config:

| Transport | Config field | Protocol |
|-----------|-------------|---------|
| **stdio** | `command` | Spawns a subprocess; communicates over stdin/stdout (MCP stdio transport) |
| **HTTP** | `url` | Connects to a running HTTP server using the MCP Streamable HTTP transport (`aiohttp`) |

Exactly one of `command` or `url` must be set per server - providing both or neither raises a `ValueError`.

For stdio servers, stderr output from the subprocess is forwarded to the Python logger as warnings under the `mcp:<server-name>` prefix.

HTTP servers accept optional `headers` (e.g., for bearer tokens).
Both transports accept a positive, finite `timeout` in seconds (default: 30).
REPL and TUI apply it to the whole initialization: connection, handshake, and catalog discovery.
A timeout closes the connection and reports the server and limit. Stopping a stdio
process can take up to five additional seconds before it is killed.

## Protocol revisions

The client is [aiohttp-tiny-mcp](https://github.com/mosquito/aiohttp-tiny-mcp), which speaks `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25` and `2026-07-28`.

The handshake starts on `2025-06-18`, because every current server answers it. A server that answers with another revision selects it, and a server that refuses and names what it supports gets a second handshake on the newest revision both sides speak. Set `protocol_version` on the server config to pin one revision; a pinned revision is never replaced.

## Tool naming

Tools from MCP servers are named using the pattern `<server_name>__<tool_name>`. For example, a server named `filesystem` that exposes a tool called `read_file` becomes `filesystem__read_file` in axio. The tool description is taken from the MCP tool's `description` field, falling back to the tool name if no description is provided.

## Error handling

When an MCP server fails to connect or start, the error is logged at `ERROR` level and the server is skipped - no exception is raised to the caller. `registry.all_tools` will simply not include any tools from the failed server. The error message is accessible via `registry.server_status(name)` (returns `"error"`) and the raw error string is stored internally.

## Lifecycle: `close()`

Call `await registry.close()` when the agent session ends to disconnect all MCP server sessions and release resources (subprocess file descriptors, HTTP connections):

## MCP server configuration

Applications read JSON and pass each server entry to `MCPServerConfig.from_dict(name, data)`.
The method validates native JSON values without reading files or opening connections.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "cwd": "/tmp"
    },
    "remote": {
      "url": "https://my-mcp-server.example.com/mcp",
      "headersFrom": {"Authorization": "MCP_AUTHORIZATION"},
      "timeout": 60.0
    }
  }
}
```

<!-- name: test_native_configuration -->
```python
import json
from axio_tools_mcp import MCPServerConfig

# The application owns the file read and JSON decoding.
document = json.loads('{"mcpServers":{"remote":{"url":"https://example.com/mcp"}}}')
servers = tuple(MCPServerConfig.from_dict(name, data) for name, data in document["mcpServers"].items())
assert servers[0].transport == "http"
assert servers[0].to_dict()["url"] == "https://example.com/mcp"
```

`await server.connect()` resolves environment references and opens the appropriate
`MCPSession`. Close the returned session with `await session.close()`. Failed or
cancelled connection attempts close their partial resources before propagating the error.
`envFrom` and `headersFrom` name environment variables; their values are resolved
for each connection attempt. `to_dict()` preserves these references.

Applications choose configuration files, combine layers, assign unique server names,
and resolve paths relative to each source file. MCP has no project roots or file
search policy. `cwd` specifies only the working directory of a stdio subprocess.
Direct configurations inherit the process environment by default. Set
`inherit_env=False` to use the documented basic environment allowlist plus explicit
`env` and `envFrom` values. REPL/TUI use this restricted environment for file entries.

The legacy TUI database stores strings. Its codec uses `from_settings()` and
`to_settings()`; these are separate from the native JSON codec.

## Plugin registration

```toml
[project.entry-points."axio.tools.settings"]
mcp = "axio_tools_mcp.plugin:MCPPlugin"
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-tools-local](https://github.com/mosquito/axio-agent) · [axio-tools-docker](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT

## Instructions and caller-supplied configuration

`MCPToolProvider` implements `axio.tools.providers`. Its `init(config=servers)`
accepts a sequence of `MCPServerConfig` objects and connects enabled servers.
Applications can also pass `{"mcp": servers}` when initializing several providers.
Use `Agent(..., providers=[provider])` while the provider is open. The agent reads
its current tools and instructions at each model iteration.

`get_instructions()` returns general `Instruction` objects for server guidance,
skills, and resource catalogs. Their short descriptions do not load full skills.
`await instruction.load()` returns the full text and its session's tools. Skill
and resource URIs remain unchanged. Resource tools work independently of skills.
Reload and close invalidate old sessions; retain the provider while its tools are
in use. A failed server does not remove tools from healthy servers.

## OAuth authorization

HTTP entries can use `"oauth": {}` instead of an `Authorization` header. The
`aiohttp-tiny-mcp~=0.5.8` client handles discovery, PKCE, registration, and refresh.
Axio stores its credentials through the core credential store in
`$XDG_CONFIG_HOME/axio/mcp.auth` (default `~/.config/axio/mcp.auth`). Token operations
are coordinated across sessions and processes. Startup never opens a browser;
missing authorization leaves that server unavailable until explicit sign-in.

The TUI exposes sign-in and local sign-out through `/mcp` → **Manage** → **Authorize**.
Optional OAuth fields are `issuer`, `client_id`, `scope` (space-separated), and
`redirect_port`. A static client ID requires its registered issuer. The callback
uses `http://127.0.0.1:<port>/callback`. Dynamic registration selects a free port
unless configured and retains that port for later sign-ins. HTTPS is required for
OAuth endpoints, with loopback HTTP allowed for local servers.

The MCP browser return page uses Axio's owl, light and dark themes, a five-second
close countdown, and manual-close fallback. It reports success only after token
exchange and credential persistence succeed. Error pages contain fixed text without
server response details or credentials.
