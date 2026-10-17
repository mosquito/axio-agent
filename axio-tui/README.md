# axio-tui

[![PyPI](https://img.shields.io/pypi/v/axio-tui)](https://pypi.org/project/axio-tui/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tui)](https://pypi.org/project/axio-tui/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Textual TUI application for [axio](https://github.com/mosquito/axio-agent).

A full-featured terminal chat interface with session management, a plugin system for transports and tools, and a built-in SQLite context store. Pick your LLM backend, load your tools, and start chatting - all from the terminal.

## Features

- **Plugin system** - transports, tools, and guards discovered automatically via entry points
- **Session management** - persistent SQLite-backed conversations; fork, switch, and resume sessions
- **Streaming UI** - text and tool calls rendered incrementally as they arrive
- **Multi-transport** - switch between Anthropic, OpenAI, Nebius, Codex, or any registered backend at runtime
- **Sub-agent support** - the `subagent` tool lets the agent spin up nested agent sessions
- **Vision** - `vision` tool for image analysis (with compatible models)
- **Serveable** - `textual-serve` support for browser-based access

## Installation

Minimal (core TUI only, bring your own transport):

```bash
pip install axio-tui
```

With everything:

```bash
pip install "axio-tui[all]"
```

Pick what you need:

```bash
pip install "axio-tui[openai,local,mcp]"
```

| Extra | Installs |
|---|---|
| `anthropic` | axio-transport-anthropic |
| `openai` | axio-transport-openai |
| `codex` | axio-transport-codex |
| `local` | axio-tools-local |
| `history` | axio-context-sqlite |
| `mcp` | axio-tools-mcp |
| `guards` | axio-tui-guards |
| `all` | Everything above |

## Quick start

```bash
pip install "axio-tui[openai,local]"
axio
```

On first launch, press **Ctrl+P** to configure a transport and choose a model. Sessions and UI settings are stored in `~/.local/share/axio-tui/axio.db`.

## Workspace instructions, skills and MCP

TUI and REPL compose providers from `axio-tools-local` and `axio-tools-mcp`. Both load global and project `AGENTS.md`,
local Agent Skills, MCP tools, MCP skills, and ordinary MCP resources. New scoped instructions
reach the model before the affected tool batch runs. Subagents keep separate instruction
snapshots and retain the same tools and resource schemas.

Start `axio` from the working directory. The nearest Git root defines the project boundary;
without Git, the working directory is the root. Use `--project-root PATH` to override it
and repeat `--additional-root PATH` for independent roots.

MCP configuration loads in this order, replacing each server entry as a whole:

1. `$XDG_CONFIG_HOME/axio/mcp.json` (default `~/.config/axio/mcp.json`).
2. `<root>/.mcp.json`.
3. `<root>/.axio/mcp.local.json`.
4. Explicit `--mcp-config PATH` files.

Enabled servers connect automatically. A failed connection does not disable other servers.
Manual servers in **Manage MCP Servers** remain available alongside file configuration;
use distinct server names to avoid tool-name collisions. Manual servers also expose skills
and resources. Resource URIs, including `skill://`, are passed unchanged to their server.

Local skills come from the same home directories as REPL. Add explicit locations with
`--skills-dir PATH`. Skill bodies load only when the agent reads them or you open them in the skills window. Use `--instructions-file PATH`
for explicit instructions. `--isolated` disables automatic instructions, home skills,
file MCP configuration, and saved manual MCP servers; explicit paths still load.

Use these commands in the input field or their equivalents in **Ctrl+P**:

| Command | Result |
|---|---|
| `/skills` | Open the local and MCP skills browser |
| `/mcp` | Open the detailed MCP server browser |
| `/permissions` | Inspect and revoke chat or project permission rules |
| `/instructions` or `/config` | Show instruction sources and redacted MCP configuration |
| `/config reload` | Reload instructions, skills, and automatic MCP connections |

The skills window shows full descriptions and source paths or URIs. Filter by name,
description, or source, then press **Enter** or **Open skill** to read the instructions.
Copies with identical full `SKILL.md` content share one entry, with their paths or
URIs separated by commas. Filtering matches every path. Different content stays
separate, even when names and descriptions match. Local files use content hashes;
MCP catalogs supply hashes that are verified when the skill is opened.
Use **Tab** to move between the list and details, **Ctrl+F** to filter, and **Esc** to close.
Automatic and manual MCP connections have separate labels. The layout stacks vertically
in narrow terminals. Read failures appear inside the window and can be retried.

The MCP window includes automatic and saved connections. Expand a server to browse
its tools, resources, URI templates, and skills. Select an item to see its full
description, input schema, or resource metadata. Server details include configuration
sources, connection state, protocol, capabilities, and discovery errors. Connection
secrets remain masked. Filter by name or metadata, use **Tab** to scroll the details,
and press **Esc** to close. **Refresh** reads the current connection state and
cached catalogs; it does not reconnect servers or execute tools.

Open **Manage** in `/mcp` to add, edit, rename, enable, disable, or delete file-based
servers. Choose **User (~)** for `~/.config/axio/mcp.json` (or the XDG path),
**Project (.)** for `<root>/.mcp.json`, or **private** for `<root>/.axio/mcp.local.json`.
Additional roots and explicit configuration files appear with their full paths.
The editor preserves other entries and rejects edits if the file changed while open.
Deleting an override can expose a lower-priority entry. Changes reconnect MCP when
you close the manager. Isolated mode still loads only explicit files.

For HTTP authorization, select **Authorization header** and enter the complete value
(for example, `Bearer ...`), or select **OAuth (browser sign-in)**. Save the entry,
select it in the manager, and click **Authorize**. OAuth uses server discovery and
dynamic client registration by default. Pre-registered public clients can specify
an issuer, client ID, scopes, and callback port. The callback is
`http://127.0.0.1:<port>/callback`. A zero port selects a free port on first sign-in;
later sign-ins reuse the registered port. Open the displayed URL manually if the
browser does not start. Closing the authorization window cancels pending sign-in.

OAuth credentials stay in `$XDG_CONFIG_HOME/axio/mcp.auth`, separate from MCP JSON
and the settings database. Connections refresh saved tokens without opening a browser.
**Sign out** removes local tokens and keeps the client registration; it does not revoke
access at the authorization server. Configurations with the same endpoint, issuer,
client ID, and fixed callback port share this local authorization. Header credentials are saved in the
selected JSON file with mode `0600`; use environment references or a private file
when the project configuration is committed. The **Additional options** field accepts
`headersFrom`, `envFrom`, environment values, timeout, and other supported JSON options.

Use **Reconnect all**, `/mcp reconnect` (alias `/mcp reload`), or **Reconnect MCP** in
the command palette to reconnect both automatic and saved servers. The window closes
and progress appears in the chat. Tools and catalogs are updated after reconnection.
Initialization has one deadline per server, covering the handshake and all catalogs.
The default is 30 seconds. Set `timeout` in the server's JSON configuration, or edit
**Initialization timeout (seconds)** in **Manage MCP Servers** for a saved server. Progress
and errors show the server and its limit. Other servers still connect after a failure.
Stopping a stdio process can take up to five additional seconds before it is killed.

`/skills reload` and `/instructions reload` reload the workspace.
Stop an active run before reconnecting MCP or changing accounts. Shell and MCP tools can
access paths outside the workspace; instruction scopes do not provide filesystem isolation.

## Tool permissions

Read-only file operations inside the project root run without a prompt. Reading
outside the root, writing files, and operations with unknown effects require a
matching rule or confirmation. Explicit denials take priority over automatic reads
and allow rules. The root is the discovered project root or `--project-root`, not
necessarily the shell's current directory. Additional workspace roots do not receive
automatic permission.

The dialog shows the tool, arguments, resolved resources, and reason for asking.
Choose **This call only**, **Until this chat ends**, or **Save for this project**,
then allow or deny. File rules cover the displayed operations and exact paths;
selecting parent directories explicitly includes their descendants. Reading never
grants permission to write. Shell, Python, MCP, and other opaque calls use exact
tool/argument approvals, not command prefixes or model-generated categories.
Changing an MCP server configuration invalidates its previous exact-call approvals.

Open **Ctrl+P → Permissions** or `/permissions` to inspect and revoke rules.
Project rules persist in the TUI configuration database under the canonical project
root. Chat rules expire when you start, restore, or fork another chat; compaction
keeps them. Exact-call rules store a fingerprint instead of raw arguments.
Revocation affects subsequent calls, not tools that are already running.

Every agent tool, including MCP and subagent tools, passes through the policy after
its original guards. Concurrent prompts form a queue and recheck rules before asking.
Cancellation closes the pending dialog without granting permission. Rules also work
without `axio-tui-guards`. The old PathGuard configuration no longer controls TUI
permissions. An optional LLM reviewer supplies advice, never an automatic grant.

Tool providers declare their effects with `Tool(access=...)` or a handler decorator:

<!-- name: test_readme_tool_access -->
```python
from pathlib import Path
from axio.tool import Tool
from axio.tool_access import file_access, tool_access


@tool_access(file_access("read", "filename"))
async def read_text(filename: str) -> str:
    return Path(filename).read_text()


tool = Tool(name="read_text", handler=read_text)
```

A custom descriptor returns every `Access(operation, path)` the handler can perform.
Paths are relative to the process working directory, as for the local file tools.
Recursive tools must declare symlink targets they follow. An absent declaration means
unknown effects; an explicit empty tuple declares no external effects. The host trusts
installed provider code, not claims in model arguments or MCP tool descriptions.

These are pre-execution checks, not an OS sandbox. Paths are resolved again after
confirmation, but another process can still change the filesystem during execution.
Running shell or Python code in the project directory does not restrict its effects
to that directory.

## ChatGPT accounts

Install `axio-tui[openai]`, open **Ctrl+P → Configure ChatGPT**, and select
**Continue with ChatGPT**. The screen supports new accounts, returning sign-in,
registration selection, plan-permission status, sign-out, and a **ChatGPT Usage** link.
Over SSH, copy the authorization URL and paste the complete browser redirect URL into
the screen. Closing the screen cancels an unfinished sign-in.

Credentials use the same protected `$XDG_CONFIG_HOME/axio/chatgpt.auth` file as
`axio-chatgpt` and REPL. SQLite stores the selected registration reference, never its
tokens. Model choices reload for the selected account. Model copies share its token
session, and existing copies retain their original account. Account settings remain
accessible when signed out or model discovery fails. API-key and Codex settings remain separate.

## Architecture

```
axio-tui
├── App (Textual)
│   ├── ChatScreen        - message list, input, streaming
│   ├── SessionScreen     - session list and management
│   └── SettingsScreen    - per-plugin configuration
├── SQLiteContextStore    - persistent conversation history
├── TransportRegistry     - discovers axio.transport entry points
└── Plugin system
    ├── ToolsPlugin       - wraps axio.tools.settings providers
    └── PermissionGuard   - wraps axio.guards providers
```

## Built-in tools

These tools are always available regardless of installed plugins:

| Tool | Description |
|---|---|
| `confirm` | Submit an LLM review verdict; does not grant permission |
| `status_line` | Update the TUI status bar from within the agent |
| `subagent` | Spawn a nested agent with its own tools and context |
| `vision` | Analyse an image file (requires a vision-capable model) |

## Plugin entry points

`axio-tui` discovers plugins automatically when installed packages declare:

```toml
# Transport backend
[project.entry-points."axio.transport"]
openai = "axio_transport_openai:OpenAITransport"

# Tool group with settings screen
[project.entry-points."axio.tools.settings"]
docker = "axio_tools_docker.plugin:DockerPlugin"

# Simple tools (no settings)
[project.entry-points."axio.tools"]
shell = "axio_tools_local.shell:Shell"

# Permission guards
[project.entry-points."axio.guards"]
llm = "axio_tui_guards.guards:LLMGuard"
```

## Serve over HTTP

```bash
axio --serve --web-listen localhost:8086
# Open http://localhost:8086 in your browser
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-transport-anthropic](https://github.com/mosquito/axio-agent) · [axio-tui-guards](https://github.com/mosquito/axio-agent) · [axio-tools-local](https://github.com/mosquito/axio-agent) · [axio-tools-mcp](https://github.com/mosquito/axio-agent) · [axio-tools-docker](https://github.com/mosquito/axio-agent)

## License

MIT

Both applications discover `axio.tools.providers` entry points. They pass initialized
providers to `Agent`, which handles instruction activation and associated tools.
TUI settings and permission guards also apply to tools added by an instruction.
The TUI's own tools use a provider enabled by `application="tui"`; installing the
TUI does not expose its UI callbacks in the REPL.


## Optional integrations

The base installation starts without tools, transports, or persistent history.
Install `axio-tui[local]` for local files and instructions, `axio-tui[mcp]` for
MCP, and `axio-tui[history]` for SQLite conversations. These extras are independent.
Transport extras and `guards` remain optional; `all` includes `history`.

Without a context backend, the interface labels history as temporary. New chats,
forks, and compaction work in memory; previous chats cannot be restored after
restart. TUI configuration still uses its own SQLite database.

Tools and instruction providers use `axio.tools.providers`; plain tools use
`axio.tools`. Refreshable providers implement `refresh(*, config=None)`.
Context backends register zero-argument factories in `axio.context`, implementing
`axio.plugins.ContextBackend`. The application initializes the first available
backend and closes it at shutdown. Failed providers report a diagnostic without
preventing independent providers from starting.

Without a transport, configuration and inspection remain available. Sending a
request requires configuring a transport. Unavailable saved models remain in
configuration until the user explicitly selects another model.
