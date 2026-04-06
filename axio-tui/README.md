# axio-tui

[![PyPI](https://img.shields.io/pypi/v/axio-tui)](https://pypi.org/project/axio-tui/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tui)](https://pypi.org/project/axio-tui/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Textual TUI application for [axio](https://github.com/axio-agent/axio).

A full-featured terminal chat interface with session management, a plugin system for transports and tools, and a built-in SQLite context store. Pick your LLM backend, load your tools, and start chatting — all from the terminal.

## Features

- **Plugin system** — transports, tools, and guards discovered automatically via entry points
- **Session management** — persistent SQLite-backed conversations; fork, switch, and resume sessions
- **Streaming UI** — text and tool calls rendered incrementally as they arrive
- **Multi-transport** — switch between Anthropic, OpenAI, Nebius, Codex, or any registered backend at runtime
- **Sub-agent support** — the `subagent` tool lets the agent spin up nested agent sessions
- **Vision** — `vision` tool for image analysis (with compatible models)
- **Serveable** — `textual-serve` support for browser-based access

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
| `mcp` | axio-tools-mcp |
| `rag` | axio-tui-rag |
| `guards` | axio-tui-guards |
| `all` | Everything above |

## Quick start

```bash
pip install "axio-tui[openai,local]"
axio
```

On first launch, open **Settings** (`s`) to configure your API key and model. Sessions are stored in `~/.local/share/axio/`.

## Architecture

```
axio-tui
├── App (Textual)
│   ├── ChatScreen        — message list, input, streaming
│   ├── SessionScreen     — session list and management
│   └── SettingsScreen    — per-plugin configuration
├── SQLiteContextStore    — persistent conversation history
├── TransportRegistry     — discovers axio.transport entry points
└── Plugin system
    ├── ToolsPlugin       — wraps axio.tools.settings providers
    └── PermissionGuard   — wraps axio.guards providers
```

## Built-in tools

These tools are always available regardless of installed plugins:

| Tool | Description |
|---|---|
| `confirm` | Ask the user a yes/no question (for guard prompts) |
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
path = "axio_tui_guards.guards:PathGuard"
```

## Serve over HTTP

```bash
textual-serve axio_tui.__main__:app
# Open http://localhost:8000 in your browser
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-transport-anthropic](https://github.com/axio-agent/axio-transport-anthropic) · [axio-tui-rag](https://github.com/axio-agent/axio-tui-rag) · [axio-tui-guards](https://github.com/axio-agent/axio-tui-guards) · [axio-tools-local](https://github.com/axio-agent/axio-tools-local) · [axio-tools-mcp](https://github.com/axio-agent/axio-tools-mcp) · [axio-tools-docker](https://github.com/axio-agent/axio-tools-docker)

## License

MIT
