# axio-tools-docker

[![PyPI](https://img.shields.io/pypi/v/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Docker sandbox tools for [axio](https://github.com/axio-agent/axio).

Run agent-generated code and commands inside isolated Docker containers. The agent gets `sandbox_exec`, `sandbox_write`, and `sandbox_read` tools that operate entirely within the sandbox — the host filesystem stays untouched.

## Features

- **Isolated execution** — code runs inside a Docker container, not on the host
- **Configurable image** — use any Docker image as the sandbox environment
- **Three sandboxed tools** — execute commands, write files, read files — all inside the container
- **Persistent sandbox** — container is reused across tool calls within a session for faster execution
- **TUI integration** — configure image, memory limits, and CPU from the `axio-tui` settings screen

## Requirements

Docker must be installed and running:

```bash
docker info   # should succeed
```

## Installation

```bash
pip install axio-tools-docker
```

## Usage

<!-- name: test_readme_usage; mark: skip -->
```python
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport
from axio_tools_docker.plugin import DockerPlugin

async def main() -> None:
    plugin = DockerPlugin()
    await plugin.init()   # uses default config (python:3.12-slim)

    agent = Agent(
        system=(
            "You are a coding assistant. Use sandbox_exec to run code safely. "
            "Never attempt to access the host filesystem directly."
        ),
        tools=plugin.all_tools,
        transport=OpenAITransport(api_key="sk-...", model="gpt-4o"),
    )

    ctx = MemoryContextStore()
    result = await agent.run(
        "Write a Python script that computes the first 20 Fibonacci numbers and run it.",
        ctx,
    )
    print(result)
```

## Sandbox tools

| Tool | Description |
|---|---|
| `sandbox_exec` | Run a shell command inside the container; returns stdout + stderr |
| `sandbox_write` | Write a file into the container's filesystem |
| `sandbox_read` | Read a file from the container's filesystem |

## Container lifecycle

The sandbox container is created **lazily** on the first tool call (`sandbox_exec`,
`sandbox_write`, or `sandbox_read`) — `plugin.init()` itself does not start Docker.
The same container is reused for all subsequent tool calls within the session.

When `await plugin.close()` is called the container is stopped and removed
(`docker rm -f`). You should always call `close()` when the agent session ends
to avoid leaving containers behind:

```python
async def run_with_cleanup(agent, ctx, plugin):
    try:
        result = await agent.run("...", ctx)
    finally:
        await plugin.close()
```

If Docker is not installed or the `docker` CLI is not on `PATH`, the container
creation step will raise a `RuntimeError` on the first tool call. You can detect
this before starting a session with `SandboxManager.docker_available()` (returns
`True` if the `docker` executable is found on `PATH`).

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `str` | `"python:latest"` | Docker image to use for the sandbox container |
| `memory` | `str` | `"256m"` | Memory limit passed to `docker run --memory` (e.g., `"512m"`, `"1g"`) |
| `cpus` | `str` | `"1.0"` | CPU limit passed to `docker run --cpus` |
| `network` | `bool` | `False` | Whether to allow network access. When `False`, `--network none` is set on the container. |
| `workdir` | `str` | `"/workspace"` | Working directory inside the container |

<!-- name: test_readme_config -->
```python
from axio_tools_docker.config import SandboxConfig

config = SandboxConfig(
    image="python:3.12-slim",
    memory="512m",
    cpus="1.0",
    workdir="/workspace",
)
```

## Plugin registration

```toml
[project.entry-points."axio.tools.settings"]
docker = "axio_tools_docker.plugin:DockerPlugin"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-tools-local](https://github.com/axio-agent/axio-tools-local) · [axio-tools-mcp](https://github.com/axio-agent/axio-tools-mcp) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
