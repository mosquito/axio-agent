# axio-tools-docker

[![PyPI](https://img.shields.io/pypi/v/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Docker sandbox tools for [axio](https://github.com/mosquito/axio-agent).

Run agent-generated code and commands inside isolated Docker containers.
`DockerSandbox` is an async context manager that creates a container on enter
and removes it on exit. Inside the context it exposes a set of `axio` tools
that are drop-in replacements for `axio-tools-local` — the agent gets the same
`shell`, `write_file`, `read_file`, `list_files`, `run_python`, and `patch_file`
tools, but they operate entirely inside the container.

## Features

- **Isolated execution** — code runs inside a Docker container, not on the host
- **No docker CLI required** — communicates with the Docker Engine API directly via `aiodocker`
- **Configurable image** — use any Docker image as the sandbox environment
- **Drop-in replacement** — same tool names and field schemas as `axio-tools-local`
- **Persistent sandbox** — container is reused across tool calls within a session
- **Volume mounts** — bind host directories into the container

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
import asyncio
import os
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport
from axio_tools_docker import DockerSandbox

async def main() -> None:
    async with DockerSandbox(
        os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock"),
        image="python:3.12-slim",
        volumes={"/workspace": "/tmp/agent-workspace"},
    ) as sandbox:
        agent = Agent(
            system="You are a coding assistant. Use the sandbox tools to run code safely.",
            tools=sandbox.tools,
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

These mirror `axio-tools-local` exactly — the same names and field schemas:

| Tool | Description |
|---|---|
| `shell` | Run a shell command; returns stdout + stderr |
| `write_file` | Create or overwrite a file |
| `read_file` | Read a file with optional line range and line numbers |
| `list_files` | List directory contents; directories shown first with trailing `/` |
| `run_python` | Execute a Python code snippet |
| `patch_file` | Replace a range of lines in an existing file |

## Container lifecycle

The container is created on `__aenter__`. On `__aexit__` it is removed with `docker rm -f`
unless `remove=False` was passed. Cleanup runs even on exceptions:

```python
async def run(ctx):
    async with DockerSandbox(image="alpine:latest") as sandbox:
        agent = Agent(..., tools=sandbox.tools)
        await agent.run("...", ctx)
    # container is removed here (unless remove=False)
```

## Configuration

<!-- name: test_readme_config -->
```python
from axio_tools_docker import DockerSandbox

sandbox = DockerSandbox(
    "unix:///var/run/docker.sock",   # Docker daemon URL (positional)
    image="python:3.12-slim",
    memory="512m",                   # e.g. "256m", "1g"
    cpus="2.0",
    network=False,                   # False=none, True=default, or a string e.g. "host"
    workdir="/workspace",
    volumes={"/workspace": "/tmp/host-dir"},   # {container_path: host_path}
    env={"PYTHONPATH": "/app", "MY_TOKEN": "secret"},
    user="nobody",
    name="my-agent-sandbox",         # named container for reuse / inspection
    remove=False,                    # keep container after exit
    read_only=True,                  # read-only root filesystem
    shm_size="64m",                  # /dev/shm size
    cap_add=["NET_ADMIN"],           # add Linux capabilities
    cap_drop=["ALL"],                # drop Linux capabilities
    privileged=False,                # extended privileges (use with care)
    ulimits={"nofile": (1024, 65536), "nproc": 512},  # resource limits
    tmpfs={"/tmp": "size=128m,mode=1777"},             # tmpfs mounts
    ports={8080: 8080},             # {container_port: host_port}
    platform="linux/amd64",          # image platform override
    extra_hosts={"host.docker.internal": "host-gateway"},
    devices=["/dev/net/tun", "/dev/sda:/dev/xvda:r"],  # host devices
    dns=["8.8.8.8", "1.1.1.1"],     # DNS servers
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"unix:///var/run/docker.sock"` | Docker daemon URL |
| `image` | `str` | `"python:latest"` | Container image |
| `memory` | `str` | `"256m"` | Memory limit (e.g. `"512m"`, `"1g"`) |
| `cpus` | `str` | `"1.0"` | CPU limit |
| `network` | `bool \| str` | `False` | Network mode. `False` → `none` (no network). `True` → Docker default. A string sets `NetworkMode` explicitly (e.g. `"host"`, `"bridge"`, `"my-project_default"`). |
| `workdir` | `str` | `"/workspace"` | Working directory inside container |
| `volumes` | `dict[str, str]` | `{}` | `{container_path: host_path}` mounts |
| `env` | `dict[str, str]` | `{}` | Environment variables passed to all commands |
| `user` | `str` | `""` | User to run as inside the container (e.g. `"nobody"`, `"1000"`) |
| `name` | `str` | `""` | Container name. If a container with this name already exists, the sandbox attaches to it (no create/start/remove). If it doesn't exist, a new one is created. |
| `remove` | `bool` | `True` | Remove the container on exit; set to `False` to keep it. Has no effect when attaching to an existing container. |
| `read_only` | `bool` | `False` | Mount the container root filesystem as read-only. Combine with `tmpfs` to allow writable scratch space. |
| `shm_size` | `str` | `""` | Size of `/dev/shm`, e.g. `"64m"`. Useful for PyTorch and any IPC via shared memory. |
| `cap_add` | `list[str]` | `[]` | Linux capabilities to add, e.g. `["NET_ADMIN", "SYS_PTRACE"]`. |
| `cap_drop` | `list[str]` | `[]` | Linux capabilities to drop, e.g. `["ALL"]`. |
| `privileged` | `bool` | `False` | Give extended privileges (full capability set + device access). Use with care. |
| `ulimits` | `dict[str, int \| tuple[int, int]]` | `{}` | Resource limits. `{"nofile": 1024}` sets soft=hard=1024. `{"nofile": (1024, 65536)}` sets soft and hard separately. |
| `tmpfs` | `dict[str, str]` | `{}` | Tmpfs mounts as `{path: options}`, e.g. `{"/tmp": "size=128m,mode=1777"}`. Empty string uses Docker defaults. |
| `ports` | `dict[int, int]` | `{}` | Port bindings as `{container_port: host_port}`. Only meaningful when `network` is not `False`. |
| `platform` | `str` | `""` | Platform override, e.g. `"linux/amd64"` or `"linux/arm64"`. |
| `extra_hosts` | `dict[str, str]` | `{}` | Additional `/etc/hosts` entries as `{hostname: ip}`, e.g. `{"host.docker.internal": "host-gateway"}`. |
| `devices` | `list[str]` | `[]` | Host devices to expose. Format: `"/dev/sda"` (same path, `rwm`), `"/dev/sda:/dev/xvda"` (custom container path), `"/dev/sda:/dev/xvda:r"` (explicit permissions). |
| `dns` | `list[str]` | `[]` | DNS servers, e.g. `["8.8.8.8", "1.1.1.1"]`. Only meaningful when `network` is not `False`. |

The running container's Docker ID is available as `sandbox.container_id` inside the `async with` block.

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-tools-local](https://github.com/mosquito/axio-agent) · [axio-tools-mcp](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
