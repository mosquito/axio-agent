# axio-tools-local

[![PyPI](https://img.shields.io/pypi/v/axio-tools-local)](https://pypi.org/project/axio-tools-local/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-local)](https://pypi.org/project/axio-tools-local/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/mosquito/axio-agent/blob/main/LICENSE)

Core filesystem and shell tool handlers for [axio](https://github.com/mosquito/axio-agent).

Gives your agent the ability to read, write, and patch files, run shell commands, execute Python snippets, and browse directory trees - the essential toolkit for a coding assistant.

## Tools

| Function | Entry point | Description |
|---|---|---|
| `shell` | `shell` | Run any shell command with configurable timeout, cwd, and stdin |
| `run_python` | `run_python` | Execute a Python snippet in a subprocess |
| `read_file` | `read_file` | Read a file, optionally with line range |
| `write_file` | `write_file` | Write or overwrite a file |
| `patch_file` | `patch_file` | Replace a range of lines in an existing file (1-indexed, both ends inclusive) |
| `list_files` | `list_files` | List files in a directory |

## Installation

```bash
pip install axio-tools-local
```

## Usage

### Standalone (without TUI)

<!-- name: test_readme_standalone -->
```python
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport, OPENAI_MODELS
from axio_tools_local.shell import shell
from axio_tools_local.read_file import read_file
from axio_tools_local.write_file import write_file
from axio_tools_local.list_files import list_files
from axio.tool import Tool

tools = [
    Tool(name="shell", handler=shell),
    Tool(name="read_file", handler=read_file),
    Tool(name="write_file", handler=write_file),
    Tool(name="list_files", handler=list_files),
]

agent = Agent(
    system="You are a coding assistant with access to the local filesystem.",
    tools=tools,
    transport=OpenAITransport(api_key="sk-...", model=OPENAI_MODELS["gpt-4o"]),
)
```

### Via plugin (with axio-tui)

```bash
pip install "axio-tui[local]"
uv run axio   # shell, read_file, write_file, patch_file, list_files, run_python appear automatically
```

## Tool details

### shell

<!-- name: test_readme_shell -->
```python
import asyncio
from axio_tools_local.shell import shell

asyncio.run(shell(command="echo hello", cwd=".", timeout=30))
asyncio.run(shell(command="cat", stdin="hello"))
```

Parameters: `command: str`, `timeout: int = 5`, `cwd: str = "."`, `stdin: str | None = None`

### patch_file

Replaces a range of lines in an existing file - safe for surgical edits without
rewriting the whole file. Lines are 1-indexed and both `from_line` and `to_line`
are inclusive. To insert without deleting any existing lines, set
`to_line = from_line - 1`. Always read the file first with `line_numbers=True` to
get correct line numbers.

<!-- name: test_readme_patch_file -->
```python
import asyncio
import tempfile, os
from axio_tools_local.patch_file import patch_file

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write("line1\nline2\nline3\n")
    name = f.name

# Replace line 2
asyncio.run(patch_file(file_path=name, from_line=2, to_line=2, content="replaced\n"))
os.unlink(name)
```

Parameters: `file_path: str`, `from_line: int`, `to_line: int`, `content: str`, `mode: int = 0o644`

### list_files

<!-- name: test_readme_list_files -->
```python
import asyncio
from axio_tools_local.list_files import list_files

asyncio.run(list_files(directory="."))
```

### run_python

<!-- name: test_readme_run_python -->
```python
import asyncio
from axio_tools_local.run_python import run_python

asyncio.run(run_python(code="import sys; print(sys.version)"))
```

## Plugin registration

```toml
[project.entry-points."axio.tools"]
shell      = "axio_tools_local.shell:shell"
run_python = "axio_tools_local.run_python:run_python"
write_file = "axio_tools_local.write_file:write_file"
patch_file = "axio_tools_local.patch_file:patch_file"
read_file  = "axio_tools_local.read_file:read_file"
list_files = "axio_tools_local.list_files:list_files"
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-tools-mcp](https://github.com/mosquito/axio-agent) · [axio-tools-docker](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT

## Local instruction provider

`LocalToolProvider` owns filesystem tools, scoped `AGENTS.md` loading, and local
Agent Skills discovery. It implements the `axio.tools.providers` contract.
Applications supply directories explicitly and bind each agent separately.

<!-- name: test_readme_local_provider -->
```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio_tools_local.provider import LocalOptions, LocalToolProvider


async def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        provider = LocalToolProvider()
        await provider.init(LocalOptions(cwd=root, roots=(root,), isolated=True))
        try:
            agent = Agent(
                "Follow the project instructions.", StubTransport([make_text_response()]), providers=[provider]
            )
            provider.bind(agent)
            await agent.run("Hello", MemoryContextStore())
        finally:
            await provider.close()


asyncio.run(main())
```

Relative tool arguments use the configured `cwd`. `roots` must be existing,
non-overlapping directories. Binding adds required instructions and hooks without
changing the base system prompt or replacing caller hooks. Bind copied agents
separately so their scoped instruction snapshots remain independent.

Unless `isolated=True`, file instructions include global Axio `AGENTS.md` and
root-to-target `AGENTS.md` files. Explicit `instruction_files` remain enabled in
isolated mode. The combined byte limit rejects oversized snapshots without
truncation. A project instruction symlink cannot escape its project root.

Preflight uses each selected tool's `Tool.access` declaration, including tools
from other providers. New or changed instructions defer the entire batch until
the model receives the updated content. An instruction edit runs separately from
other operations in its batch. These checks do not provide filesystem isolation.
They cannot observe paths hidden inside shell commands, Python code, or MCP calls
that do not declare their local file effects.

Local skills expose short metadata through `Instruction`. Loading a skill returns
its full text, a resource reader, and the provider's local execution tools.
Dependencies are not inferred from Markdown or installed automatically. Tools
retain their access declarations and remain subject to the agent's guards.
`reload()` replaces the skill catalog; each bound agent updates it before its next
model call. Removed skills lose their active text and tools unless those tools
remain independently configured on the agent.
