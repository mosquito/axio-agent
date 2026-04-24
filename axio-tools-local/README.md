# axio-tools-local

[![PyPI](https://img.shields.io/pypi/v/axio-tools-local)](https://pypi.org/project/axio-tools-local/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-local)](https://pypi.org/project/axio-tools-local/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Core filesystem and shell tool handlers for [axio](https://github.com/axio-agent/axio).

Gives your agent the ability to read, write, and patch files, run shell commands, execute Python snippets, and browse directory trees — the essential toolkit for a coding assistant.

## Tools

| Tool | Entry point | Description |
|---|---|---|
| `Shell` | `shell` | Run any shell command with configurable timeout, cwd, and stdin |
| `RunPython` | `run_python` | Execute a Python snippet in a subprocess |
| `ReadFile` | `read_file` | Read a file, optionally with line range |
| `WriteFile` | `write_file` | Write or overwrite a file |
| `PatchFile` | `patch_file` | Replace a range of lines in an existing file (1-indexed, both ends inclusive) |
| `ListFiles` | `list_files` | List files matching a glob pattern |

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
from axio_tools_local.shell import Shell
from axio_tools_local.read_file import ReadFile
from axio_tools_local.write_file import WriteFile
from axio_tools_local.list_files import ListFiles
from axio.tool import Tool

tools = [
    Tool(name="shell",      description=Shell.__doc__ or "",     handler=Shell),
    Tool(name="read_file",  description=ReadFile.__doc__ or "",  handler=ReadFile),
    Tool(name="write_file", description=WriteFile.__doc__ or "", handler=WriteFile),
    Tool(name="list_files", description=ListFiles.__doc__ or "", handler=ListFiles),
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
uv run axio   # Shell, ReadFile, WriteFile, PatchFile, ListFiles, RunPython appear automatically
```

## Tool details

### Shell

<!--
name: test_readme_shell
```python
from axio_tools_local.shell import Shell
```
-->
<!-- name: test_readme_shell -->
```python
Shell(command="git log --oneline -5", cwd="/path/to/repo", timeout=30)
Shell(command="python -m pytest", stdin=None)
```

Parameters: `command: str`, `timeout: int = 5`, `cwd: str = "."`, `stdin: str | None = None`

### PatchFile

Replaces a range of lines in an existing file — safe for surgical edits without
rewriting the whole file. Lines are 1-indexed and both `from_line` and `to_line`
are inclusive. To insert without deleting any existing lines, set
`to_line = from_line - 1`. Always read the file first with `indexed=True` to
get correct line numbers.

<!--
name: test_readme_patch_file
```python
from axio_tools_local.patch_file import PatchFile
```
-->
<!-- name: test_readme_patch_file -->
```python
# Replace line 5 with a new function signature
PatchFile(
    file_path="src/main.py",
    from_line=5,
    to_line=5,
    content="def foo(x: int) -> int:",
)

# Replace lines 2-4 (both inclusive) with two new lines
PatchFile(
    file_path="src/main.py",
    from_line=2,
    to_line=4,
    content="line_a\nline_b",
)

# Insert before line 3 (no lines deleted)
PatchFile(
    file_path="src/main.py",
    from_line=3,
    to_line=2,
    content="# inserted comment",
)
```

Parameters: `file_path: str`, `from_line: int`, `to_line: int`, `content: str`, `mode: int = 0o644`

### ListFiles

<!--
name: test_readme_list_files
```python
from axio_tools_local.list_files import ListFiles
```
-->
<!-- name: test_readme_list_files -->
```python
ListFiles(pattern="src/**/*.py")
ListFiles(pattern="tests/test_*.py")
```

### RunPython

<!--
name: test_readme_run_python
```python
from axio_tools_local.run_python import RunPython
```
-->
<!-- name: test_readme_run_python -->
```python
RunPython(code="import sys; print(sys.version)")
```

## Plugin registration

```toml
[project.entry-points."axio.tools"]
shell      = "axio_tools_local.shell:Shell"
run_python = "axio_tools_local.run_python:RunPython"
write_file = "axio_tools_local.write_file:WriteFile"
patch_file = "axio_tools_local.patch_file:PatchFile"
read_file  = "axio_tools_local.read_file:ReadFile"
list_files = "axio_tools_local.list_files:ListFiles"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-tools-mcp](https://github.com/axio-agent/axio-tools-mcp) · [axio-tools-docker](https://github.com/axio-agent/axio-tools-docker) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
