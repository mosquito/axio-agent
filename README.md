# axio-tui-guards

[![PyPI](https://img.shields.io/pypi/v/axio-tui-guards)](https://pypi.org/project/axio-tui-guards/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tui-guards)](https://pypi.org/project/axio-tui-guards/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Permission guard plugins for [axio-tui](https://github.com/axio-agent/axio-tui).

Guards intercept tool calls before execution and can allow, modify, or deny them. Ships two guards: a path-access guard that asks the user before touching filesystem locations, and an LLM-based guard that reviews tool calls for safety.

## Features

- **PathGuard** — intercepts tools that touch the filesystem; prompts the user once per directory and remembers the decision for the session
- **LLMGuard** — runs a secondary LLM call to review each tool invocation before allowing it
- **PermissionGuard protocol** — both guards implement `axio.PermissionGuard` and compose cleanly
- **TUI-aware** — prompts appear as native `axio-tui` dialogs, not blocking stdin reads

## Installation

```bash
pip install axio-tui-guards
```

Or install as part of the TUI bundle:

```bash
pip install "axio-tui[guards]"
```

## Guards

### PathGuard

Intercepts tool calls that contain filesystem paths (`file_path`, `filename`, `directory`, `path`, `cwd`). On the first access to a new directory it asks the user to **allow**, **allow all** (subtree), or **deny**.

```python
from axio_tui_guards.guards import PathGuard
from axio.tool import Tool

guard = PathGuard()   # uses TUI prompt_fn by default

tool = Tool(
    name="write_file",
    description="Write a file",
    handler=WriteFile,
    guards=(guard,),
)
```

Decision caching:
- **Allow** — grants access to the parent directory for the current session
- **Allow all** — grants access to the directory and all subdirectories
- **Deny** — blocks this path; raises `GuardError` immediately on retry

### LLMGuard

Uses a secondary LLM call to review the tool handler arguments before execution. If the reviewer deems the call unsafe it raises `GuardError` with the reason.

```python
from axio_tui_guards.guards import LLMGuard
from axio_transport_openai import OpenAITransport

reviewer = OpenAITransport(api_key="sk-...", model="gpt-4o-mini")
guard = LLMGuard(transport=reviewer)
```

### Composing guards

Guards are applied in order — attach both for layered protection:

```python
tool = Tool(
    name="shell",
    description="Run shell commands",
    handler=Shell,
    guards=(PathGuard(), LLMGuard(transport=reviewer)),
)
```

## Custom guards

Implement the `PermissionGuard` protocol to write your own:

```python
from axio.permission import PermissionGuard
from axio.exceptions import GuardError

class MyGuard(PermissionGuard):
    async def check(self, handler):
        if "rm -rf" in getattr(handler, "command", ""):
            raise GuardError("Refusing to run rm -rf")
        return handler
```

## Plugin registration

```toml
[project.entry-points."axio.guards"]
path = "axio_tui_guards.guards:PathGuard"
llm  = "axio_tui_guards.guards:LLMGuard"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-tui](https://github.com/axio-agent/axio-tui) · [axio-tui-rag](https://github.com/axio-agent/axio-tui-rag)

## License

MIT
