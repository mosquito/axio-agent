# axio-tui-guards

[![PyPI](https://img.shields.io/pypi/v/axio-tui-guards)](https://pypi.org/project/axio-tui-guards/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tui-guards)](https://pypi.org/project/axio-tui-guards/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/mosquito/axio-agent/blob/main/LICENSE)

Permission guard plugins for [axio-tui](https://github.com/mosquito/axio-agent).

The TUI applies its built-in permission policy to every agent tool. Read-only file
access inside the project is automatic. Other operations require explicit rules or
confirmation. See [TUI permissions](https://github.com/mosquito/axio-agent/tree/main/axio-tui#tool-permissions).

This optional package supplies an LLM reviewer and the legacy standalone PathGuard.
The TUI uses LLM review as advice in its permission dialog. A SAFE verdict cannot
replace user approval or override an explicit denial.

## Installation

```bash
pip install axio-tui-guards
```

The base package requires only `axio`. Both guards accept an asynchronous
`prompt_fn` callback and work without Textual or `axio-tui`. The verdict type
is `axio_tui_guards.guards.Confirm`. Install `axio-tui-guards[tui]` only when
using the optional Textual dialogs.

Or install as part of the TUI bundle:

```bash
pip install "axio-tui[guards]"
```

## Guards

### PathGuard

Legacy standalone guard for applications that attach it explicitly. The TUI no longer
instantiates it: its directory-wide grants cannot distinguish reads from writes.
New applications should use `axio.permission_policy.PolicyGuard` and provider access
declarations instead. PathGuard examines only the first recognized path parameter;
it is not filesystem isolation.

<!--
name: test_readme_path_guard
```python
async def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    return "ok"


async def shell(command: str, cwd: str = ".") -> str:
    """Run a shell command."""
    return "ok"
```
-->
<!-- name: test_readme_path_guard -->
```python
from axio_tui_guards.guards import PathGuard
from axio.tool import Tool

guard = PathGuard()  # defaults to console input; applications can pass prompt_fn

tool = Tool(
    name="write_file",
    description="Write a file",
    handler=write_file,
    guards=(guard,),
)
```

Legacy decisions: `y` grants the inferred directory and its descendants for this
instance, `n` denies once, and `deny` remembers an exact path denial. No decision
persists across process restarts. This API remains for compatibility only.

### LLMGuard

`review()` returns advice for the TUI's unified permission prompt. Missing or invalid
verdicts and review failures require explicit permission. Categories never create
stored grants.

When used directly as a standalone `PermissionGuard`, `check()` accepts SAFE,
rejects DENY, and asks for an explicit `y` on RISKY. This standalone API does not
replace the TUI policy.

<!-- name: test_readme_llm_guard -->
```python
from axio_tui_guards.guards import LLMGuard
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response

reviewer = Agent(system="", tools=[], transport=StubTransport([make_text_response("allow")]))
guard = LLMGuard(agent=reviewer, context=MemoryContextStore())
```

### Composing guards

Guards are applied in order. Existing standalone applications can still compose them:

<!--
name: test_readme_composing
```python
async def shell(command: str, cwd: str = ".") -> str:
    """Run a shell command."""
    return "ok"
```
-->
<!-- name: test_readme_composing -->
```python
from axio_tui_guards.guards import PathGuard, LLMGuard
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool

reviewer = Agent(system="", tools=[], transport=StubTransport([make_text_response("allow")]))
tool = Tool(
    name="shell",
    description="Run shell commands",
    handler=shell,
    guards=(PathGuard(), LLMGuard(agent=reviewer, context=MemoryContextStore())),
)
```

## Custom guards

Implement the `PermissionGuard` protocol to write your own:

<!-- name: test_readme_custom_guard -->
```python
from typing import Any
from axio.permission import PermissionGuard
from axio.exceptions import GuardError


class MyGuard(PermissionGuard):
    async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        if "rm -rf" in str(kwargs.get("command", "")):
            raise GuardError("Refusing to run rm -rf")
        return kwargs
```

## Plugin registration

```toml
[project.entry-points."axio.guards"]
path = "axio_tui_guards.guards:PathGuard"
llm  = "axio_tui_guards.guards:LLMGuard"
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
