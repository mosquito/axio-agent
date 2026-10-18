# Permission Guards

Guards gate tool execution. They sit between parameter validation and handler
invocation, forming a composable middleware chain that can allow, deny, or
modify tool calls.

For an application that needs project rules and user approval, use `PolicyGuard`
from `axio.permission_policy`. It builds on the same guard interface described
below. Tool providers declare the effects of each operation, the policy decides
whether approval is required, and your interface collects the user's answer.

(tool-effects)=
## Declare what a tool will do

A tool name or description cannot tell a permission system which resources a
call touches. The provider supplies `Tool(access=...)`: a function of the input
arguments returning `Access` values. A file writer, for example, declares
`Access("write", filename)`. The declaration must match the handler's behavior.

`Access` supports `read`, `write`, `execute`, `external`, and `unknown`.
`file_access(operation, parameter)` builds a descriptor for one local path
argument. Multiple effects require a descriptor that returns all of them.
Recursive tools must also declare any symlink targets they can follow.

An absent descriptor means unknown effects. An explicit empty tuple declares
that the handler has no external effects. Only use it for a tool that really
has none. Descriptors report effects without executing the tool.

## Apply project permissions

`PolicyGuard` automatically allows declared reads inside its project root.
Writes, reads outside the root, execution, and opaque calls need a matching
rule or an approval. Explicit denials take priority over allows and automatic
project reads. A grant to read a file does not grant permission to write it.

Here is a complete example with a simulated user approving a single write.
Replace `approve_once()` with your UI or console prompt in an application:

<!-- name: test_concept_permission_policy -->
```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from axio import Tool
from axio.permission_policy import Approval, PermissionRequest, PolicyGuard
from axio.tool_access import file_access


async def read_note(filename: str) -> str:
    """Read a note."""
    return Path(filename).read_text()


async def write_note(filename: str, text: str) -> str:
    """Replace a note's contents."""
    Path(filename).write_text(text)
    return "Written"


async def main() -> None:
    requests = []

    async def approve_once(request: PermissionRequest) -> Approval:
        requests.append(request)
        # Stand-in for a user accepting the displayed operation once.
        print(request.describe_scope())
        return Approval(decision="allow", lifetime="once")

    with TemporaryDirectory() as directory:
        root = Path(directory)
        note = root / "notes.txt"
        note.write_text("Draft")
        policy = PolicyGuard(project_root=root, prompt=approve_once)
        policy.set_session("chat-1")
        read = Tool(name="read_note", handler=read_note, access=file_access("read", "filename"), guards=(policy,))
        write = Tool(name="write_note", handler=write_note, access=file_access("write", "filename"), guards=(policy,))

        assert await read(filename=str(note)) == "Draft"
        assert requests == []
        await write(filename=str(note), text="Reviewed")
        assert len(requests) == 1
        assert note.read_text() == "Reviewed"


asyncio.run(main())
```

To cover a whole agent, pass `tool_guards=(policy,)` to `Agent` instead of
attaching the policy to each tool. It then covers provider tools, dynamically
loaded instruction tools, and `load_instruction` itself. Application guards
run after each tool's original guards, so they see the final arguments.

Individual entry-point handlers can use
`@tool_access(file_access("read", "filename"))` from `axio.tool_access` to attach
a descriptor. The resulting `Tool` preserves it. This lets a third-party tool
participate in both permission checks and scoped instruction preflight.

## Choose approval scope and lifetime

An approval answers the displayed `PermissionRequest`. Your prompt returns an
`Approval` with `decision="allow"` or `"deny"` and one lifetime:

| Lifetime | Effect |
|---|---|
| `once` | Applies to this call only; no rule is stored |
| `chat` | Reused in the current conversation |
| `project` | Saved through the application's `RuleStore` |

Call `policy.set_session(id)` when switching, restoring, or forking a chat.
Changing the session ID clears chat rules. Project approvals require a supplied
store; without one, saving a project rule is an error. `policy.rules()` lists
active rules, and `policy.revoke(rule_id)` removes a rule. The TUI supplies its
own store and permission dialog; `/permissions` exposes these operations.

File rules cover the displayed operations and paths. `directories=True`
explicitly extends a chat or project rule to their parent directories and
descendants. For shell, Python, remote, and other opaque operations, approvals
cover the exact tool, arguments, working directory, and provider identity.
They are not command-prefix rules. `Tool.permission_identity` lets a provider
distinguish changed server configurations so old approvals cannot carry over.

Concurrent permission prompts are serialized and recheck rules before asking.
An optional advisor can add an LLM review to the prompt, but its recommendation
cannot replace approval or override an explicit denial. A provider's effect
declaration is a contract; it cannot prevent a handler lying about its effects.
Use an operating-system or container sandbox when filesystem isolation is required.

## PermissionGuard ABC

<!-- name: test_permission_guard_abc -->
```python
from abc import ABC, abstractmethod
from typing import Any
from axio import Tool


class PermissionGuard(ABC):
    async def __call__(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        return await self.check(tool, **kwargs)

    @abstractmethod
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]: ...
```

A guard receives the `Tool` object and the raw keyword arguments, and must either:

- **Return** a `dict` of (possibly modified) kwargs to allow execution.
- **Raise** `GuardError` to deny execution.

## Guard chain

When a `Tool` has multiple guards, they run **sequentially**. Each guard's
output kwargs become the next guard's input:

<!-- name: test_guard_chain -->
```python
import asyncio
from typing import Any
from axio.permission import AllowAllGuard
from axio import Tool


async def echo(text: str) -> str:
    """Echo text."""
    return text


_tool: Tool[Any] = Tool(name="echo", handler=echo)


async def main():
    guards = (AllowAllGuard(),)
    kwargs: dict[str, Any] = {"text": "hello"}
    for guard in guards:
        kwargs = await guard(_tool, **kwargs)
    assert kwargs["text"] == "hello"


asyncio.run(main())
```

This lets you compose guards freely. For example, a path-validation guard
followed by an LLM-based risk-assessment guard:

<!--
name: test_tool_with_guards
```python
from axio.permission import AllowAllGuard


async def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return "ok"


# Application-specific guards can be supplied by external packages
# For testing, you can use AllowAllGuard as a placeholder:
PathGuard = AllowAllGuard
LLMGuard = AllowAllGuard
```
-->
<!-- name: test_tool_with_guards -->
```python
from axio import Tool

tool = Tool(
    name="write_file",
    handler=write_file,
    guards=(PathGuard(), LLMGuard()),
)
assert tool.name == "write_file"
assert len(tool.guards) == 2
```

## ConcurrentGuard

For guards that need concurrency control (e.g., rate-limiting or guards that
call external services), subclass `ConcurrentGuard`:

<!-- name: test_concurrent_guard -->
```python
import asyncio
from abc import ABC, abstractmethod
from typing import Any
from axio import PermissionGuard, Tool


class ConcurrentGuard(PermissionGuard, ABC):
    concurrency: int = 1

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.concurrency)

    async def __call__(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        async with self._semaphore:
            return await self.check(tool, **kwargs)
```

Set the `concurrency` class variable to control how many tool calls can
pass through the guard simultaneously.

## Built-in guards

`AllowAllGuard`
: Always returns kwargs unchanged. Useful as a default.

`DenyAllGuard`
: Always raises `GuardError("denied")`. Useful for testing or disabling tools.


See [Writing Guards](../guides/writing-guards.md) for a step-by-step
guide to creating your own.
