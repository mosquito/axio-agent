# Writing Tools

This guide walks through creating a custom tool from scratch and registering
it as a plugin.

## 1. Create the handler

A tool handler is a Pydantic `BaseModel` subclass. Fields become the tool's
input parameters; the `__call__` method implements execution.

<!-- name: test_word_count_tool -->
```python
# my_tools/word_count.py
from axio.tool import ToolHandler


class WordCount(ToolHandler):
    """Count the number of words in the given text."""

    text: str

    async def __call__(self) -> str:
        count = len(self.text.split())
        return f"The text contains {count} words."
```

Key points:

- The **docstring** becomes the tool description sent to the LLM.
- Fields support all Pydantic features: defaults, validators, `Field()`
  metadata.
- `__call__` must be `async`. It can return a `str`, a `dict`, or any
  JSON-serialisable value. The agent coerces non-string return values to
  JSON when building the `ToolResultBlock`.

## 2. Wrap it in a Tool

<!-- name: test_word_count_tool -->
```python
from axio.tool import Tool

word_count_tool = Tool(
    name="word_count",
    description="Count words in text",
    handler=WordCount,
)
```

The `handler` parameter takes the **class**, not an instance. Axio creates a
fresh instance for each invocation via `model_validate()`.

## 3. Use it with an agent

<!--
name: test_word_count_tool
```python
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
my_transport = StubTransport([make_text_response("ok")])
context = MemoryContextStore()
```
-->
<!-- name: test_word_count_tool -->
```python
agent = Agent(
    system="You are a helpful assistant.",
    tools=[word_count_tool],
    transport=my_transport,
)
```

## 4. Register as a plugin

To make your tool discoverable by the TUI and other Axio applications, add an
entry point to your `pyproject.toml`:

```toml
[project.entry-points."axio.tools"]
word_count = "my_tools.word_count:WordCount"
```

After installing or syncing, `discover_tools()` will find it automatically.

## Adding guards

Attach guards to control when the tool can run:

<!--
name: test_tool_with_guard
```python
from axio.tool import Tool, ToolHandler

class WordCount(ToolHandler):
    """Count words."""
    text: str
    async def __call__(self) -> str:
        return str(len(self.text.split()))
```
-->
<!-- name: test_tool_with_guard -->
```python
from axio.permission import AllowAllGuard

tool = Tool(
    name="word_count",
    description="Count words in text",
    handler=WordCount,
    guards=(AllowAllGuard(),),
)
```

See [Guards](../concepts/guards.md) for more on the guard system.

## Concurrency control

Limit how many instances of your tool can run simultaneously:

```python
tool = Tool(
    name="web_fetch",
    description="Fetch a URL",
    handler=WebFetch,
    concurrency=3,  # at most 3 concurrent fetches
)
```

## Error handling

If your handler raises an exception, Axio wraps it in `HandlerError` and
sends the error message back to the model as a `ToolResultBlock` with
`is_error=True`. The model sees the error and can adjust its approach.

For expected failures, raise `HandlerError` directly with a clear message:

<!--
name: test_error_handling
```python
from pathlib import Path
from axio.tool import ToolHandler
```
-->
<!-- name: test_error_handling -->
```python
from axio.exceptions import HandlerError

class ReadFile(ToolHandler):
    """Read a file."""
    path: str

    async def __call__(self) -> str:
        p = Path(self.path)
        if not p.exists():
            raise HandlerError(f"File not found: {self.path}")
        return p.read_text()
```

## Dynamic tool providers

If your package needs to provide a variable number of tools based on
configuration (like MCP servers or Docker containers), implement the
`ToolsPlugin` protocol instead. `ToolsPlugin` is defined in the
`axio_tui` package:

```python
from typing import Any
from axio_tui.plugin import ToolsPlugin
from axio.tool import Tool


class MyPlugin:
    """A dynamic tool provider."""

    @property
    def label(self) -> str:
        return "My Plugin"

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        # Load configuration, connect to external services, etc.
        pass

    @property
    def all_tools(self) -> list[Tool]:
        # Build and return tools based on current configuration
        return [...]

    def settings_screen(self) -> Any:
        # Return a Textual Screen for configuring this plugin in the TUI
        return None

    async def close(self) -> None:
        # Clean up connections and resources
        pass
```

Register under `axio.tools.settings`:

```toml
[project.entry-points."axio.tools.settings"]
my_plugin = "my_package.plugin:MyPlugin"
```

See [Plugin System](../concepts/plugins.md) for the full `ToolsPlugin` protocol
and lifecycle documentation.
