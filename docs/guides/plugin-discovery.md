# Register and discover plugins

Axio discovers extensions from the metadata of installed Python distributions.
A package declares entry points in `pyproject.toml`; the application loads the
groups it supports. Your code can use the same discovery functions from core.

An entry-point group is a metadata key, not a Python import path. Choose the
group by what the extension supplies:

- `axio.tools`: one async tool handler per entry.
- `axio.tools.providers`: a tool provider that supplies a set of tools and
  related instructions, with shared initialization and cleanup. Local tools
  and MCP use this group.
- `axio.transport`: an LLM transport, such as OpenAI, Anthropic, or Google.

The word *provider* in `ToolProvider` means a supplier of tools and instructions.
It does not mean the company or service that runs the language model.

## How discovery works

1. Install the extension in the application's Python environment. Its build
   backend writes the entry-point declarations into distribution metadata.
2. A discovery function calls `importlib.metadata.entry_points(group=...)`.
   It reads installed metadata, not source directories or arbitrary
   `pyproject.toml` files.
3. For each declaration, `EntryPoint.load()` imports the target module and
   resolves the named object. The group adapter checks and adapts that object.
   Provider and context-backend discovery also call their zero-argument factories.
4. The function returns a catalog. The application selects entries, initializes
   their resources, and attaches them to an agent or interface.

Discovery executes Python imports and, for factory groups, constructors. Keep
these operations free of network connections and other acquired resources.
Put resource acquisition in the lifecycle methods described below.

Importing `axio` or constructing an `Agent` does not discover or attach plugins.
REPL and TUI call discovery during their startup. In a custom application, call
the functions explicitly. Each discovery call builds a new catalog; provider
factories can therefore create new instances on each call.

## Choose the entry-point group

The first six groups are available through core. The remaining groups have
specific consumers and do not change what an ordinary `Agent` loads.

| Group | Registration target | Consumer |
|---|---|---|
| `axio.tools` | Plain `async def` tool handler | `axio.plugins.discover_tools()` |
| `axio.tools.providers` | Zero-argument factory returning a `ToolProvider` | `axio.plugins.discover_tool_providers()` |
| `axio.guards` | `PermissionGuard` subclass | `axio.plugins.discover_guards()` |
| `axio.selector` | Class implementing `ToolSelector.select()` | `axio.plugins.discover_selectors()` |
| `axio.context` | Zero-argument factory returning a `ContextBackend` | `axio.plugins.discover_context_backends()` |
| `axio.transport` | Transport class or managed plugin object | `axio.transport.discover_transports()` |
| `axio.transport.settings` | Textual settings screen class | TUI `discover_transport_settings()` |
| `axio.tools.settings` | Zero-argument factory returning a TUI `ToolsPlugin` | TUI `discover_tools_plugins()` |
| `axio.transport.realtime` | Factory or class returning a `RealtimeTransport` | `examples/realtime_chat/chat.py` |

Core discovery requires neither REPL nor TUI. Applications decide which groups
to consume and which discovered components to activate. For example, TUI uses
`axio.context` backends for conversation history; registering one does not
replace the context store that your own code passes to `Agent.run()`.

## Declare and install an extension

Each value has the form `importable.module:object`. Do not add parentheses:
the consumer decides whether and when to call the object. The key on the left
is the entry's name within its group. It need not match the distribution name
or Python class name.

For example, a package containing `src/example_tools/__init__.py` can declare:

```toml
[project]
name = "example-axio-tools"
version = "0.1.0"
dependencies = ["axio"]

[project.entry-points."axio.tools.providers"]
example = "example_tools:Provider"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/example_tools"]
```

Install it from your application project with
`uv add --editable /path/to/example-axio-tools`, then restart the application.
If you change entry-point declarations, reinstall or resync the package so its
installed metadata changes too. Editing the source `pyproject.toml` alone does
not update an already installed distribution.

### Individual tools and providers

Use `axio.tools` for an independent async handler. Discovery wraps the handler
in `Tool`, using the entry-point name as its tool name. It derives the schema
from the function's annotations and the description from its docstring.
Register the function itself, not a `Tool` instance:

```toml
[project.entry-points."axio.tools"]
echo = "example_tools:echo"
```

Use `axio.tools.providers` when tools need initialization, shared state, a changing
tool set, or instructions. A provider supplies `async init(config=None,
global_config=None)`, `async close()`, an `all_tools` property, and synchronous
`get_instructions()`. The latter returns a sequence of `Instruction` objects;
their full content is loaded separately. Configuration belongs to the provider.

Here is a complete `example_tools/__init__.py` for the provider registration:

<!-- name: test_plugin_provider_example -->
```python
from __future__ import annotations

from typing import Any

from axio import Tool
from axio.instructions import Instruction


async def echo(text: str) -> str:
    """Return the supplied text."""
    return text


class Provider:
    def __init__(self) -> None:
        self._tools: list[Tool[Any]] = []

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        self._tools = [Tool(name="echo", handler=echo)]

    @property
    def all_tools(self) -> list[Tool[Any]]:
        return list(self._tools)

    def get_instructions(self) -> list[Instruction]:
        return []

    async def close(self) -> None:
        self._tools.clear()
```

In application code, select and initialize that provider before collecting tools:

<!-- name: test_plugin_provider_example -->
```python
from axio.plugins import collect_tools, discover_tool_providers, open_tool_providers


async def use_provider() -> str:
    catalog = discover_tool_providers()
    provider = catalog.require("example")
    async with open_tool_providers([provider]) as opened:
        tools = collect_tools(opened)
        # In an agent application, pass these tools to Agent(tools=tools, ...).
        return str(await tools[0](text="hello"))
```

<!--
name: test_plugin_provider_example
```python
import asyncio
import sys
from importlib.metadata import EntryPoint
from types import ModuleType
from unittest.mock import patch

module = ModuleType("example_tools")
module.Provider = Provider
point = EntryPoint(name="example", value="example_tools:Provider", group="axio.tools.providers")
with patch.dict(sys.modules, {"example_tools": module}):
    with patch("importlib.metadata.entry_points", return_value=[point]):
        assert asyncio.run(use_provider()) == "hello"
```
-->

`open_tool_providers()` registers cleanup before `init()` and closes providers
in reverse order. A failed initialization closes that provider and those already
started, then propagates the error. `close()` must handle partial initialization.
Exceptions and cancellation inside the context also trigger cleanup. Keep agent
runs that use these tools inside this context.

`collect_tools()` reads the current tool sets. It rejects different tool objects
with the same name, including conflicts with separately supplied tools. Repeated
references to the same object are allowed. Provider entry names do not prefix
the tool names that providers choose.

If a distribution registers both plain handlers and a provider, use
`standalone_tools(tool_catalog, provider_catalog)` after provider initialization
to omit plain handlers already supplied by that distribution's provider.
It also omits handlers from distributions with provider load errors.
This does not resolve conflicts between different distributions.

### Guards and selectors

Register classes, not instances. Discovery returns the classes without calling
their constructors. The application constructs them with the required settings.

```toml
[project.entry-points."axio.guards"]
audit = "example_tools.guards:AuditGuard"

[project.entry-points."axio.selector"]
relevant = "example_tools.selectors:RelevantTools"
```

`AuditGuard` must inherit from `axio.permission.PermissionGuard` and implement
`async check(tool, **kwargs)`. It returns allowed arguments or raises
`GuardError`. Attach guard instances through `Tool.guards`.
See {doc}`writing-guards` for implementation examples.

`RelevantTools` must implement `async select(messages, tools)`, returning an
iterable of tools. It can satisfy `axio.selector.ToolSelector` structurally,
without inheriting from it. Pass its instance as `Agent(selector=...)`.

### Context backends

Register a factory that returns an object with `async init(path: Path)`,
`create(session_id: str | None = None) -> ContextStore`, and `async close()`:

```toml
[project.entry-points."axio.context"]
example = "example_store:Backend"
```

Discovery constructs the backend but does not call `init()`. The application
initializes it, creates conversation stores, and closes its shared resources
after their users finish. A raw `ContextStore` class does not satisfy this
factory contract. See {doc}`writing-context-stores` for the store interface.

### Completion and realtime transports

```toml
[project.entry-points."axio.transport"]
example = "example_transport:Transport"

[project.entry-points."axio.transport.realtime"]
example = "example_transport.realtime:RealtimeTransport"
```

Completion discovery returns `TransportEntry` adapters without opening the
transports. A class accepts `session=...` and provider settings when opened.
A managed plugin object supplies `open(...)`, or a class supplies
`open_for_app(...)`, returning an async context manager. Register a plugin
instance as `example = "example_transport:plugin"` when using that form.
See {doc}`writing-transports` for these contracts and {doc}`choosing-transports`
for model selection, authentication, and resource ownership in user code.

Realtime is a separate registration. Its target constructs an object with
`RealtimeTransport.connect()`. `discover_transports()` does not read this group;
the realtime chat example loads it directly through `importlib.metadata`.
That example has its own selection and construction logic, so registration
alone does not add provider-specific configuration handling.

### TUI settings integrations

```toml
[project.entry-points."axio.transport.settings"]
example = "example_transport.tui:SettingsScreen"

[project.entry-points."axio.tools.settings"]
example = "example_tools.tui:ToolsPlugin"
```

For a transport configuration screen, use the same entry name as its
`axio.transport` registration. TUI constructs `SettingsScreen(settings)` with
saved string settings. The screen dismisses with a `dict[str, str]` to save
changes, or `None` to cancel. TUI also supports hub screens that create dynamic
providers; see the existing transport UI implementations for that extension.

`axio.tools.settings` is the legacy TUI tool-plugin interface. Its factory
returns an object with `init()`, `close()`, `all_tools`, a `label` property,
and `settings_screen()`. It is not an alias for `axio.tools.providers`: the two groups
are discovered and instantiated separately. Use `axio.tools.providers` for tools and
instructions that must also work outside TUI. Keep Textual imports in the UI
module so importing the core provider does not require TUI dependencies.

## Diagnose discovery

Core catalogs expose `entries`, `errors`, and `sources`. Sources record the
group, declaring distribution, and target. An empty catalog means that no
usable entries were loaded; inspect `errors` to distinguish missing packages
from failed imports or invalid contracts.

<!-- name: test_plugin_catalog_diagnostics -->
```python
from axio.plugins import discover_tool_providers

catalog = discover_tool_providers()
for name, error in catalog.errors.items():
    source = catalog.sources[name]
    print(f"{name}: {source.distribution} / {source.value}: {error}")
```

`catalog.require(name)` raises `PluginError` for a missing or failed core plugin.
The transport catalog uses `TransportError`. Other valid entries remain usable.

The shared loader reserves the first declaration of each name within a group,
even if loading fails. Later duplicates are skipped with a warning. Do not rely
on distribution enumeration order to override another package: choose distinct
names. The same name in different groups is allowed.

TUI settings discovery uses this loader but returns successful entries as
dictionaries; load failures are logged. The realtime example uses a separate
loader that skips load failures and lets later names replace earlier ones.
Its behavior is not the core catalog contract.

If a plugin is missing, verify the active Python environment, its installed
distribution metadata, the exact group name, and the importability of the target.
Installing a dependency extra makes its packages available; discovery still
follows their entry-point declarations.
