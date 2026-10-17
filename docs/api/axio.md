# `axio`

Core agent loop, protocols, messages, tools, events, and context interfaces.

## Public API

```{eval-rst}
.. automodule:: axio
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
```

```{eval-rst}
.. automodule:: axio.agent
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.tool
   :members:
   :undoc-members:
   :show-inheritance:
```

## Instructions

`axio.instructions.Instruction` is an abstract base class for instructions in
general: rules, reference material, procedures, and skills. A skill is one use
of this contract. Core does not interpret skill formats or provider locations.

A provider supplies an identifier, a short summary, an optional source label,
and an asynchronous `load()` implementation. Reading the metadata does not load
the content. Use an identifier that distinguishes sources with the same name.

<!-- name: test_instruction_provider -->
```python
from __future__ import annotations

import asyncio

from axio.instructions import Instruction, LoadedInstruction
from axio.tool import Tool


async def convert_temperature(celsius: float) -> str:
    """Convert degrees Celsius to degrees Fahrenheit."""
    return str(celsius * 9 / 5 + 32)


class TemperatureInstructions(Instruction):
    async def load(self) -> LoadedInstruction:
        return LoadedInstruction(
            "Use convert_temperature and include the unit in the answer.",
            (Tool(name="convert_temperature", handler=convert_temperature),),
        )


async def example() -> None:
    instruction = TemperatureInstructions("example:temperature", "Temperature conversion")
    assert instruction.summary == "Temperature conversion"
    content = await instruction.load()
    assert await content.tools[0](celsius=0.0) == "32.0"


asyncio.run(example())
```

Instructions without tools return `LoadedInstruction(text)`. Each call to
`load()` requests current content; providers may cache it only while they can
validate its freshness. Loading does not execute the described procedure or
change an agent. The caller decides when to load mandatory or optional content.

Exceptions and cancellation propagate to the caller. A provider cleans up
resources acquired by a failed load and owns sessions used by returned tools.
Those sessions must remain open while the tools are in use. Tools keep their
guards and access declarations; loading an instruction does not grant access.

### Use instructions with an agent

Pass instruction objects in `Agent(instructions=[...])`. The agent adds their
short summaries and sources to the system prompt and exposes `load_instruction`.
The model calls this tool with an instruction ID to read the full content.
Associated tools become available on the next model iteration. They cannot run
in the same batch that loads them.

For instructions that must apply before the first model request, pass their IDs
in `required_instructions`. These instructions are refreshed before each model
iteration. Providers should reuse their tool objects when the tools are unchanged.
Conflicting instruction IDs or tool names raise an error; they never silently
replace another provider's declaration.

<!-- name: test_agent_instruction_loading -->
```python
from __future__ import annotations

import asyncio

from axio import Agent, MemoryContextStore
from axio.instructions import Instruction, LoadedInstruction
from axio.testing import StubTransport, make_text_response, make_tool_use_response


class StyleRules(Instruction):
    async def load(self) -> LoadedInstruction:
        return LoadedInstruction("Use short sentences and state assumptions explicitly.")


async def example() -> None:
    rules = StyleRules("example:style", "Writing conventions")
    transport = StubTransport([
        make_tool_use_response("load_instruction", tool_input={"id": rules.id}),
        make_text_response("The rules are loaded."),
    ])
    agent = Agent("Answer the user.", transport, instructions=[rules])
    assert await agent.run("Read the writing conventions.", MemoryContextStore()) == "The rules are loaded."


asyncio.run(example())
```

The agent collects base tools and loaded instruction tools, applies `tool_guards`,
and then calls its `ToolSelector`. Only tools in that iteration's selected set
can execute. Application guards supplement the guards supplied by each tool.
They also apply to tools added by instructions and to `load_instruction` itself.

`before_turn_hooks` and `before_tools_hooks` run in tuple order after the existing
`before_turn` and `before_tools` callbacks. Tool hooks receive the calls that
earlier hooks did not defer. Each hook must return results with distinct IDs
from those calls. A failed hook prevents the whole batch from executing.

A preflight hook can call `await agent.load_instruction(id)` to load newly
applicable mandatory instructions. If content changes, the whole batch is
deferred so the model can reconsider it with the updated instructions. Unchanged
content and identical tool objects do not trigger another deferral. Hook errors
and cancelled runs discard staged changes.

`system_contributions` contains synchronous callbacks that supply additional
prompt sections. The agent combines the base `system`, these sections, and its
current instruction content without changing the base string.

Replace `agent.instructions` between runs to reload a catalog. Removed or replaced
objects lose their active content and associated tools at the next iteration.
`agent.copy()` creates independent instruction state and a separate catalog list.
The caller must keep each provider open while any agent uses its tools.
Pass initialized providers with `Agent(..., providers=[provider])`. The agent reads
their catalogs at each iteration; applications own initialization, reload, and cleanup.
`agent.get_tools()` returns currently available tools before selection, including
activated tools and application guards.

`Instruction.load_access()` can declare the effects of loading content for
permission guards. The default is unknown effects. Local skills declare their
source file as a read. These declarations do not grant permission. Required
instructions are loaded by the application-controlled preflight path.

```{eval-rst}
.. automodule:: axio.instructions
   :members:
   :undoc-members:
   :show-inheritance:
```

## Provider discovery

`axio.plugins` loads installed entry points into `PluginCatalog`. Each catalog
retains entries, load errors, and the declaring distribution. The first
declaration reserves its name even when loading fails. `require(name)` reports
a stored failure or lists the available names when the name is unknown.

The groups have separate contracts:

| Group | Target |
|---|---|
| `axio.tools` | A plain async tool handler |
| `axio.tools.providers` | A zero-argument `ToolProvider` factory |
| `axio.guards` | A `PermissionGuard` subclass |
| `axio.selector` | A `ToolSelector` class |
| `axio.transport` | A transport class or plugin, through `discover_transports()` |

A tool provider implements `init(config=None, global_config=None)`, `close()`,
the `all_tools` property, and `get_instructions()`. The last method returns
`Instruction` objects with metadata available before loading. Provider factories
must not acquire resources: initialization and cleanup belong to the lifecycle.

Register a provider in its package:

```toml
[project.entry-points."axio.tools.providers"]
example = "example_tools:Provider"
```

Use `open_tool_providers()` as an async context manager. It registers cleanup
before each initialization and closes providers in reverse order, including
on partial startup, cancellation, and exceptions in the context body. A startup
failure propagates after cleanup. Use separate contexts to isolate independent
startup failures. Providers must permit `close()` after partial initialization.

`collect_tools()` reads the current tool sets and rejects conflicting names.
Multiple references to the same tool object are allowed. It does not cache a
provider's tools, so callers can collect them again after a provider reload.

`standalone_tools()` excludes plain handlers also supplied by an initialized
provider from the same distribution. Tools from different distributions still
conflict when their names match.

The TUI retains its settings groups and `ToolsPlugin` interface. Settings screens
are separate from the core provider contract. Core imports no UI implementation.

```{eval-rst}
.. automodule:: axio.plugins
   :members:
   :undoc-members:
   :show-inheritance:
```

## Context and events

```{eval-rst}
.. automodule:: axio.context
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.events
   :members:
   :undoc-members:
   :show-inheritance:
```

## Protocols

Transport discovery is available to user code through `axio.transport`.
`discover_transports()` returns a `TransportCatalog`; use `require(name)` to get
a `TransportEntry`, `open()` to enter its lifetime, and `select_model()` to choose
a model. See {doc}`../guides/choosing-transports` for runnable application examples,
authentication settings, and error handling. These helpers do not require REPL
or TUI and are not re-exported from top-level `axio`.

```{eval-rst}
.. automodule:: axio.transport
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.permission
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.selector
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.realtime
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data types

```{eval-rst}
.. automodule:: axio.messages
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.blocks
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.types
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.models
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.field
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.schema
   :members:
   :undoc-members:
   :show-inheritance:
```

## Credentials

`axio.credentials` provides typed credential storage without provider dependencies.
`Credentials[T]` supplies locking, serialization, and record selection for the
file and SQLite backends. `CredentialStore[T]` is the protocol for other backends.

A provider supplies its dataclass and a default filename. Mark exactly one string
field with `metadata={"credential_key": True}`. The key can be inside a nested
dataclass. Use `repr=False` for secret fields.

<!-- name: test_generic_credentials -->
```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from axio.credentials import FileCredentialStore


@dataclass(frozen=True, slots=True)
class ServiceAccount:
    name: str = field(metadata={"credential_key": True})
    api_key: str = field(repr=False)


class ServiceCredentials(FileCredentialStore[ServiceAccount]):
    filename = "service.auth"


async def example(path: Path) -> None:
    store = ServiceCredentials(path)
    async with store.lock("work"):
        await store.save(ServiceAccount("work", "example-key"))
    async with store.lock("work"):
        account = await store.load("work")  # ServiceAccount | None
        assert account == ServiceAccount("work", "example-key")


with TemporaryDirectory() as directory:
    asyncio.run(example(Path(directory) / "service.auth"))
```

For a caller-selected path, `FileCredentialStore[ServiceAccount](path)` also
retains the record type. Alternatively, pass `record_type=ServiceAccount`.
`SQLiteCredentialStore[ServiceAccount](database_path)` has the same record API.
The default file path is `$XDG_CONFIG_HOME/axio/<filename>`, falling back to
`~/.config/axio/<filename>` when `XDG_CONFIG_HOME` is unset or relative.

The shared codec uses `asdict()` and restores nested dataclasses, optional values,
unions, literals, enums, lists, tuples, sets, frozensets, and dictionaries with
string keys. Scalar types are checked without coercing strings or booleans into
numbers. Non-finite numbers, missing fields, and unknown fields are rejected.
Dataclass `__post_init__` methods can check relationships between fields; raise
`ValueError` for invalid data. Codec errors do not include stored values.

Call `load()` and `save()` under the corresponding `lock()`. Each save commits
independently, even if a later operation fails. Cancellation waits for an active
write before releasing the lock. Keep locks held across credential renewal and
its writes. Local backends require POSIX permissions and `flock`; stored data is
not encrypted. Preserve the adjacent `.locks` directory while processes use it.

`list_records()` returns complete records, including secrets. Applications must
remove secret fields before displaying account metadata. `select_account()` and
`selected_account()` store a preference for future sessions; `host_id()` retains
a stable local identity. The storage layer does not perform OAuth or refresh.

For Redis, PostgreSQL, or another backend, implement `CredentialStore[T]` with
`lock(key)`, `load(key)`, and `save(record)`. Use `record_id(record)` for the key
and `CredentialCodec(T)` for shared validation and serialization. Distributed
locks must remain exclusive throughout each operation. Saves must be atomic and
durable before returning.

Local backend extensions override `read_value()` and `write_value()` with
dictionary values. These methods run in worker threads; JSON encoding belongs
at the persistence boundary. SQLite opens and closes a connection within each
worker-thread operation. Existing ChatGPT JSON records and SQLite databases
remain readable, with their original lock names and recovery markers.

```{eval-rst}
.. autoclass:: axio.credentials.Credentials
   :members:

.. autoclass:: axio.credentials.CredentialStore
   :members:

.. autoclass:: axio.credentials.FileCredentialStore
   :members:

.. autoclass:: axio.credentials.SQLiteCredentialStore
   :members:

.. autoclass:: axio.credentials.CredentialCodec
   :members:
```

## Utilities

```{eval-rst}
.. automodule:: axio.stream
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.agent_loader
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.tool_args
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.compaction
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: axio.testing
   :members:
   :undoc-members:
   :show-inheritance:
```
