(protocols)=

# Protocols

Axio's extensibility comes from a small set of **runtime-checkable protocols**
and abstract base classes. Implement any of them to plug your own components
into the framework — no subclassing the agent, no monkey-patching.

```{mermaid}
classDiagram
    class CompletionTransport {
        <<Protocol>>
        +stream(messages, tools, system) AsyncIterator~StreamEvent~
    }
    class ContextStore {
        <<ABC>>
        +append(message)*
        +get_history() list~Message~*
        +session_id str
        +fork() ContextStore
        +clear()
        +close()
        +list_sessions() list~SessionInfo~
    }
    class PermissionGuard {
        <<ABC>>
        +check(handler) handler*
    }
    Agent --> CompletionTransport
    Agent --> ContextStore
    Tool --> PermissionGuard
```

## CompletionTransport

The transport protocol has a single method:

<!-- name: test_completion_transport_protocol -->
```python
from typing import runtime_checkable, Protocol
from collections.abc import AsyncIterator
from axio.messages import Message
from axio.tool import Tool
from axio.events import StreamEvent


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]: ...
```

The agent calls `stream()` on every iteration, passing the full conversation
history, the available tools, and the system prompt. The transport yields
`StreamEvent` values as they arrive from the LLM.

Available transports (each in its own installable package):

| Transport | Package | Notes |
|---|---|---|
| `AnthropicTransport` | `axio-transport-anthropic` | Anthropic Claude models |
| `OpenAITransport` | `axio-transport-openai` | OpenAI and OpenAI-compatible APIs |
| `CodexTransport` | `axio-transport-codex` | ChatGPT via OAuth |

The core `axio` package does not bundle any transport implementation — install
the appropriate package for your model provider.

See [Writing Transports](../guides/writing-transports.md) for a step-by-step guide.

## ContextStore

The context store manages conversation history. It is an abstract base class
with async methods. Only `append` and `get_history` are truly abstract —
everything else has a working default implementation:

<!-- name: test_context_store_abc -->
```python
from axio.messages import Message
from axio.context import ContextStore


class MyContextStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    # Default implementations provided by ContextStore (override as needed):
    #   session_id          — lazy UUID hex property
    #   clear()             — raises NotImplementedError by default
    #   fork()              — deep-copies history into a MemoryContextStore
    #   close()             — no-op by default
    #   set_context_tokens(input, output)  — no-op by default
    #   get_context_tokens()               — returns (0, 0) by default
    #   add_context_tokens(input, output)  — increments via get/set above
    #   list_sessions()     — returns a single SessionInfo for the current session

store = MyContextStore()
assert store.session_id  # auto-generated UUID hex
```

Built-in implementations:

- `MemoryContextStore` (in `axio`) — in-memory, no persistence; ideal for
  short-lived agents, tests, and prototypes.
- `SQLiteContextStore` (in `axio-context-sqlite`) — persistent, SQLite-backed;
  survives process restarts and supports multiple named sessions per project.

Implement your own `ContextStore` to back conversations with Redis, a database,
or any other storage layer.

See [Context & Messages](context.md) for details on the message model.

## PermissionGuard

Guards gate tool execution. They sit between parameter validation and handler
invocation. `PermissionGuard` is an abstract base class (ABC) — not a
Protocol. Subclass it and implement `check()`:

<!-- name: test_permission_guard_abc -->
```python
from typing import Any
from axio.permission import PermissionGuard


class MyGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        # return handler to allow, raise GuardError to deny
        return handler
```

A guard receives the validated handler instance and either returns it (allowing
execution) or raises `GuardError` to deny. Guards can also modify the handler
before returning it.

Tool calls are made via `await guard(handler)`, which delegates to `check()`.
The `ConcurrentGuard` subclass additionally wraps `check()` in an
`asyncio.Semaphore` to control parallelism.

Axio ships three built-in guards:

`AllowAllGuard`
: Always returns the handler unchanged — useful as a no-op default.

`DenyAllGuard`
: Always raises `GuardError("denied")` — useful for locked-down environments.

`ConcurrentGuard`
: Abstract base that serializes (or rate-limits) concurrent `check()` calls
  via a semaphore. Set the `concurrency` class attribute to control
  parallelism (default: 1).

Multiple guards compose sequentially — each guard's output is passed to the
next.

See [Guards](guards.md) for the full guard system and
[Writing Guards](../guides/writing-guards.md) for a how-to guide.

## Additional transport protocols

Beyond `CompletionTransport`, Axio defines protocols for other AI modalities:

`ImageGenTransport`
: `async generate(prompt, *, size, n) -> list[bytes]`

`TTSTransport`
: `synthesize(text, *, voice) -> AsyncIterator[bytes]`

`STTTransport`
: `async transcribe(audio, media_type) -> str`

`EmbeddingTransport`
: `async embed(texts) -> list[list[float]]`

These follow the same pattern — implement the protocol, pass the object in,
and the framework handles the rest.
