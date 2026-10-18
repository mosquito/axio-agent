# Core Concepts

Axio is assembled from a small set of well-defined building blocks. Each one
has a single responsibility and a stable interface. Swap any of them without
touching the rest.

## Building blocks

### Agent

The {doc}`Agent <agent>` is the central orchestrator. It is a `slots=True`
dataclass that wires a transport, a context store, and a set of tools into a
loop:

1. Send conversation history to the transport.
2. Collect `StreamEvent` values as they arrive, forwarding every one of them to
   the consumer and folding text, refusals and reasoning into the turn.
3. When a completed turn requests tools, preflight the batch, dispatch valid
   calls concurrently, append the results, and loop.
4. Otherwise store the turn and let the stop reason decide. `end_turn` completes
   the run, `pause_turn` resumes it, and refusal or incomplete generation ends
   with its own reason. Transport failures produce an `Error`.

Derive a configured agent with `agent.copy(**overrides)` rather than mutating
one.

The agent is deliberately thin. It has no opinions about retries, rate limits,
or logging. Those belong to the transport or the application layer.

### Instructions and skills

{doc}`Instructions <instructions>` let an agent discover guidance from short
summaries, then load full procedures and their tools when needed. Applications
can require rules before the first request. Local and MCP providers supply
project instructions and skills through the same mechanism.

### Transport

A {doc}`transport <protocols>` is any object that implements the
`CompletionTransport` protocol - a single `stream()` method that takes the
conversation history and yields `StreamEvent` values. The core package ships no
transport of its own. Install the one that matches your model provider:

| Package | Provider |
|---|---|
| `axio-transport-anthropic` | Anthropic Claude |
| `axio-transport-openai` | OpenAI and any OpenAI-compatible API |
| `axio-transport-codex` | ChatGPT via OAuth |
| `axio-transport-google` | Google Gemini (Developer API and Vertex AI) |

Because a transport is just a protocol, you can implement your own with nothing
more than `aiohttp`. No SDK is required. The shipped transports read their
`text/event-stream` responses through `axio-sse` rather than parsing SSE by
hand. See {doc}`../guides/writing-transports`.

### Tools

A plain `async def` function is a tool handler. The {doc}`Tool <tools>`
dataclass is the glue that declares it as an agent tool. It attaches a stable
name, generated input schema, description, guards, and an optional concurrency
limit. A tool call is one concrete model request to invoke that definition.

The `context` field on `Tool` lets you pass arbitrary state - a database
connection, a sandbox object, a file path - to the handler via `CONTEXT.get()`
at call time without any global state or class-level variables.

Use a {ref}`tool provider <tool-providers>` when an integration owns a changing
set of tools, instructions, or shared connections. The application opens the
provider, attaches it to an agent, and closes it after agent runs finish.

### Permissions

{doc}`Guards <guards>` enforce tool permissions. Providers declare effects such
as reading or writing a file; `PolicyGuard` combines these declarations with
project rules and user approval. Instructions and tool selection do not grant
permission to perform actions.

### Field Metadata

The {doc}`field metadata <field>` system uses `Field()` to add descriptions,
defaults, and constraints to tool parameters. These annotations are converted
to JSON Schema for LLM consumption.

### JSON Schema Generation

The {doc}`schema builder <schema>` converts Python type annotations into JSON
Schema objects sent to LLM providers in tool definitions.

### Context store

The {doc}`context store <context>` manages conversation history. It is an
abstract base class with two required methods: `append(message)` and
`get_history()`. Built-in options:

- `MemoryContextStore` - in-memory, lives for the duration of a session.
- `SQLiteContextStore` (`axio-context-sqlite`) - persistent across process
  restarts, supports multiple named sessions.

Implement your own to back conversations with Redis, a relational database,
or any other storage layer.

```{toctree}
:maxdepth: 1

agent
instructions
protocols
tools
selector
field
schema
events
tool-args-streaming
context
guards
models
```
