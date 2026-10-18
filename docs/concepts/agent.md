# Agent & the Agentic Loop

The {class}`Agent` is the central orchestrator. It connects a transport, a set of
tools, and a context store into a single loop that streams LLM responses and
dispatches tool calls until the model signals it is done.

## Configure an agent

Start with a transport, a role, and the tools your assistant needs. Supply a
separate context store for each conversation:

<!-- name: test_agent_dataclass -->
```python
import asyncio

from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response

agent = Agent(
    system="Help with the user's project.",
    transport=StubTransport([make_text_response("Ready.")]),
)
assert asyncio.run(agent.run("Hello", MemoryContextStore())) == "Ready."
```

The stub makes this example independent of an API account. For a real provider,
follow {doc}`../guides/choosing-transports`. An agent definition can also carry
{doc}`instructions`, initialized {ref}`tool providers <tool-providers>`,
application {doc}`guards`, and hooks that update state before a request or tool
batch. These parts compose on the same agent.

### Basic settings

`system`
: The system prompt sent with every request.

`transport`
: Any object satisfying the {ref}`CompletionTransport <protocols>` protocol.

`tools`
: Available tools. The agent searches this list by name when the model
  issues a tool call. Defaults to an empty list.

`selector`
: An optional {ref}`ToolSelector <tool-selector>` that filters the active tool
  list before each iteration. When `None`, all tools are passed to the
  transport on every iteration.

`max_iterations`
: Safety limit preventing runaway loops. The agent emits a
  `SessionEndEvent` with an error if this limit is reached. Defaults to 50.

`last_iteration_message`
: An optional `Message` appended to the effective history **only** on the
  final iteration (when `max_iterations` is about to be exceeded). Useful for
  injecting a stop instruction such as "you must answer now without calling
  more tools" to coerce a final response before the loop terminates.

`before_turn`
: Optional async callback receiving the iteration number before each model
  request. It can refresh application state or `agent.system`. An exception
  ends the run with `Error` and `SessionEndEvent(error)` before the model is called.

`before_tools`
: Optional async callback receiving all valid calls in a batch and its iteration
  number. It runs before any handler, including streaming handlers. Return a
  `ToolResultBlock` for each call that must not execute; return an empty list
  to allow the whole batch. Returned IDs must be distinct and belong to that
  batch. The agent stores and emits these results like ordinary tool results.
  An exception or invalid result IDs fail the entire batch without executing
  handlers. Cancellation uses the normal interrupted-call results.

These callbacks are keyword-only fields and survive `agent.copy()`. Their
policy belongs to the application. Axio does not discover instruction files.

### Guidance and integrations

`instructions`
: A catalog of {doc}`instructions` the model can choose to load. Each has a
  short summary and an async loader for its full text and associated tools.

`required_instructions`
: IDs of instructions the application requires before every model request.
  Use this for mandatory project rules rather than relying on model choice.

`providers`
: Initialized {ref}`tool providers <tool-providers>` whose current tools and
  instruction catalogs are read at every iteration. The application opens and
  closes them; the agent does not own their resources.

`tool_guards`
: Application guards appended to every available tool, including instruction
  loaders and tools added later. Original tool guards run first.

(agent-hooks)=
## Update state before requests and tool calls

Hooks let integrations add behavior without replacing another integration's
callbacks. Use `before_turn_hooks` for async preparation before each model
request, `system_contributions` for current prompt sections, and
`before_tools_hooks` to check a whole batch before any handler runs.

For example, publish the current application mode without rewriting the base
system prompt:

<!-- name: test_agent_system_contribution -->
```python
import asyncio

from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response


async def main() -> None:
    state = {"mode": "review"}
    prepared = []

    async def prepare(iteration: int) -> None:
        prepared.append(iteration)

    agent = Agent(
        system="Help with the project.",
        transport=StubTransport([make_text_response("Reviewed.")]),
        before_turn_hooks=(prepare,),
        system_contributions=(lambda: f"Current application mode: {state['mode']}",),
    )
    await agent.run("Review the changes.", MemoryContextStore())
    assert prepared == [1]
    assert agent.system == "Help with the project."


asyncio.run(main())
```

Before each request, the agent runs `before_turn`, then `before_turn_hooks` in
tuple order. It refreshes required instructions, activates staged content and
tools, and constructs the effective prompt from `system`, synchronous
`system_contributions`, and instruction content. It then applies tool guards
and selection. Contributions should return text promptly; put asynchronous I/O
in a preparation hook. A preparation failure ends the run before a model request.

After the model requests tools, `before_tools` runs first, then
`before_tools_hooks` in tuple order. Each receives only the calls not already
deferred by an earlier hook. Return replacement `ToolResultBlock`s for calls
that must not execute, or `[]` to allow the remaining calls. IDs must be unique
and belong to that hook's input. Invalid results or exceptions prevent execution
of the whole batch.

A scoped-instruction hook can call `await agent.load_instruction(id)`. If this
changes active guidance or associated tools, the batch is deferred until the
model sees the update. Unchanged text and identical tool objects do not cause
another deferral. Failed hooks and cancellation discard staged instruction changes.
`LocalToolProvider.bind()` installs this behavior for local file instructions;
applications normally use the provider rather than implementing it themselves.

`agent.copy()` keeps configuration and hooks, copies the catalog lists, and starts
with fresh instruction activation. It shares the referenced transports, providers,
and other resource-owning objects. Keep their lifetimes open for all copies.

## How the loop works

```{mermaid}
flowchart TD
    A[User message] --> B[Append to context]
    B --> C[Get history from context]
    C --> D[Stream from transport]
    D --> E{Tool calls?}
    E -- Yes --> F[Dispatch tools concurrently]
    F --> G[Append results to context]
    G --> C
    E -- No --> H[Append assistant turn to context]
    H --> I{Stop reason}
    I -- end_turn --> J[SessionEndEvent end_turn]
    I -- refusal --> K[SessionEndEvent refusal]
    I -- pause_turn --> C
    I -- incomplete or unknown --> M[SessionEndEvent with the reason]
    I -- error --> L[Error, then SessionEndEvent error]
```

1. The user message is appended to the context store, prefixed with a local
   timestamp. The stored text is `[2026-08-27 14:03:11 CEST] your message`, not
   the string you passed. The model needs to know when it was asked. The
   history is the only place to say it.
2. The agent retrieves the full conversation history - including the message
   just appended - and streams it to the transport along with the tool
   definitions and system prompt. On the final iteration only,
   `last_iteration_message` is appended to the history handed to the transport.
   It is never written to the store.
3. Every event the transport yields is passed through to the consumer
   unchanged. The agent builds the stored turn from text, refusal text, reasoning,
   signed content, generated media, and replayable provider output. Tool-call
   fragments are buffered until the iteration ends.
4. When the transport yields `IterationEnd`, the buffered fragments are parsed
   into `ToolUseBlock`s. The turn's token usage is then added to the running
   total and reported to the context store.
   - Tool calls run only when the stop reason is `tool_use`, `end_turn`, or
     `pause_turn`. Hooks preflight the batch, then valid remaining calls run
     concurrently. The agent appends the assistant turn and results and loops.
     Calls from a refused, truncated, cancelled, unknown, or failed turn receive
     error results without executing their handlers.
   - Otherwise the assistant turn is appended to context. The stop reason
     then decides what happens next.
5. If `max_iterations` is exceeded, the loop ends with a `SessionEndEvent`
   carrying `StopReason.error`.

## Stop reasons and how the loop ends

The assistant turn is stored **before** the stop reason is examined, so a
refusal, a pause and an unrecognised reason all keep the content the model
produced.

| Stop reason | What the loop does |
|---|---|
| `tool_use` | Dispatches valid calls and iterates; an empty turn with no call is an error. |
| `end_turn` | Ends with `SessionEndEvent(stop_reason=end_turn)`. |
| `refusal` | Ends with `SessionEndEvent(stop_reason=refusal)`. Not an `Error`. |
| `pause_turn` | **Resumes**: the loop goes round again. |
| `max_tokens`, `context_window_exceeded`, `cancelled`, `unknown` | Ends with the same reason, preserving partial output. |
| `repetition` | Axio stops repetitive output and reports the interruption. |
| anything else | Yields `Error`, then `SessionEndEvent(stop_reason=error)`. |

`refusal` is terminal and deliberately not reported as an error. The model
declined. The decline is stored as the turn's content. The same prompt sent
again will be declined again. Reported as an error, the decline is
indistinguishable from a broken connection. A caller then retries something
that can never work. The decline itself reaches `run()` as the returned text.
See {doc}`events`.

`pause_turn` is the one reason that does not end the run. The provider stopped
its own server-side tool loop and expects the assistant content back so it can
finish. That content was appended a line earlier, so going round again *is* the
resume. The resume takes the same code path as any other iteration, bounded by
the same `max_iterations`.

Inspect `SessionEndEvent.stop_reason` when displaying a result: partial text can
be useful while still being unfinished. `axio.INCOMPLETE` groups `max_tokens`,
`context_window_exceeded`, `cancelled`, and `unknown` for that purpose. `run()`
returns their collected text, so use `run_stream()` when the interface must
distinguish a complete answer from a truncated one. `IterationEnd.raw` retains
the provider's unmapped stop value for an `unknown` reason.

Other reasons, including `error` and unsupported additions, end with an `Error`
and `SessionEndEvent(error)`. A transport that ends without `IterationEnd` also
fails; an interrupted stream is not a completed answer.

## What the agent stores

The assistant message written to context holds more than text and tool calls:

`TextBlock`
: Accumulated `TextDelta`s, plus the text of any `Refusal`. A refusal is what
  the assistant said. Left out, the stored turn is empty. The next request
  then carries a blank assistant message the provider rejects.

`ReasoningBlock`
: Accumulated `ReasoningDelta`s, with the `ReasoningSignature` that proves them
  attached. This is what makes the turn replayable. Anthropic refuses a
  returned thinking block whose signature is missing or changed. Google reports
  `MISSING_THOUGHT_SIGNATURE` for the same failure.

  A signed block is finished. A reasoning delta arriving after a signature
  starts a **new** `ReasoningBlock` rather than extending the signed one,
  because the provider computed that signature over the text it had.

  A signature repeating the index of the one before it is the rest of the same
  proof, and is appended to it. Transports index a signature by the block it
  proves, so a repeated index cannot be a second proof. Stored apart, the block
  replays with half a signature. The turn after it is then refused.

`ImageBlock` / `AudioBlock` / `VideoBlock`
: Media the model generated inline, so a later turn still has it in view.

`ToolUseBlock`
: One per parsed tool call, added when the iteration ends.

`ProviderBlock`
: Replayable output from tools the model provider ran itself. Its opaque payload
  must survive storage so the originating transport can send it back. See
  {doc}`context` and {doc}`events`.

A context store that round-trips these through `to_dict`/`from_dict` must keep
`ReasoningBlock.signature` intact. Dropping it fails on the *next* turn, not on
the one that dropped it.

## When the model repeats itself

The agent watches accumulated text for a model stuck in a loop - a repeated
token, phrase, or paragraph. When it fires, the agent stops reading the stream,
appends `[Output truncated: repetitive content detected]` to the turn, yields
that note as a `TextDelta`, stores the turn, and ends the session with
`StopReason.repetition`. The stream was abandoned before `IterationEnd`, so no
usage arrived with it. The agent reads `transport.last_usage` if the transport
keeps one, and adds it, rather than reporting the turn as free.

## Streaming API

`Agent` exposes two methods:

`run_stream(user_message, context) -> AgentStream`
: Returns an `AgentStream` - an async iterator over `StreamEvent` values.
  Use this when you need per-token streaming or want to observe tool calls
  as they happen.

`run(user_message, context) -> str`
: Convenience wrapper that consumes the stream and returns the final text. The
  text of a `Refusal` counts as final text, so a declined turn returns the
  decline rather than the empty string it used to. Raises `StreamError` on an
  `Error` event.

## Concurrent tool dispatch

When the model requests multiple tool calls in a single response, the agent
runs them all concurrently via `asyncio.gather`. The public method signature is:

```python
from axio.blocks import ToolResultBlock, ToolUseBlock


async def dispatch_tools(
    self,
    blocks: list[ToolUseBlock],
    iteration: int,
) -> list[ToolResultBlock]: ...
```

Each tool call goes through the full guard chain before execution. If a tool
raises an exception, the agent catches it and wraps it in a `ToolResultBlock`
with `is_error=True`. The model sees the error and can react accordingly.

If a tool's JSON arguments could not be parsed from the stream, the agent
returns a `ToolResultBlock` with `is_error=True` and a message asking the
model to retry with valid JSON, rather than passing malformed input to the
handler.

## ToolSelector

(tool-selector)=

The `ToolSelector` protocol lets you trim the active tool list before each
iteration. This is useful for reducing noise in the model's context, enforcing
capability restrictions, or implementing dynamic tool routing.

```python
from typing import Any, Protocol, runtime_checkable
from collections.abc import Iterable
from axio.messages import Message
from axio import Tool


@runtime_checkable
class ToolSelector(Protocol):
    async def select(
        self, messages: Iterable[Message], tools: Iterable[Tool[Any]]
    ) -> Iterable[Tool[Any]]: ...
```

Pass a `ToolSelector` via the `selector` field when constructing an `Agent`.
On each iteration the agent calls `selector.select(history, tools)` and passes
only the returned subset of tools to the transport.

When `selector` is `None` (the default) all tools are passed on every
iteration.

## Copying an Agent

`Agent.copy(**overrides)` returns a new `Agent` with selected fields replaced.
Because `Agent` uses `slots=True`, this is the correct way to derive a
modified agent without mutating the original:

<!-- name: test_agent_copy -->
```python
import asyncio
from axio import Agent
from axio.testing import StubTransport, make_text_response

transport = StubTransport([make_text_response("ok")])
agent = Agent(system="You are helpful.", transport=transport)

# Derive an agent with a different system prompt
strict_agent = agent.copy(system="Be concise. Answer in one sentence.")
assert strict_agent.system == "Be concise. Answer in one sentence."
assert strict_agent.transport is agent.transport  # shared by default
```
