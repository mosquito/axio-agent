# Instructions and skills

An agent often needs a small set of rules on every request and a larger body of
guidance only for particular tasks. For example, a coding assistant should always
follow the project's conventions, but it needs the release procedure only when
preparing a release. Sending every procedure in every prompt makes unrelated
material compete with the current task.

An `Instruction` gives the model a short description first. Its full text loads
when needed, together with any tools required to follow it. The same mechanism
supports project rules, reference material, runbooks, and skills.

## Choose where guidance belongs

| Guidance | How to supply it |
|---|---|
| The agent's permanent role and behavior | `Agent(system=...)` |
| A procedure or reference the model may choose to read | `Agent(instructions=[...])` |
| Content that must be present before a model request | Add its ID to `required_instructions` |
| Scoped project files or skills supplied by an integration | Attach an initialized tool provider |

Instructions describe what to do. Tools perform actions, and guards decide
whether those actions are permitted. Text in an instruction cannot grant a tool
permission. See {doc}`tools` and {doc}`guards` for those parts of the application.

## Load a procedure and its tools

Suppose a release procedure needs a tool that checks the release status. Create
an instruction with a stable ID, a summary that tells the model when it applies,
and an async `load()` method:

<!-- name: test_concept_instruction_activation -->
```python
import asyncio

from axio import Agent, MemoryContextStore, Tool
from axio.instructions import Instruction, LoadedInstruction
from axio.testing import StubTransport, make_text_response, make_tool_use_response


async def release_status() -> str:
    """Report whether the release is ready."""
    return "All required checks passed."


class ReleaseProcedure(Instruction):
    def __init__(self) -> None:
        super().__init__(
            "project:release",
            "Read before preparing a release; includes the release status check.",
            source="Project release handbook",
        )
        self.status_tool = Tool(name="release_status", handler=release_status)

    async def load(self) -> LoadedInstruction:
        return LoadedInstruction(
            "Call release_status before reporting readiness. Do not publish a release.",
            tools=(self.status_tool,),
        )


async def main() -> None:
    procedure = ReleaseProcedure()
    transport = StubTransport([
        make_tool_use_response("load_instruction", tool_input={"id": procedure.id}),
        make_tool_use_response("release_status"),
        make_text_response("The release is ready for review."),
    ])
    agent = Agent(
        system="Help prepare releases.",
        transport=transport,
        instructions=[procedure],
    )
    assert "release_status" not in {tool.name for tool in agent.get_tools()}
    reply = await agent.run("Check release readiness.", MemoryContextStore())
    assert reply == "The release is ready for review."
    assert "release_status" in {tool.name for tool in agent.get_tools()}


asyncio.run(main())
```

The stub makes the example run without credentials. With a real transport, the
model chooses whether to load the procedure from its summary. The sequence is:

1. The first request includes the instruction's ID, summary, and source, plus a
   `load_instruction` tool. Its full text and `release_status` are not yet active.
2. The model calls `load_instruction(id="project:release")` and receives the text.
3. On the next model iteration, the full text is included in the system prompt
   and the associated tool is available.
4. The model can call that tool and use its result to answer.

Loading a procedure does not execute its tools. Calls to a newly loaded tool in
the same batch as `load_instruction` cannot run: that tool becomes available at
the next model boundary. A selector must retain `load_instruction` when optional
instructions should be usable, and the selected model must support tool calls.

Keep tools on the instruction object when their implementation is unchanged,
as above. Creating different tools with an already-used name causes a conflict.
Instructions without associated tools simply return `LoadedInstruction(text)`.

## Require guidance before the first request

Use `required_instructions` for content whose loading is an application decision.
For example, a release-only assistant should not depend on the model choosing
to read the release procedure. Using the class above:

<!-- name: test_concept_instruction_activation -->
```python
async def required_example() -> None:
    procedure = ReleaseProcedure()
    agent = Agent(
        system="Review release readiness.",
        transport=StubTransport([make_text_response("Ready for review.")]),
        instructions=[procedure],
        required_instructions=(procedure.id,),
    )
    await agent.run("Review this release.", MemoryContextStore())
    assert "release_status" in {tool.name for tool in agent.get_tools()}


asyncio.run(required_example())
```

Required instructions are loaded before each model iteration, including the first.
Their loaders should return current content. A failed required load stops the run
before the next model request; it does not silently continue without the rules.
Optional loading failures are reported as tool errors for the model to handle.

## Use local and MCP skills

A skill is a task-oriented instruction, often accompanied by scripts or reference
files. Core uses the same `Instruction` contract for it. Providers understand
where that content lives and how to read it:

- `LocalToolProvider` from `axio-tools-local` loads scoped `AGENTS.md` files and
  discovers local `SKILL.md` documents. It supplies local file and execution tools.
- `MCPToolProvider` from `axio-tools-mcp` exposes server guidance, skills, and
  resource catalogs through its open MCP sessions.

Attach providers with `Agent(providers=[...])`; see {ref}`tool-providers` for a
complete local example and {doc}`../guides/mcp-tools` for MCP. In REPL and TUI,
the applications configure these providers. Their `/skills` and `/instructions`
commands show available sources; see {doc}`../guides/axio-repl`.

Local `AGENTS.md` rules apply by directory scope. The local provider checks a
tool's declared file effects before execution. If entering a subtree reveals new
or changed rules, it defers the batch so the model can reconsider with the new
content. Paths hidden inside shell commands or opaque MCP calls cannot be
inferred this way. Instruction scope is not filesystem isolation.

## Refresh content and own its lifetime

An ID must distinguish its source, even when two documents have the same title.
`source` is a display label; core does not open it as a path or URI. A loader
must obtain content itself and may cache it only while it can validate freshness.

The agent reads provider catalogs at each iteration. Removing or replacing an
instruction object removes its previously active text and tools at the next
iteration. Reload through the owning provider, or replace `agent.instructions`
between runs. An optional instruction with unchanged identity stays active until
removed, replaced, or explicitly loaded again; editing its backing data alone
does not force an optional reload.

Activation state belongs to one agent. `agent.copy()` starts with fresh activation
state, while providers and their open sessions can still be shared. Keep those
resources open for every agent using them, and bind copies separately when a
provider requires per-agent hooks. `agent.load_instruction(id)` stages content
for the next iteration; application hooks can use it for mandatory preflight.
See {ref}`agent-hooks` for that integration and {doc}`../api/axio` for signatures.
