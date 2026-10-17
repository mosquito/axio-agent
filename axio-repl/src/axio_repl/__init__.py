"""Interactive REPL coding assistant powered by axio agent framework.

Auto-detects transport from available API keys (OPENAI_API_KEY, NEBIUS_API_KEY,
OPENROUTER_API_KEY), or use --transport to pick explicitly.

Run:
    axio-repl
    axio-repl "your prompt here"
"""

from __future__ import annotations

import asyncio
import atexit
import importlib.util
import os
import signal
import sys
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.credentials import CredentialError
from axio.events import (
    AudioOutput,
    BlockEnd,
    Citation,
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    Refusal,
    SessionEndEvent,
    TextDelta,
    ToolFieldDelta,
    ToolFieldEnd,
    ToolFieldStart,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
    VideoOutput,
)
from axio.exceptions import StreamError
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.plugins import (
    RefreshableToolProvider,
    ToolProvider,
    discover_tool_providers,
    discover_tools,
    standalone_tools,
)
from axio.tool import Tool
from axio.tool_args import ToolArgStream
from axio.transport import TransportEntry, TransportError, discover_transports, select_transport_model
from axio.types import INCOMPLETE

from .paths import ConfigPaths
from .presentation import describe_local_skills, describe_mcp, explain_config
from .unconfigured import UnconfiguredTransport

if TYPE_CHECKING:
    from .configuration import MCPConfiguration

DEFAULT_MAX_BYTES = 128 * 1024


_readline: Any
try:
    import readline as _readline
except ImportError:
    _readline = None

readline: Any = _readline

AGENT_NAME = "axio-repl"
AGENT_VERSION = "0.2.3"

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


# ── Tools ────────────────────────────────────────────────────────────


# ── Transport auto-detection ─────────────────────────────────────────


async def _select_transport(name: str | None) -> TransportEntry:
    catalog = discover_transports()
    if name:
        return catalog.require(name)
    for entry in catalog.entries.values():
        if not entry.auto_select:
            continue
        try:
            if entry.has_credentials() or (entry.auth is not None and await entry.auth.has_accounts()):
                return entry
        except Exception as error:
            print(f"Transport {entry.name!r} credentials unavailable: {entry.format_error(error)}", file=sys.stderr)
    hints = [f"{', '.join(entry.env_vars)} ({entry.name})" for entry in catalog.entries.values() if entry.env_vars]
    accounts = [entry.name for entry in catalog.entries.values() if entry.auth is not None]
    hint = "Set one of: " + "; ".join(hints) + ". " if hints else ""
    if accounts:
        hint += f"Use --transport <name> to sign in: {', '.join(accounts)}. "
    if not catalog.entries:
        hint += "Install a transport package. "
    raise TransportError("No transports with credentials found. " + hint)


# ── AGENTS.md & system prompt ────────────────────────────────────────


def build_system_prompt(
    root: Path,
    model: ModelSpec,
    tools: list[Tool[Any]],
    agents_text: str = "",
) -> str:
    caps = model.capabilities
    ctx_k = model.context_window // 1000
    out_k = model.max_output_tokens // 1000
    tool_names = ", ".join(t.name for t in tools)

    has_tools = Capability.tool_use in caps

    lines = [
        f"You are {AGENT_NAME} (v{AGENT_VERSION}) — a terminal coding assistant.",
        f"Model: {model.id} ({ctx_k}K context, {out_k}K max output)",
        f"Current directory: {root} (perform actions here unless specified otherwise)",
    ]
    if has_tools:
        lines.append(f"Tools: {tool_names}")
    lines.append("")

    # Capability-aware guidance
    cap_notes: list[str] = []
    if Capability.vision in caps:
        cap_notes.append("You can see images via read_file (screenshots, diagrams, photos).")
    if Capability.audio in caps:
        cap_notes.append("You can listen to audio files via read_file (speech, music, podcasts).")
    if Capability.video in caps:
        cap_notes.append("You can see video files via read_file.")
    if Capability.image_generation in caps:
        cap_notes.append("You can generate images inline — describe what to draw in your response.")
    if Capability.reasoning in caps:
        cap_notes.append("Extended thinking is available for complex reasoning.")
    if cap_notes:
        lines += cap_notes + [""]

    lines.append("Rules:")
    if has_tools:
        lines += [
            "- Start every task by listing the current directory to understand the project.",
            "- Read files before editing. Use line_numbers=True before patch_file.",
            "- Keep edits minimal and targeted — don't reformat surrounding code.",
            "- Ground answers on project context gathered through tools.",
        ]
    lines += [
        "- Write idiomatic code — follow the conventions and best practices of the "
        "languages and frameworks used in the project.",
        "- When the user asks about a file they provided or you read, base your answer "
        "strictly on the actual file contents. Do not guess, assume, or fill in details "
        "from general knowledge — only state what the file actually contains.",
        f"- Your max output is {out_k}K tokens. Use as many as the task requires — "
        "do not stop early. If the user asks for a full transcript, detailed analysis, "
        f"or comprehensive review, produce the complete output up to the {out_k}K limit.",
        "- Never refuse safe requests or claim inability.",
    ]
    if has_tools:
        lines += [
            "- If a tool call fails, analyze the error and try a different approach. "
            "If stuck after 3 attempts at the same sub-problem, "
            "explain what you tried and ask for guidance.",
            "- Do not return a final answer until all necessary work is done or you are stuck.",
            "- For compound requests, build a checklist of all items and verify each is addressed before finishing.",
            "- Don't narrate your tool calls — the user sees their full output.",
            "- After completing work, summarize what changed briefly.",
            "- Not tested — not done. Always run tests or builds to verify your changes. "
            "Re-read edited files, observe actual results — don't assume success "
            "from exit codes alone.",
            "- After any test or build that produces images or video, you MUST read_file "
            "every output file to actually see the results. Never describe visual output "
            "you haven't viewed. 'Tests passed' is not the same as 'I looked at the "
            "screenshots and they look correct'.",
            "- To verify UI, use browser automation (Playwright, Puppeteer) to capture "
            "real screenshots at multiple viewport sizes (desktop 1280×800, tablet 768×1024, "
            "mobile 375×667), then read_file every screenshot.",
            "- When you read a screenshot, you MUST critically analyze it. List every "
            "visual defect you notice: broken layout, text overflow, misaligned elements, "
            "poor contrast, missing images, clipped content, wrong spacing, responsive "
            "issues. Do NOT say 'looks good' unless you can specifically confirm each "
            "aspect is correct.",
            "- UI review is iterative: screenshot → list issues → fix code → re-screenshot "
            "→ verify fixes. Repeat until zero defects. Never declare UI done after a "
            "single screenshot pass.",
            "- Never use generate_image as a substitute for real UI testing.",
            "- Never run destructive shell commands (rm -rf, git reset --hard) without user confirmation.",
            "- For large files, read specific line ranges instead of the entire file.",
        ]
    lines.append("")

    if agents_text:
        lines += ["AGENTS.md instructions:", agents_text, ""]

    return "\n".join(lines)


# ── Readline history ─────────────────────────────────────────────────


def setup_history() -> None:
    if readline is None:
        return
    history_path = Path.home() / ".axio_repl_history"
    if history_path.exists():
        try:
            readline.read_history_file(str(history_path))
        except (OSError, RuntimeError):
            pass
    try:
        readline.set_history_length(5000)
    except (AttributeError, ValueError):
        pass

    def _save() -> None:
        try:
            readline.write_history_file(str(history_path))
        except (OSError, RuntimeError):
            pass

    atexit.register(_save)


# ── Event rendering ──────────────────────────────────────────────────


async def run_prompt(agent: Agent, ctx: MemoryContextStore, prompt: str) -> None:
    if isinstance(agent.transport, UnconfiguredTransport):
        print(agent.transport.reason, file=sys.stderr)
        return
    in_text = False
    reasoning_line_start = True
    reasoning_index: int | None = None
    declined = False
    arg_streams: dict[str, ToolArgStream] = {}
    streamed_tool_ids: set[str] = set()

    async for event in agent.run_stream(prompt, ctx):
        if not reasoning_line_start and (
            isinstance(
                event,
                (
                    TextDelta,
                    Refusal,
                    Citation,
                    ImageOutput,
                    AudioOutput,
                    VideoOutput,
                    ToolUseStart,
                    ToolInputDelta,
                    ToolOutputDelta,
                    ToolResult,
                    IterationEnd,
                    Error,
                    SessionEndEvent,
                ),
            )
            or isinstance(event, BlockEnd)
            and event.index == reasoning_index
            or isinstance(event, ReasoningDelta)
            and event.index != reasoning_index
        ):
            print()
            reasoning_line_start = True
        match event:
            case ReasoningDelta(index=index, delta=delta):
                if not delta:
                    continue
                if in_text:
                    print()
                    in_text = False
                reasoning_index = index
                sys.stdout.write(DIM)
                for line in delta.splitlines(keepends=True):
                    if reasoning_line_start:
                        sys.stdout.write("> ")
                    sys.stdout.write(line)
                    reasoning_line_start = line.endswith("\n")
                sys.stdout.write(RESET)
                sys.stdout.flush()

            case TextDelta(delta=delta):
                if not in_text:
                    in_text = True
                if "[Output truncated:" in delta:
                    sys.stdout.write(f"\n{RED}{delta.strip()}{RESET}\n")
                    in_text = False
                else:
                    sys.stdout.write(delta)
                sys.stdout.flush()

            case Refusal(text=text, category=category, blocked_input=blocked):
                # Rendered, and not as ordinary text: unrendered, a declined turn looked to the
                # user like the model answering with nothing. The banner goes once per turn: a
                # refusal arrives in fragments, and one banner each read as several refusals.
                if in_text:
                    print()
                    in_text = False
                if not declined:
                    declined = True
                    what = "prompt blocked" if blocked else "declined"
                    tail = f" ({category})" if category else ""
                    sys.stdout.write(f"{RED}[{what}{tail}]{RESET} ")
                sys.stdout.write(text)
                sys.stdout.flush()

            case Citation(cited_text=cited, title=title, url=url):
                if in_text:
                    print()
                    in_text = False
                print(f"{DIM}[cited: {url or title or cited or 'source'}]{RESET}")

            case ImageOutput(data=data, media_type=mt):
                if in_text:
                    print()
                    in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[image saved: {path}]{RESET}")

            case AudioOutput(data=data, media_type=mt):
                if in_text:
                    print()
                    in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[audio saved: {path}]{RESET}")

            case VideoOutput(data=data, media_type=mt):
                if in_text:
                    print()
                    in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[video saved: {path}]{RESET}")

            case ToolUseStart(index=index, tool_use_id=tid, name=name):
                if in_text:
                    print()
                    in_text = False
                sys.stdout.write(f"\n{BOLD}{CYAN}\u25b6 {name}{RESET}")
                sys.stdout.flush()
                arg_streams[tid] = ToolArgStream(tid, index)

            case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                stream = arg_streams.get(tid)
                if stream:
                    for fe in stream.feed(pj):
                        _render_field_event(fe)
                    if stream.done:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        del arg_streams[tid]

            case ToolOutputDelta(tool_use_id=tid, key=key, delta=delta):
                if tid not in streamed_tool_ids:
                    sys.stdout.write("\n")
                streamed_tool_ids.add(tid)
                color = RED if key == "stderr" else DIM
                sys.stdout.write(f"{color}{delta}{RESET}")
                sys.stdout.flush()

            case ToolResult(tool_use_id=tid, is_error=is_error, content=content):
                if is_error:
                    sys.stdout.write(f"{RESET}\n{RED}{content}{RESET}\n")
                elif tid in streamed_tool_ids:
                    sys.stdout.write(f"{RESET}\n")
                else:
                    sys.stdout.write(f"{RESET}\n{GREEN}{content}{RESET}\n")
                sys.stdout.flush()

            case IterationEnd():
                # The banner goes once per turn, and a prompt that runs tools has several. Set
                # once for the whole prompt, a refusal in a later turn printed its text with
                # nothing marking it as a refusal.
                declined = False

            case Error(exception=exc):
                message = str(getattr(agent.transport, "format_error", str)(exc))
                print(f"\n{RED}Error: {message}{RESET}", file=sys.stderr)

            case SessionEndEvent(stop_reason=reason, total_usage=usage):
                if in_text:
                    print()
                if reason in INCOMPLETE:
                    # Nothing else says so. The answer stops mid-sentence and reads exactly like
                    # one the model finished.
                    print(f"{RED}[incomplete: {reason}]{RESET}")
                print(f"{DIM}[{usage.input_tokens}in/{usage.output_tokens}out tokens]{RESET}")


_media_counter = 0


def _save_media(data: bytes, media_type: str) -> str:
    """Save media bytes to a temp file, return the path."""
    import tempfile

    global _media_counter
    _media_counter += 1
    ext = media_type.split("/")[-1].split(";")[0]
    fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix=f"axio_{_media_counter:03d}_")
    os.write(fd, data)
    os.close(fd)
    return path


_field_first_delta = True


def _render_field_event(event: ToolFieldStart | ToolFieldDelta | ToolFieldEnd) -> None:
    global _field_first_delta
    match event:
        case ToolFieldStart(key=key):
            sys.stdout.write(f"\n  {YELLOW}{key}{RESET}: {DIM}")
            sys.stdout.flush()
            _field_first_delta = True
        case ToolFieldDelta(text=text):
            if _field_first_delta and "\n" in text:
                sys.stdout.write("\n")
            _field_first_delta = False
            sys.stdout.write(text)
            sys.stdout.flush()
        case ToolFieldEnd():
            sys.stdout.write(RESET)
            sys.stdout.flush()


# ── Input handling ───────────────────────────────────────────────────


def _read_input() -> str:
    """Read user input, collecting extra lines from a multiline paste."""
    import select

    first = input("repl> ")
    lines = [first]
    fd = sys.stdin.fileno()
    while select.select([fd], [], [], 0.05)[0]:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        extra = chunk.decode(errors="replace").splitlines()
        for line in extra:
            print(f"  ... {line}")
        lines.extend(extra)
    return "\n".join(lines).strip()


# ── REPL commands ────────────────────────────────────────────────────


class Command(NamedTuple):
    """A REPL command with separate show (no arg) and apply (with arg) modes."""

    show: Callable[[], None]
    apply: Callable[[str], None]


# CLI arg attr → slash command name (for unified init).
_CLI_TO_SLASH: dict[str, str] = {
    "thinking": "/thinking",
    "temperature": "/temperature",
    "max_tokens": "/max-tokens",
    "debug": "/debug",
}


def _apply_cli_args(args: object, commands: dict[str, Command]) -> None:
    """Apply CLI arguments through the same command handlers as slash commands."""
    for attr, cmd_name in _CLI_TO_SLASH.items():
        val: Any = getattr(args, attr, None)
        if val is None or val is False:
            continue
        arg = "on" if isinstance(val, bool) else val if isinstance(val, str) else str(val)
        commands[cmd_name].apply(arg)


# ── model ──


def _model_listing(models: ModelRegistry) -> str:
    return "Available models:\n" + ("\n".join(f"  {name}" for name in models.keys()) or "  (none)")


def _select_model(transport: Any, model: ModelSpec, entry: TransportEntry | None = None) -> Any:
    previous = getattr(transport, "reasoning_effort", None)
    selected = entry.select_model(transport, model) if entry is not None else select_transport_model(transport, model)
    if previous is not None and getattr(selected, "reasoning_effort", None) is None:
        print(f"Reasoning effort reset to default: {model.id} does not support {previous!r}.")
    return selected


def _warn_missing_tools(model: ModelSpec, tools: list[Tool[Any]]) -> None:
    if tools and Capability.tool_use not in model.capabilities:
        print(
            f"Tools are disabled for {model.id}: Axio has no declared tool support for this model. "
            "Select another model with /model.",
            file=sys.stderr,
        )


def _show_model(transport: Any) -> None:
    model = transport.model
    caps = ", ".join(sorted(c.value for c in model.capabilities))
    print(f"Current model: {BOLD}{model.id}{RESET}")
    print(f"Capabilities: {caps}")
    print(f"Available: {', '.join(transport.models.keys())}")


@dataclass(slots=True)
class REPLState:
    """Own the inputs used to synchronize the agent's configuration."""

    cwd: Path
    transport: Any
    providers: list[ToolProvider] = field(default_factory=list)
    tools: list[Tool[Any]] = field(default_factory=list)
    entry: TransportEntry | None = None
    agent: Agent = field(init=False)

    def __post_init__(self) -> None:
        self.agent = Agent(system="", transport=self.transport, tools=self.tools, providers=self.providers)
        for provider in self.providers:
            bind = getattr(provider, "bind", None)
            if callable(bind):
                bind(self.agent)
        self.sync_agent()

    def sync_agent(self) -> None:
        model = self.transport.model
        if model is None:
            raise CredentialError("Select a visible model before starting the agent.")
        if self.entry is not None:
            self.entry.bind_tools(self.transport)
        self.agent.transport = self.transport
        self.agent.system = build_system_prompt(self.cwd, model, self.agent.get_tools(), "")

    def change_model(self, arg: str) -> None:
        matches = self.transport.models.search(arg)
        if len(matches) == 1:
            self.transport = _select_model(self.transport, next(iter(matches.values())), self.entry)
        self.sync_agent()
        if len(matches) == 1:
            print(f"Switched to {BOLD}{self.transport.model.id}{RESET}")
            _warn_missing_tools(self.transport.model, self.agent.get_tools())
        elif len(matches) == 0:
            print(f"No model matching {arg!r}.\n{_model_listing(self.transport.models)}", file=sys.stderr)
        else:
            print(f"Ambiguous — matches: {', '.join(matches.keys())}")


# ── thinking ──


def _show_thinking(transport: Any) -> None:
    describe = getattr(transport, "describe_thinking", None)
    if callable(describe):
        print(describe())
        return
    if callable(getattr(transport, "supported_thinking_levels", None)):
        print(f"Reasoning effort: {transport.reasoning_effort or 'default'}")
        print(f"Valid levels: {', '.join(('default', *transport.supported_thinking_levels(transport.model)))}")
        return
    level = getattr(transport, "thinking_level", None)
    budget = getattr(transport, "thinking_budget", None)
    get_opts = getattr(transport, "get_thinking_options", None)
    valid_levels = get_opts() if get_opts else None
    if level:
        print(f"Thinking level: {BOLD}{level}{RESET}")
    elif budget is not None:
        print(f"Thinking budget: {BOLD}{budget}{RESET} tokens")
    else:
        print("Thinking: default")
    if valid_levels is not None:
        print(f"Valid levels: {', '.join(valid_levels)}")
    elif get_opts is not None:
        print("Usage: /thinking <budget_tokens>")


def _apply_thinking(transport: Any, arg: str) -> None:
    configure = getattr(transport, "configure_thinking", None)
    if callable(configure):
        try:
            print(configure(arg))
        except ValueError as error:
            print(str(error), file=sys.stderr)
        return
    if not getattr(transport, "supports_thinking", True):
        print("This transport does not support thinking settings.")
        return
    if callable(getattr(transport, "supported_thinking_levels", None)):
        if arg == "default":
            transport.reasoning_effort = None
            print("Reasoning effort: default")
            return
        levels = transport.supported_thinking_levels(transport.model)
        if arg not in levels:
            print(
                f"Unsupported reasoning effort {arg!r}. Valid levels: {', '.join(('default', *levels))}.",
                file=sys.stderr,
            )
            return
        transport.reasoning_effort = arg
        print(f"Reasoning effort: {arg}")
        return
    get_opts = getattr(transport, "get_thinking_options", None)
    valid_levels = get_opts() if get_opts else None
    if arg.isdigit():
        if valid_levels is not None:
            model_id = getattr(getattr(transport, "model", None), "id", "?")
            print(f"{model_id} uses thinking levels, not token budgets.")
            print(f"Valid levels: {', '.join(valid_levels)}")
            return
        transport.thinking_budget = int(arg)
        transport.thinking_level = None
        print(f"Thinking budget: {BOLD}{arg}{RESET} tokens")
    else:
        name = arg.upper()
        if valid_levels is not None and name not in valid_levels:
            print(f"{name} is not valid. Valid levels: {', '.join(valid_levels)}")
            return
        transport.thinking_level = name
        transport.thinking_budget = None
        print(f"Thinking level: {BOLD}{name}{RESET}")


# ── temperature ──


def _show_temperature(transport: Any) -> None:
    temp = getattr(transport, "temperature", None)
    print(f"Temperature: {BOLD}{temp if temp is not None else 'default'}{RESET}")


def _apply_temperature(transport: Any, arg: str) -> None:
    if not getattr(transport, "supports_temperature", True):
        print("This transport does not support temperature.")
        return
    try:
        val = float(arg)
    except ValueError:
        print(f"Invalid temperature: {arg!r}")
        return
    if hasattr(transport, "temperature"):
        transport.temperature = val
        print(f"Temperature: {BOLD}{val}{RESET}")
    else:
        print("Transport does not support temperature")


# ── iterations ──


def _show_iterations(agent: Agent) -> None:
    print(f"Max iterations: {BOLD}{agent.max_iterations}{RESET}")


def _apply_iterations(agent: Agent, arg: str) -> None:
    try:
        val = int(arg)
    except ValueError:
        print(f"Invalid value: {arg!r}")
        return
    agent.max_iterations = val
    print(f"Max iterations: {BOLD}{val}{RESET}")


# ── max-tokens ──


def _show_max_tokens(transport: Any) -> None:
    cur = getattr(transport, "max_output_tokens", None)
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if cur:
        print(f"Max output tokens: {BOLD}{cur}{RESET} (model default: {model_default})")
    else:
        print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")


def _apply_max_tokens(transport: Any, arg: str) -> None:
    if not getattr(transport, "supports_max_tokens", True):
        print("This transport does not support max output tokens.")
        return
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if arg == "default":
        transport.max_output_tokens = None
        print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")
        return
    try:
        val = int(arg)
    except ValueError:
        print(f"Invalid value: {arg!r}")
        return
    transport.max_output_tokens = val
    print(f"Max output tokens: {BOLD}{val}{RESET}")


# ── debug ──


def _show_debug(transport: Any) -> None:
    cur = getattr(transport, "debug", False)
    print(f"Debug: {BOLD}{'on' if cur else 'off'}{RESET}")


def _apply_debug(transport: Any, arg: str) -> None:
    if not getattr(transport, "supports_debug", True):
        print("Request body debugging is not supported for this transport.")
        return
    val = arg.lower()
    if val == "on":
        transport.debug = True
        print(f"Debug: {BOLD}on{RESET} (request/response bodies logged to stderr)")
    elif val == "off":
        transport.debug = False
        print(f"Debug: {BOLD}off{RESET}")
    else:
        print("Usage: /debug on|off")


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    if sys.argv[1:3] == ["mcp", "import"]:
        if importlib.util.find_spec("axio_tools_mcp") is None:
            raise ValueError("MCP is unavailable. Install axio-repl[mcp].")
        from .mcp_import import import_main

        import_main(sys.argv[3:])
        return

    parser = argparse.ArgumentParser(
        description="REPL coding assistant (axio)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Single prompt (non-interactive)")
    catalog = discover_transports()
    parser.add_argument(
        "--transport",
        choices=sorted(catalog.entries.keys() | catalog.errors.keys()),
        default=None,
        help="Transport name (auto-detected if omitted)",
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--account", help="OAuth account ID; defaults to the saved account selection")
    parser.add_argument("--credentials", type=Path, help="Override the transport's OAuth credential file")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--thinking", default=None, help="Thinking level or token budget (integer)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum agent iterations")
    parser.add_argument("--debug", action="store_true", help="Log request/response bodies to stderr")
    parser.add_argument("--project-root", type=Path, help="Project root containing the working directory")
    parser.add_argument("--workspace-root", type=Path, action="append", default=[], help="Additional independent root")
    parser.add_argument(
        "--isolated", action="store_true", help="Disable automatic instructions, MCP, and local skills"
    )
    parser.add_argument(
        "--skills-dir", type=Path, action="append", default=[], help="Additional local skills directory"
    )
    parser.add_argument("--skills-list", action="store_true", help="List local skills without starting a model or MCP")
    parser.add_argument("--mcp-config", type=Path, action="append", default=[], help="Additional MCP configuration")
    parser.add_argument(
        "--config-explain", "--why", action="store_true", help="Explain configuration and check MCP without a model"
    )
    parser.add_argument(
        "--instructions-file", type=Path, action="append", default=[], help="Explicit session instructions"
    )
    parser.add_argument(
        "--instructions-max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Total instruction byte limit"
    )
    parser.add_argument(
        "--instructions-explain", type=Path, metavar="PATH", help="Show applicable files without a model"
    )
    args = parser.parse_args()

    local_available = importlib.util.find_spec("axio_tools_local") is not None
    mcp_available = importlib.util.find_spec("axio_tools_mcp") is not None
    local_paths = None
    instructions = None
    configuration: dict[str, Any] = {"application": "repl"}
    try:
        workspace = ConfigPaths.discover(Path.cwd(), args.project_root, args.workspace_root)
        if local_available:
            from axio_tools_local.instructions import load_instructions
            from axio_tools_local.paths import LocalPaths
            from axio_tools_local.provider import LocalOptions
            from axio_tools_local.skills import LocalSkills

            local_paths = LocalPaths.discover(workspace.cwd, workspace.project_root, workspace.roots[1:])
            instructions = load_instructions(
                local_paths,
                args.instructions_explain,
                isolated=args.isolated,
                files=args.instructions_file,
                max_bytes=args.instructions_max_bytes,
            )
            configuration["local"] = LocalOptions(
                workspace.cwd,
                workspace.roots,
                args.isolated,
                tuple(args.instructions_file),
                tuple(args.skills_dir),
                args.instructions_max_bytes,
            )
        elif args.instructions_file or args.skills_dir or args.instructions_explain or args.skills_list:
            parser.error("Local instructions and skills are unavailable. Install axio-repl[local].")
    except ValueError as error:
        parser.error(str(error))
    if args.instructions_explain is not None:
        assert instructions is not None and local_paths is not None
        print(instructions.explain(local_paths, args.instructions_explain))
        return
    if args.skills_list:
        local_skills = LocalSkills(workspace.cwd, directories=args.skills_dir, isolated=args.isolated)
        await local_skills.reload()
        print(describe_local_skills(local_skills))
        return
    mcp_projects: tuple[MCPConfiguration, ...] = ()
    if mcp_available:
        from axio_tools_mcp.provider import MCPToolProvider

        from .configuration import connection_configs, load_mcp_config

        mcp_projects = load_mcp_config(workspace, files=args.mcp_config, isolated=args.isolated)
        configuration["mcp"] = connection_configs(mcp_projects)
    elif args.mcp_config:
        parser.error("MCP is unavailable. Install axio-repl[mcp].")
    if args.config_explain:
        print(explain_config(workspace, mcp_projects, instructions, local_paths))
        if mcp_available:
            diagnostic_mcp = MCPToolProvider()
            try:
                await diagnostic_mcp.reload(
                    connection_configs(mcp_projects), reserved={"load_instruction", "read_local_skill"}
                )
                print("\nMCP connection check:")
                print(describe_mcp(diagnostic_mcp.runtime, mcp_projects))
            finally:
                await diagnostic_mcp.close()
        else:
            print("MCP is unavailable. Install axio-repl[mcp].")
        return

    if args.transport is None and (args.account or args.credentials):
        parser.error("--account and --credentials require an explicit --transport")
    try:
        entry = await _select_transport(args.transport)
    except TransportError as error:
        print(str(error), file=sys.stderr)
        if args.transport is not None:
            raise SystemExit(1) from None
        reason = str(error)
        entry = TransportEntry.from_target("unconfigured", lambda **kwargs: UnconfiguredTransport(reason))
    if entry.auth is None and (args.account or args.credentials):
        parser.error(f"Transport {entry.name!r} does not support account settings")
    for option in ("temperature", "max_tokens", "debug", "thinking"):
        value = getattr(args, option)
        if value is not None and value is not False and not getattr(entry, f"supports_{option}"):
            parser.error(f"Transport {entry.name!r} does not support --{option.replace('_', '-')}")
    setup_history()
    if entry.auth is not None:
        try:
            await entry.auth.prompt_login_if_needed(args.credentials, args.account)
        except Exception as error:
            raise TransportError(entry.format_error(error)) from error

    async with aiohttp.ClientSession() as session, AsyncExitStack() as stack:
        try:
            transport = await stack.enter_async_context(
                entry.open(session=session, account=args.account, credentials=args.credentials)
            )
        except Exception as error:
            raise TransportError(entry.format_error(error)) from error
        if args.model is not None:
            if args.model not in transport.models:
                diagnostic = getattr(transport, "model_selection_error", None)
                detail = diagnostic(args.model) if callable(diagnostic) else f"Unknown model {args.model!r}."
                parser.error(f"{detail}\n{_model_listing(transport.models)}")
            transport = _select_model(transport, transport.models[args.model], entry)

        levels_for = getattr(transport, "supported_thinking_levels", None)
        if callable(levels_for):
            levels = levels_for(transport.model)
            default = getattr(transport, "default_reasoning_effort", None)
            if args.thinking is None and default in levels:
                transport.reasoning_effort = default
            elif args.thinking is not None and args.thinking != "default" and args.thinking not in levels:
                parser.error(
                    f"Unsupported reasoning effort {args.thinking!r}. Valid levels: {', '.join(('default', *levels))}."
                )

        providers = discover_tool_providers()
        for name, issue in providers.errors.items():
            print(f"Provider {name!r} unavailable: {issue}", file=sys.stderr)
        stack.callback(entry.bind_tools, None)
        for name, provider in list(providers.entries.items()):
            provider_stack = await stack.enter_async_context(AsyncExitStack())
            provider_stack.push_async_callback(provider.close)
            try:
                await provider.init(config=configuration)
            except Exception as error:
                print(f"Provider {name!r} failed: {error}", file=sys.stderr)
                await provider_stack.aclose()
                providers.errors[name] = str(error)
                del providers.entries[name]
        plain = discover_tools()
        for name, issue in plain.errors.items():
            print(f"Tool {name!r} unavailable: {issue}", file=sys.stderr)
        state = REPLState(
            workspace.cwd, transport, list(providers.entries.values()), standalone_tools(plain, providers), entry
        )
        agent = state.agent
        agent.max_iterations = args.max_iterations

        def setting(name: str, show: Callable[[Any], None], apply: Callable[[Any, str], None]) -> Command:
            def change(value: str) -> None:
                if not getattr(entry, f"supports_{name}"):
                    print(f"Transport {entry.name!r} does not support {name.replace('_', ' ')}.")
                    return
                apply(state.transport, value)

            return Command(lambda: show(state.transport), change)

        commands: dict[str, Command] = {
            "/thinking": setting("thinking", _show_thinking, _apply_thinking),
            "/temperature": setting("temperature", _show_temperature, _apply_temperature),
            "/max-tokens": setting("max_tokens", _show_max_tokens, _apply_max_tokens),
            "/debug": setting("debug", _show_debug, _apply_debug),
        }
        _apply_cli_args(args, commands)

        state.sync_agent()
        _warn_missing_tools(state.transport.model, agent.get_tools())
        ctx = MemoryContextStore()

        def show_mcp() -> None:
            if not mcp_available:
                print("MCP is unavailable. Install axio-repl[mcp].")
                return
            from axio_tools_mcp.provider import MCPToolProvider

            active = [provider for provider in state.providers if isinstance(provider, MCPToolProvider)]
            for provider in active:
                print(describe_mcp(provider.runtime, mcp_projects))
            if not active:
                print("No MCP provider is active.")

        commands["/mcp"] = Command(show_mcp, lambda _: print("Usage: /mcp [reload]"))

        def show_skills() -> None:
            for provider in state.providers:
                for instruction in provider.get_instructions():
                    print(f"{instruction.id}: {instruction.summary}")
            if not any(provider.get_instructions() for provider in state.providers):
                print("No instructions or skills available.")

        commands["/skills"] = Command(show_skills, lambda _: print("Usage: /skills [reload]"))

        async def refresh_providers() -> None:
            for name, provider in providers.entries.items():
                if isinstance(provider, RefreshableToolProvider):
                    try:
                        await provider.refresh(config=configuration)
                    except Exception as error:
                        print(f"Provider {name!r} reload failed: {error}", file=sys.stderr)

        commands["/model"] = Command(lambda: _show_model(state.transport), state.change_model)
        commands["/iterations"] = Command(
            lambda: _show_iterations(agent),
            lambda a: _apply_iterations(agent, a),
        )

        async def account_command(user_input: str) -> None:
            auth = entry.auth
            assert auth is not None
            lowered = user_input.lower()
            if lowered == "/account":
                print(await auth.describe_accounts(args.credentials, state.transport))
                return
            if lowered == "/logout":
                await auth.logout(state.transport)
                await ctx.clear()
                print("Local credentials cleared. Use /login to sign in again.")
                return
            selected_id = auth.account_id(state.transport)
            if lowered == "/login":
                selected_id = await auth.sign_in(args.credentials, selected_id)
            else:
                selected_id = user_input.split(maxsplit=1)[1].strip()
            selected_transport = await stack.enter_async_context(
                entry.open(
                    session=session,
                    credentials=args.credentials,
                    account=selected_id,
                )
            )
            await auth.select_account(args.credentials, selected_id)
            state.transport = selected_transport
            state.sync_agent()
            await ctx.clear()
            print(f"{entry.display_name} account ready. Model: {selected_transport.model.id}. History cleared.")

        loop = asyncio.get_event_loop()
        prompt_task: asyncio.Task[None] | None = None

        def _on_sigint() -> None:
            nonlocal prompt_task
            if prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        try:
            if args.prompt:
                if isinstance(state.transport, UnconfiguredTransport):
                    raise TransportError(state.transport.reason)
                prompt_task = asyncio.create_task(run_prompt(agent, ctx, args.prompt))
                try:
                    await prompt_task
                except asyncio.CancelledError:
                    print(f"\n{DIM}[interrupted]{RESET}")
                finally:
                    prompt_task = None
                return

            commands_list = ", ".join(["/help", *commands, "/quit"])
            if entry.auth is not None:
                commands_list += ", /account [ID], /login, /logout"
            label = getattr(state.transport, "name", entry.display_name)
            print(f"REPL ready ({label}). Commands: {commands_list}")
            if mcp_available:
                show_mcp()

            while True:
                try:
                    user_input = await loop.run_in_executor(None, _read_input)
                except EOFError:
                    print()
                    break

                if not user_input:
                    continue
                lowered = user_input.lower()
                if lowered in {"/quit", "/exit", "/q"}:
                    break
                if entry.auth is not None and (
                    lowered in {"/account", "/login", "/logout"} or lowered.startswith("/account ")
                ):
                    prompt_task = asyncio.create_task(account_command(user_input))
                    try:
                        await prompt_task
                    except asyncio.CancelledError:
                        print(f"\n{DIM}[Account command interrupted]{RESET}")
                    except Exception as error:
                        print(entry.format_error(error), file=sys.stderr)
                    finally:
                        prompt_task = None
                    continue
                if lowered == "/help":
                    tool_list = (
                        ", ".join(t.name for t in agent.get_tools())
                        if Capability.tool_use in state.transport.model.capabilities
                        else "none (tool support is not declared for this model)"
                    )
                    print(f"Type your request. Tools: {tool_list}")
                    print(f"Commands: {commands_list}")
                    continue
                if lowered in {"/skills reload", "/mcp reload"}:
                    try:
                        if lowered == "/mcp reload" and mcp_available:
                            mcp_projects = load_mcp_config(workspace, files=args.mcp_config, isolated=args.isolated)
                            configuration["mcp"] = connection_configs(mcp_projects)
                        prompt_task = asyncio.create_task(refresh_providers())
                        await prompt_task
                    except asyncio.CancelledError:
                        print(f"\n{DIM}[Provider reload interrupted]{RESET}")
                    except ValueError as error:
                        print(str(error), file=sys.stderr)
                    finally:
                        prompt_task = None
                        state.sync_agent()
                    if lowered == "/skills reload":
                        show_skills()
                    else:
                        show_mcp()
                    continue

                matched = False
                for prefix, cmd in commands.items():
                    if lowered == prefix or lowered.startswith(prefix + " "):
                        arg = user_input[len(prefix) :].strip() or None
                        if arg is None:
                            cmd.show()
                        else:
                            cmd.apply(arg)
                        matched = True
                        break
                if matched:
                    continue

                state.sync_agent()
                prompt_task = asyncio.create_task(run_prompt(agent, ctx, user_input))
                try:
                    await prompt_task
                except asyncio.CancelledError:
                    print(f"\n{DIM}[interrupted]{RESET}")
                finally:
                    prompt_task = None
        finally:
            loop.remove_signal_handler(signal.SIGINT)


def main_sync() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except (TransportError, CredentialError, StreamError, aiohttp.ClientError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main_sync()
