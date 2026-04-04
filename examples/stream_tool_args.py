"""Minimal CLI demonstrating incremental streaming of tool call arguments
with partial JSON decoding — field values appear as they stream in.

Auto-detects transport from available API keys (OPENAI_API_KEY, NEBIUS_API_KEY,
OPENROUTER_API_KEY), or use --transport to pick explicitly.

Run:
    uv run --extra examples python examples/stream_tool_args.py "your prompt here"
"""

from __future__ import annotations

import asyncio
import os
from importlib.metadata import entry_points
import sys

import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import (
    Error,
    IterationEnd,
    SessionEndEvent,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.tool import Tool
from axio_tools_local.patch_file import PatchFile
from axio_tools_local.read_file import ReadFile
from axio_tools_local.write_file import WriteFile

TOOLS = [
    Tool(name="read_file", description=ReadFile.__doc__ or "", handler=ReadFile),
    Tool(name="write_file", description=WriteFile.__doc__ or "", handler=WriteFile),
    Tool(name="patch_file", description=PatchFile.__doc__ or "", handler=PatchFile),
]

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# ── Streaming JSON state machine ───────────────────────────────────


class ToolArgTracker:
    """Streams top-level tool argument values via a JSON state machine with stack.

    O(1) per character.  Top-level string values are decoded (escape sequences
    resolved, quotes stripped).  All other top-level values are emitted as raw JSON.
    """

    _ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}

    def __init__(self, name: str) -> None:
        self.name = name
        self._st = "INIT"
        self._stack: list[str] = []  # "o" (object) / "a" (array)
        self._key_buf: list[str] = []
        self._current_key = ""
        self._mode = ""  # "" | "raw" | "str"
        self._u_buf: list[str] = []
        self._high = 0  # pending high surrogate for \uXXXX pairs
        self._val_buf: list[str] = []  # buffer for multiline detection
        self._ml = False  # multiline resolved?

    def feed(self, chunk: str) -> None:
        for ch in chunk:
            self._step(ch)
        sys.stdout.flush()

    # -- output helpers ------------------------------------------------

    def _emit(self, ch: str) -> None:
        if self._mode == "str":
            self._str_out(ch)
        elif self._mode:
            sys.stdout.write(ch)

    def _str_out(self, text: str) -> None:
        """Output decoded text for a top-level string, detecting multiline."""
        if not self._ml:
            self._val_buf.append(text)
            if '\n' in text:
                self._ml = True
                sys.stdout.write(
                    f"\r\033[2K  {YELLOW}{self._current_key}{RESET}:{DIM}\n"
                )
                sys.stdout.write("".join(self._val_buf))
                self._val_buf.clear()
        else:
            sys.stdout.write(text)

    def _str_flush(self) -> None:
        """Flush buffered content for an inline string value."""
        if self._val_buf:
            sys.stdout.write("".join(self._val_buf))
            self._val_buf.clear()

    def _start_val(self, is_str: bool) -> None:
        if len(self._stack) == 1:
            self._mode = "str" if is_str else "raw"
            if is_str:
                self._val_buf.clear()
                self._ml = False

    def _end_val(self) -> None:
        if len(self._stack) == 1:
            self._mode = ""

    def _pop(self) -> None:
        self._stack.pop()
        self._end_val()
        self._st = "AFTER" if self._stack else "DONE"

    def _flush_high(self) -> None:
        if self._mode == "str":
            self._str_out(chr(self._high))
        elif self._mode == "raw":
            sys.stdout.write(f"\\u{self._high:04x}")
        self._high = 0

    def _emit_uchar(self, code: int) -> None:
        if self._mode == "str":
            self._str_out(chr(code))
        elif self._mode == "raw":
            sys.stdout.write("\\u" + "".join(self._u_buf))

    # -- value / delimiter dispatch ------------------------------------

    def _val_start(self, ch: str) -> None:
        if ch == '"':
            self._start_val(True)
            if self._mode == "raw":
                sys.stdout.write('"')
            self._st = "STR"
        elif ch == '{':
            self._start_val(False)
            self._emit('{')
            self._stack.append("o")
            self._st = "OBJ"
        elif ch == '[':
            self._start_val(False)
            self._emit('[')
            self._stack.append("a")
            self._st = "ARR"
        elif ch in '-0123456789':
            self._start_val(False)
            self._emit(ch)
            self._st = "NUM"
        elif ch in 'tfn':
            self._start_val(False)
            self._emit(ch)
            self._st = "LIT"

    def _after(self, ch: str) -> None:
        if ch == ',':
            self._emit(',')
            top = self._stack[-1] if self._stack else None
            self._st = "OBJ" if top == "o" else "VAL"
        elif ch == '}':
            self._emit('}')
            self._pop()
        elif ch == ']':
            self._emit(']')
            self._pop()
        else:
            self._emit(ch)
            self._st = "AFTER"

    # -- main state machine --------------------------------------------

    def _step(self, ch: str) -> None:
        st = self._st

        if st == "INIT":
            if ch == '{':
                self._stack.append("o")
                self._st = "OBJ"

        elif st == "OBJ":
            if ch == '"':
                self._emit('"')
                self._key_buf.clear()
                self._st = "KEY"
            elif ch == '}':
                self._emit('}')
                self._pop()
            else:
                self._emit(ch)

        elif st == "KEY":
            if ch == '\\':
                self._emit('\\')
                self._st = "KESC"
            elif ch == '"':
                if len(self._stack) == 1:
                    self._current_key = "".join(self._key_buf)
                self._emit('"')
                self._st = "COL"
            else:
                self._key_buf.append(ch)
                self._emit(ch)

        elif st == "KESC":
            self._key_buf.append(ch)
            self._emit(ch)
            self._st = "KEY"

        elif st == "COL":
            if ch == ':':
                if len(self._stack) == 1:
                    sys.stdout.write(
                        f"\n  {YELLOW}{self._current_key}{RESET}: {DIM}"
                    )
                else:
                    self._emit(':')
                self._st = "VAL"
            else:
                self._emit(ch)

        elif st == "VAL":
            if ch in ' \t\r\n':
                self._emit(ch)
            else:
                self._val_start(ch)

        elif st == "ARR":
            if ch in ' \t\r\n':
                self._emit(ch)
            elif ch == ']':
                self._emit(']')
                self._pop()
            else:
                self._val_start(ch)

        elif st == "STR":
            if ch == '\\':
                self._st = "SESC"
            elif ch == '"':
                if self._high:
                    self._flush_high()
                if self._mode == "str":
                    self._str_flush()
                elif self._mode == "raw":
                    sys.stdout.write('"')
                self._end_val()
                self._st = "AFTER"
            else:
                if self._high:
                    self._flush_high()
                self._emit(ch)

        elif st == "SESC":
            if self._high and ch != 'u':
                self._flush_high()
            if ch == 'u':
                self._u_buf.clear()
                self._st = "UESC"
            elif self._mode == "str":
                self._str_out(self._ESC.get(ch, ch))
                self._st = "STR"
            elif self._mode == "raw":
                sys.stdout.write('\\')
                sys.stdout.write(ch)
                self._st = "STR"
            else:
                self._st = "STR"

        elif st == "UESC":
            self._u_buf.append(ch)
            if len(self._u_buf) == 4:
                code = int("".join(self._u_buf), 16)
                if self._high:
                    if 0xDC00 <= code <= 0xDFFF:
                        full = (
                            0x10000
                            + (self._high - 0xD800) * 0x400
                            + (code - 0xDC00)
                        )
                        if self._mode == "str":
                            self._str_out(chr(full))
                        elif self._mode == "raw":
                            sys.stdout.write(
                                f"\\u{self._high:04x}\\u{code:04x}"
                            )
                    else:
                        self._flush_high()
                        self._emit_uchar(code)
                    self._high = 0
                elif 0xD800 <= code <= 0xDBFF:
                    self._high = code
                else:
                    self._emit_uchar(code)
                self._st = "STR"

        elif st == "NUM":
            if ch in '0123456789.eE+-':
                self._emit(ch)
            else:
                self._end_val()
                self._after(ch)

        elif st == "LIT":
            if ch.isalpha():
                self._emit(ch)
            else:
                self._end_val()
                self._after(ch)

        elif st == "AFTER":
            self._after(ch)


# ── Transport auto-detection ─────────────────────────────────────────


def _discover_transports() -> dict[str, type]:
    """Load transport classes from axio.transport entry points."""
    result = {}
    for ep in entry_points(group="axio.transport"):
        try:
            result[ep.name] = ep.load()
        except Exception:
            pass
    return result


def _select_transport(name: str | None) -> tuple[type, str]:
    """Return (transport_class, api_key) based on --transport or env auto-detection."""
    available = _discover_transports()
    if name:
        if name not in available:
            print(f"Unknown transport {name!r}. Available: {', '.join(sorted(available))}", file=sys.stderr)
            sys.exit(1)
        cls = available[name]
        meta = getattr(cls, "META", None)
        env_var = meta.api_key_env if meta else ""
        api_key = os.environ.get(env_var, "") if env_var else ""
        if not api_key:
            print(f"Set {env_var} for transport {name!r}", file=sys.stderr)
            sys.exit(1)
        return cls, api_key

    for tname, cls in available.items():
        meta = getattr(cls, "META", None)
        if meta and meta.api_key_env:
            api_key = os.environ.get(meta.api_key_env, "")
            if api_key:
                return cls, api_key

    print("No API key found. Set one of:", file=sys.stderr)
    for tname, cls in available.items():
        meta = getattr(cls, "META", None)
        if meta and meta.api_key_env:
            print(f"  {meta.api_key_env}  ({meta.label})", file=sys.stderr)
    sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stream tool call arguments with partial JSON decoding")
    parser.add_argument("prompt", help="prompt to send to the model")
    parser.add_argument("--transport", default=None, help="transport name (auto-detected from API keys if omitted)")
    parser.add_argument("--model", default=None, help="model name (uses transport default if omitted)")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    args = parser.parse_args()

    transport_cls, api_key = _select_transport(args.transport)

    async with aiohttp.ClientSession() as session:
        transport = transport_cls(api_key=api_key, session=session)

        if args.model:
            transport.model = transport.models[args.model]

        # Inject temperature into payload if requested
        if args.temperature is not None:
            _orig_build = transport.build_payload

            def build_payload_with_temp(messages, tools, system):
                payload = _orig_build(messages, tools, system)
                payload["temperature"] = args.temperature
                return payload

            transport.build_payload = build_payload_with_temp  # type: ignore[method-assign]

        agent = Agent(
            system="You are a coding assistant. Use the provided tools.",
            tools=TOOLS,
            transport=transport,
        )
        ctx = MemoryContextStore()

        print(f"{BOLD}👤 user:{RESET} {args.prompt}")

        trackers: dict[str, ToolArgTracker] = {}
        in_text = False

        async for event in agent.run_stream(args.prompt, ctx):
            match event:
                case TextDelta(delta=delta):
                    if not in_text:
                        sys.stdout.write(f"\n{BOLD}💬 model:{RESET} {DIM}")
                        in_text = True
                    sys.stdout.write(delta)
                    sys.stdout.flush()

                case ToolUseStart(tool_use_id=tid, name=name):
                    in_text = False
                    trackers[tid] = ToolArgTracker(name)
                    sys.stdout.write(f"{RESET}\n{BOLD}{CYAN}▶ {name}{RESET}")
                    sys.stdout.flush()

                case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                    if tid in trackers:
                        trackers[tid].feed(pj)

                case ToolResult(tool_use_id=tid, name=name, is_error=is_error, content=content):
                    color = RED if is_error else GREEN
                    sys.stdout.write(f"{RESET}\n{color}{content}{RESET}\n")
                    sys.stdout.flush()
                    trackers.pop(tid, None)

                case IterationEnd():
                    pass

                case Error(exception=exc):
                    print(f"\n{RED}Error: {exc}{RESET}", file=sys.stderr)

                case SessionEndEvent():
                    print()


if __name__ == "__main__":
    asyncio.run(main())
