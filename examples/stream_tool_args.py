"""Minimal CLI demonstrating incremental streaming of tool call arguments
with partial JSON decoding — field values appear as they stream in.

Run:
    uv run --extra examples python examples/stream_tool_args.py

Requires OPENAI_API_KEY in env.
"""

from __future__ import annotations

import asyncio
import os
import sys

import aiohttp
from partial_json_parser import loads as partial_json_loads

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
from axio.tool import Tool, ToolHandler
from axio_transport_openai import OPENAI_MODELS, OpenAITransport

# ── Tools with chunky arguments to make streaming visible ────────────


class EditFile(ToolHandler):
    """Replace old_string with new_string in a file."""

    file_path: str
    old_string: str
    new_string: str

    async def __call__(self) -> str:
        return f"Replaced content in {self.file_path}"


class WriteFile(ToolHandler):
    """Write content to a file."""

    file_path: str
    content: str

    async def __call__(self) -> str:
        return f"Wrote {len(self.content)} chars to {self.file_path}"


TOOLS = [
    Tool(name="edit_file", description="Replace old_string with new_string in a file.", handler=EditFile),
    Tool(name="write_file", description="Write content to a file.", handler=WriteFile),
]

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"

# ── Partial JSON tracker ────────────────────────────────────────────


class ToolArgTracker:
    """Tracks partial JSON for one tool call, diffs decoded fields on each chunk."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.raw = ""
        self.prev_parsed: dict[str, str] = {}
        self.prev_value_lens: dict[str, int] = {}
        self.seen_newline: set[str] = set()

    def feed(self, partial_json: str) -> None:
        self.raw += partial_json
        try:
            parsed = partial_json_loads(self.raw)
        except Exception:
            return
        if not isinstance(parsed, dict):
            return

        for key, value in parsed.items():
            value_str = str(value)
            prev_len = self.prev_value_lens.get(key, 0)
            new_text = value_str[prev_len:]

            if key not in self.prev_parsed:
                sys.stdout.write(f"\n  {YELLOW}{key}{RESET}: {DIM}")
                if "\n" in new_text:
                    self.seen_newline.add(key)
                    # Reprint from newline: "label: \nfirst_line\n..."
                    sys.stdout.write("\n" + value_str)
                else:
                    sys.stdout.write(value_str)
            elif new_text:
                if key not in self.seen_newline and "\n" in new_text:
                    self.seen_newline.add(key)
                    # First newline arrived — reprint entire value from new line
                    sys.stdout.write(f"\r\033[2K  {YELLOW}{key}{RESET}:{DIM}\n{value_str}")
                else:
                    sys.stdout.write(new_text)

            self.prev_parsed[key] = value_str
            self.prev_value_lens[key] = len(value_str)

        sys.stdout.flush()


# ── Main ─────────────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "Write a small Python hello-world web server to hello.py using write_file, "
    "then use edit_file to change the greeting from 'Hello' to 'Howdy'."
)


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stream tool call arguments with partial JSON decoding")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="prompt to send to the model")
    parser.add_argument("--model", default="gpt-5.4", help="model name (default: gpt-5.4)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    model_name = args.model

    async with aiohttp.ClientSession() as session:
        transport = OpenAITransport(
            api_key=api_key,
            model=OPENAI_MODELS[model_name],
            session=session,
        )
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
                    sys.stdout.write(f"{RESET}\n  {color}→ {content}{RESET}\n")
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
