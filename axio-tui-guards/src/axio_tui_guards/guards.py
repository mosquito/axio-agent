"""Permission guards for the Axio TUI."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from axio.agent import Agent
from axio.blocks import ToolUseBlock
from axio.context import ContextStore
from axio.exceptions import GuardError
from axio.permission import PermissionGuard

if TYPE_CHECKING:
    from axio.tool import Tool


type PromptFn = Callable[[str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class Confirm:
    """A structured verdict produced by the guard's reviewer."""

    verdict: Literal["SAFE", "RISKY", "DENY"]
    reason: str
    category: str


class PathGuard(PermissionGuard):
    """Ask user about path access; allow grants access to the parent directory."""

    PATH_FIELDS = ("file_path", "filename", "directory", "path", "cwd")

    def __init__(self, prompt_fn: PromptFn | None = None) -> None:
        self.allowed: set[str] = set()
        self.denied: set[str] = set()
        self._prompt = prompt_fn or ask_user

    def _extract_path(self, kwargs: dict[str, Any]) -> tuple[str | None, str | None]:
        for name in self.PATH_FIELDS:
            if name in kwargs:
                return str(kwargs[name]), name
        return None, None

    @staticmethod
    def _parent_dir(path: str) -> str:
        p = Path(path)
        return str(p if p.suffix == "" else p.parent)

    def _is_allowed(self, path: str) -> bool:
        d = Path(self._parent_dir(path))
        return any(d == Path(a) or Path(a) in d.parents for a in self.allowed)

    def _is_denied(self, path: str) -> bool:
        return path in self.denied

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        path, field = self._extract_path(kwargs)
        if path is None or self._is_allowed(path):
            return kwargs
        if self._is_denied(path):
            raise GuardError(f"Path denied: {field}={path!r}")

        directory = self._parent_dir(path)
        msg = f"{field}={path!r} (directory: {directory})"
        answer = (await self._prompt(msg)).strip()
        if answer.lower() == "deny":
            self.denied.add(path)
            raise GuardError(f"Path denied: {field}={path!r}")
        if answer.lower() != "y":
            raise GuardError(f"Path access denied: {field}={path!r}")
        # Explicit legacy approval grants this directory for future calls
        self.allowed.add(directory)
        return kwargs


class LLMGuard(PermissionGuard):
    """Optional reviewer. The TUI uses review() as advice for its policy prompt."""

    def __init__(self, agent: Agent, context: ContextStore, prompt_fn: PromptFn | None = None) -> None:
        self.agent = agent
        self.context = context
        self._prompt = prompt_fn or ask_user

    async def extract_confirm(self, context: ContextStore, *, since: int = 0) -> Confirm:
        """Read a validated verdict from the current review only."""
        history = await context.get_history()
        for msg in reversed(history[since:]):
            if msg.role == "assistant":
                for block in reversed(msg.content):
                    if isinstance(block, ToolUseBlock) and block.name == "confirm":
                        try:
                            result = Confirm(**block.input)
                            if result.verdict not in {"SAFE", "RISKY", "DENY"}:
                                raise ValueError("Invalid verdict")
                            if not isinstance(result.reason, str) or not isinstance(result.category, str):
                                raise ValueError("Invalid review details")
                            return result
                        except Exception:
                            return Confirm(verdict="RISKY", reason="Unparseable", category="unknown")
        return Confirm(verdict="RISKY", reason="No verdict provided", category="unknown")

    async def assess(self, tool: Tool[Any], **kwargs: Any) -> Confirm:
        description = f"Tool: {tool.name}\nArguments: {json.dumps(kwargs, default=str)}"
        forked = await self.context.fork()
        since = len(await forked.get_history())
        try:
            await self.agent.run(description, forked)
        except Exception:
            return Confirm(verdict="RISKY", reason="Review failed", category="unknown")
        return await self.extract_confirm(forked, since=since)

    async def review(self, tool: Tool[Any], **kwargs: Any) -> str:
        result = await self.assess(tool, **kwargs)
        return f"{result.verdict}: {result.reason}"

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        """Standalone compatibility guard; approval never creates a category grant."""
        result = await self.assess(tool, **kwargs)
        if result.verdict == "SAFE":
            return kwargs
        if result.verdict == "DENY":
            raise GuardError(f"DENIED: {result.reason}")
        answer = (await self._prompt(f"{tool.name}({json.dumps(kwargs, default=str)})\n\n{result.reason}")).strip()
        if answer.lower() != "y":
            raise GuardError(f"User denied: {result.reason}")
        return kwargs


# Helper function for user input
ASK_LOCK = threading.Lock()


async def ask_user(message: str) -> str:
    def _blocking() -> str:
        with ASK_LOCK:
            return input(message)

    return await asyncio.to_thread(_blocking)
