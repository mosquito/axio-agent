"""Expose application tools only when initialized by the TUI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from axio.instructions import Instruction
from axio.plugins import tool_from_handler
from axio.tool import Tool

from . import tools


class TUIToolProvider:
    def __init__(self) -> None:
        self._tools: list[Tool[Any]] = []

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        if isinstance(config, Mapping) and config.get("application") == "tui":
            self._tools = [
                tool_from_handler(name, handler)
                for name, handler in (
                    ("status_line", tools.status_line),
                    ("confirm", tools.confirm),
                    ("subagent", tools.subagent),
                    ("vision", tools.vision_analyze),
                )
            ]

    @property
    def all_tools(self) -> list[Tool[Any]]:
        return list(self._tools)

    def get_instructions(self) -> tuple[Instruction, ...]:
        return ()

    async def close(self) -> None:
        self._tools.clear()
