"""MCPPlugin: ToolsPlugin implementation for the TUI plugin system."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from axio.instructions import Instruction
from axio.tool import Tool

from .inspection import ServerInspection
from .registry import MCPRegistry
from .skills import SkillCatalog


class MCPPlugin:
    """Dynamic tool provider that bridges MCP servers into Axio.

    Discovered via the ``axio.tools.settings`` entry point group.
    The TUI interacts with this class through the ToolsPlugin protocol.
    """

    def __init__(self) -> None:
        self._registry = MCPRegistry()
        self._config: Any = None
        self._global_config: Any = None

    @property
    def label(self) -> str:
        return "MCP Servers"

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        self._config = config
        self._global_config = global_config
        await self._registry.init(config=config, global_config=global_config)

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        self._registry.on_status = callback

    async def reconnect(self) -> None:
        await self._registry.reconnect()

    @property
    def all_tools(self) -> list[Tool[Any]]:
        return self._registry.all_tools

    def settings_screen(self) -> Any:
        from .tui import MCPHubScreen

        return MCPHubScreen(self._registry, config=self._config, global_config=self._global_config)

    def inspect_servers(self) -> list[ServerInspection]:
        return self._registry.inspect_servers()

    def skill_catalogs(self) -> dict[str, SkillCatalog]:
        return self._registry.skill_catalogs()

    def get_instructions(self) -> tuple[Instruction, ...]:
        return self._registry.get_instructions()

    def describe_skills(self) -> str:
        return self._registry.describe_skills()

    def describe(self) -> str:
        return self._registry.describe()

    def instructions(self) -> str:
        return self._registry.instructions()

    async def close(self) -> None:
        await self._registry.close()
