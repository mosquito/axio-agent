"""Connect caller-supplied MCP configurations and expose tools and instructions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from axio.instructions import Instruction
from axio.tool import Tool

from .config import MCPServerConfig
from .inspection import ServerInspection
from .instructions import session_instructions
from .runtime import MCPRuntime, Server


class MCPToolProvider:
    """Own connections without discovering local files or project roots."""

    def __init__(self) -> None:
        self.runtime = MCPRuntime()
        self.configurations: tuple[MCPServerConfig, ...] = ()
        self._instructions: dict[int, tuple[Server, tuple[Instruction, ...]]] = {}

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        self.runtime.on_status = callback

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        configurations = config.get("mcp", ()) if isinstance(config, Mapping) else config or ()
        if not isinstance(configurations, (tuple, list)) or not all(
            isinstance(item, MCPServerConfig) for item in configurations
        ):
            raise TypeError("MCP provider requires a sequence of parsed MCPServerConfig objects")
        await self.reload(tuple(configurations))

    async def refresh(self, *, config: Any = None) -> None:
        await self.init(config=config)

    async def reload(self, configurations: tuple[MCPServerConfig, ...], *, reserved: set[str] | None = None) -> None:
        await self.runtime.reload(configurations, reserved=reserved or set())
        self.configurations = configurations
        self._instructions.clear()

    @property
    def all_tools(self) -> list[Tool[Any]]:
        return self.runtime.tools

    def get_instructions(self) -> Sequence[Instruction]:
        active: dict[int, tuple[Server, tuple[Instruction, ...]]] = {}
        for server in self.runtime.servers:
            if server.state != "connected" or server.session is None:
                continue
            entry = self._instructions.get(id(server))
            if entry is None or entry[0] is not server:
                entry = (server, session_instructions(server.session, server.tools, server.skills, server.resources))
            active[id(server)] = entry
        self._instructions = active
        return tuple(instruction for _, instructions in active.values() for instruction in instructions)

    def inspect_servers(self) -> list[ServerInspection]:
        return self.runtime.inspect_servers()

    async def close(self) -> None:
        self._instructions.clear()
        await self.runtime.close()
