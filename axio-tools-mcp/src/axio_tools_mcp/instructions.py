"""Provider-owned instructions backed by a live MCP session."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from axio.instructions import Instruction, LoadedInstruction
from axio.tool import Tool

from .resources import ResourceCatalog
from .session import MCPSession
from .skills import SkillCatalog


class MCPInstruction(Instruction):
    def __init__(
        self,
        id: str,
        summary: str,
        *,
        source: str,
        session: MCPSession,
        read: Callable[[], Awaitable[str]],
        tools: Sequence[Tool[Any]],
    ) -> None:
        super().__init__(id, summary, source=source)
        self.session = session
        self.read = read
        self.tools = tuple(tools)

    async def load(self) -> LoadedInstruction:
        if not self.session.is_connected:
            raise RuntimeError("MCP instruction belongs to a closed connection; reload the provider")
        text = await self.read()
        if not self.session.is_connected:
            raise RuntimeError("MCP connection closed while loading the instruction")
        return LoadedInstruction(text, self.tools)


def session_instructions(
    session: MCPSession,
    tools: Sequence[Tool[Any]],
    skills: SkillCatalog | None = None,
    resources: ResourceCatalog | None = None,
) -> tuple[Instruction, ...]:
    """Build metadata without reading full skill content or executing tools."""
    name = session.config.name
    instructions: list[Instruction] = []
    if session.server_instructions:

        async def read_server() -> str:
            return session.server_instructions

        instructions.append(
            MCPInstruction(
                f"mcp:{name}:server",
                f"Usage instructions from MCP server {name}",
                source=name,
                session=session,
                read=read_server,
                tools=tools,
            )
        )
    if skills is not None:
        for skill in skills.skills.values():

            def reader(catalog: SkillCatalog, uri: str) -> Callable[[], Awaitable[str]]:
                async def read() -> str:
                    document = json.loads(await catalog.read(uri))
                    text: str = document.pop("text")
                    return (
                        "Paths in this skill refer to its MCP server. Use its resource reader for supporting files; "
                        "preserve every URI exactly.\n" + json.dumps(document, ensure_ascii=False) + "\n\n" + text
                    )

                return read

            instructions.append(
                MCPInstruction(
                    f"mcp:{name}:{skill.uri}",
                    f"{name}:{skill.name}: {skill.description}",
                    source=skill.uri,
                    session=session,
                    read=reader(skills, skill.uri),
                    tools=tools,
                )
            )
    if resources is not None:

        async def read_resources() -> str:
            assert resources is not None
            return (
                f"Read resources with {name}__read_resource(uri). Refresh the catalog with {name}__list_resources(). "
                "Keep URI schemes unchanged. Resources remain available independently of skills.\n"
                + json.dumps(resources.snapshot(), ensure_ascii=False)
            )

        instructions.append(
            MCPInstruction(
                f"mcp:{name}:resources",
                f"Resource and template catalog from MCP server {name}",
                source=name,
                session=session,
                read=read_resources,
                tools=tools,
            )
        )
    return tuple(instructions)
