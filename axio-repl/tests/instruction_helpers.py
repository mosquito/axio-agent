from __future__ import annotations

from axio_tools_mcp.instructions import session_instructions
from axio_tools_mcp.runtime import MCPRuntime


async def instruction_text(runtime: MCPRuntime, kind: str) -> str:
    parts = []
    for server in runtime.servers:
        if server.state != "connected" or server.session is None:
            continue
        for instruction in session_instructions(server.session, server.tools, server.skills, server.resources):
            if kind == "resources" and instruction.id.endswith(":resources"):
                parts.append((await instruction.load()).text)
            elif kind == "skills" and not instruction.id.endswith((":server", ":resources")):
                parts.append(instruction.summary)
    return "\n".join(parts)
