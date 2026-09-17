"""Load MCP tools from server configurations."""

from __future__ import annotations

import logging
from typing import Any

from axio.tool import Tool

from .config import MCPServerConfig
from .handler import build_tools
from .session import MCPSession

logger = logging.getLogger(__name__)


async def load_mcp_tools(
    servers: list[MCPServerConfig],
) -> tuple[list[Tool[Any]], list[MCPSession]]:
    """Connect to MCP servers and discover their tools.

    Returns ``(tools, sessions)``. The caller is responsible for closing sessions.
    Failed servers are logged and skipped.
    """
    all_tools: list[Tool[Any]] = []
    sessions: list[MCPSession] = []

    for config in servers:
        session = MCPSession(config)
        try:
            await session.connect()
            mcp_tools = await session.list_tools()
        except Exception:
            logger.error("Failed to connect to MCP server %r", config.name, exc_info=True)
            try:
                await session.close()
            except Exception:
                pass
            continue

        sessions.append(session)
        for tool in build_tools(session, mcp_tools):
            all_tools.append(tool)
            logger.info("Loaded MCP tool %r from server %r", tool.name, config.name)

    return all_tools, sessions
