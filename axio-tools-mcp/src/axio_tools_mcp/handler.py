"""MCP tool handler factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from types import MappingProxyType
from typing import Any

from aiohttp_tiny_mcp import CallToolResult, EmbeddedResource, TextContent, TextResourceContents
from aiohttp_tiny_mcp.models import ToolDef
from axio.tool import Tool

from .session import MCPSession


def result_text(result: CallToolResult) -> str:
    """Join the text a tool returned. Binary and linked content carry no text."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, EmbeddedResource) and isinstance(block.resource, TextResourceContents):
            parts.append(block.resource.text)
    return "\n".join(parts)


def build_handler(
    tool_name: str,
    mcp_tool_name: str,
    description: str,
    session: MCPSession,
) -> Callable[..., Awaitable[str]]:
    """Return a plain async handler that forwards calls to the MCP session.

    The schema is provided separately via ``Tool(schema=MappingProxyType(input_schema))``;
    no annotation injection is needed here.
    """

    async def handler(**kwargs: Any) -> str:
        result = await session.call_tool(mcp_tool_name, kwargs)
        text = result_text(result)
        if result.is_error:
            raise RuntimeError(text or "MCP tool error")
        return text

    handler.__doc__ = description
    handler.__name__ = tool_name
    handler.__annotations__ = {"return": str}
    return handler


def build_tools(session: MCPSession, definitions: Iterable[ToolDef]) -> list[Tool[Any]]:
    """Wrap every tool a server offers as an Axio tool named ``<server>__<tool>``."""
    tools: list[Tool[Any]] = []
    for definition in definitions:
        tool_name = f"{session.config.name}__{definition.name}"
        description = definition.description or definition.name
        handler = build_handler(
            tool_name=tool_name,
            mcp_tool_name=definition.name,
            description=description,
            session=session,
        )
        tools.append(
            Tool(
                name=tool_name,
                description=description,
                handler=handler,
                schema=MappingProxyType(dict(definition.input_schema)),
            )
        )
    return tools
