"""Dynamic async function creation from MCP tool schemas."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from axio.field import FieldInfo
from mcp.types import TextContent

from .session import MCPSession

_JSON_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_fields(
    input_schema: dict[str, Any],
) -> dict[str, tuple[Any, FieldInfo]]:
    """Convert JSON Schema properties to (py_type, FieldInfo) pairs."""
    properties: dict[str, Any] = input_schema.get("properties", {})
    required: set[str] = set(input_schema.get("required", []))
    fields: dict[str, tuple[Any, FieldInfo]] = {}

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        py_type: type[Any] = _JSON_TYPE_MAP.get(json_type, str)
        description = prop_schema.get("description", "")

        if prop_name in required:
            fields[prop_name] = (py_type, FieldInfo(description=description))
        else:
            default = prop_schema.get("default")
            fields[prop_name] = (
                py_type | None,
                FieldInfo(description=description, default=default),
            )

    return fields


def build_handler(
    tool_name: str,
    mcp_tool_name: str,
    description: str,
    input_schema: dict[str, Any],
    session: MCPSession,
) -> Callable[..., Awaitable[str]]:
    """Create a plain async function handler for an MCP tool.

    The returned function accepts ``**kwargs`` matching the tool's JSON schema
    and forwards the call to the MCP session.  Type annotations are set on
    ``__annotations__`` so that ``Tool.__post_init__`` can build the schema.
    """
    field_defs = _build_fields(input_schema)

    async def handler(**kwargs: Any) -> str:
        data = {k: v for k, v in kwargs.items() if v is not None}
        result = await session.call_tool(mcp_tool_name, data)
        if result.isError:
            parts = [c.text for c in result.content if isinstance(c, TextContent)]
            raise RuntimeError("\n".join(parts) or "MCP tool error")
        parts = [c.text for c in result.content if isinstance(c, TextContent)]
        return "\n".join(parts) or ""

    annotations: dict[str, Any] = {}
    for name, (py_type, fi) in field_defs.items():
        annotations[name] = Annotated[py_type, fi]
    annotations["return"] = str
    handler.__annotations__ = annotations
    handler.__doc__ = description
    handler.__name__ = tool_name

    return handler
