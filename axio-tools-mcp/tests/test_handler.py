"""Tests for build_handler() dynamic function creation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from axio_tools_mcp.handler import build_handler
from axio_tools_mcp.session import MCPSession


def _make_mock_session() -> MCPSession:
    session = MagicMock(spec=MCPSession)
    session.call_tool = AsyncMock()
    return session


def test_schema_fidelity() -> None:
    """Built handler has correct fields from JSON schema."""
    session = _make_mock_session()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "count": {"type": "integer", "description": "Number of items"},
            "verbose": {"type": "boolean"},
        },
        "required": ["path"],
    }
    handler = build_handler("fs__read", "read", "Read a file", schema, session)
    from axio.schema import build_tool_schema

    json_schema = build_tool_schema(handler)

    assert "path" in json_schema["properties"]
    assert "count" in json_schema["properties"]
    assert "verbose" in json_schema["properties"]


async def test_call_forwarding() -> None:
    """Handler forwards to MCP session.call_tool."""
    session = _make_mock_session()
    mock_call = cast(AsyncMock, session.call_tool)
    mock_call.return_value = CallToolResult(
        content=[TextContent(type="text", text="hello world")],
        isError=False,
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }
    handler = build_handler("echo__say", "say", "Say something", schema, session)
    result = await handler(message="hi")

    assert result == "hello world"
    mock_call.assert_awaited_once_with("say", {"message": "hi"})


async def test_error_handling() -> None:
    """Handler raises RuntimeError when isError=True."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="not found")],
        isError=True,
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    handler = build_handler("fs__read", "read", "Read file", schema, session)

    with pytest.raises(RuntimeError, match="not found"):
        await handler(path="/missing")


async def test_empty_schema() -> None:
    """Handler works with empty input schema (no params)."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="done")],
        isError=False,
    )

    handler = build_handler("sys__status", "status", "Get status", {}, session)
    result = await handler()
    assert result == "done"


def test_type_mapping() -> None:
    """All JSON schema types are mapped to Python types."""
    from axio.schema import build_tool_schema

    session = _make_mock_session()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "a": {"type": "array"},
            "o": {"type": "object"},
        },
        "required": ["s", "i", "n", "b", "a", "o"],
    }
    handler = build_handler("test__types", "types", "Type test", schema, session)
    json_schema = build_tool_schema(handler)
    props: dict[str, Any] = json_schema["properties"]
    assert props["s"] == {"type": "string"}
    assert props["i"] == {"type": "integer"}
    assert props["n"] == {"type": "number"}
    assert props["b"] == {"type": "boolean"}


def test_optional_param_without_schema_default_has_no_default_key() -> None:
    """Optional MCP params with no schema default must not emit 'default: null'."""
    from axio.schema import build_tool_schema

    session = _make_mock_session()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "required_field": {"type": "string"},
            "optional_no_default": {"type": "string"},
            "optional_with_default": {"type": "string", "default": "hello"},
        },
        "required": ["required_field"],
    }
    handler = build_handler("test__defaults", "defaults", "Default test", schema, session)
    json_schema = build_tool_schema(handler)
    props: dict[str, Any] = json_schema["properties"]

    # optional param with no default in schema must not have 'default' key
    assert "default" not in props["optional_no_default"]
    # optional param with an explicit default must surface it
    assert props["optional_with_default"].get("default") == "hello"
