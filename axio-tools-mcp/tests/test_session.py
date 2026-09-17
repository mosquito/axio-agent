"""Tests for MCPSession against real MCP servers, over stdio and HTTP."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.typedefs import Handler
from aiohttp_tiny_mcp import ClientError, Registry
from aiohttp_tiny_mcp.protocol.selection import AdapterSet
from aiohttp_tiny_mcp.protocol.v2025_11_25 import Adapter2025_11_25
from aiohttp_tiny_mcp.testing import serving
from pydantic import BaseModel

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.handler import result_text
from axio_tools_mcp.session import DEFAULT_PROTOCOL_VERSION, MCPSession

SERVER_SCRIPT = str(Path(__file__).parent / "stdio_server.py")

AUTHORIZATION: list[str | None] = []


class Nothing(BaseModel):
    pass


class Message(BaseModel):
    message: str


def http_registry() -> Registry:
    registry = Registry("fixture", "1.0")

    @registry.tool
    async def echo(args: Message) -> str:
        """Return the message."""
        return args.message

    @registry.tool
    async def explode(args: Nothing) -> str:
        """Always fail."""
        raise RuntimeError("tool exploded")

    return registry


@web.middleware
async def record_authorization(request: web.Request, handler: Handler) -> web.StreamResponse:
    AUTHORIZATION.append(request.headers.get("Authorization"))
    return await handler(request)


@pytest.fixture
def stdio_config() -> MCPServerConfig:
    return MCPServerConfig(name="fixture", command=sys.executable, args=[SERVER_SCRIPT])


@pytest.fixture
async def http_url() -> AsyncIterator[str]:
    async with serving(http_registry(), middlewares=[record_authorization]) as url:
        AUTHORIZATION.clear()
        yield url


@pytest.fixture
async def legacy_url() -> AsyncIterator[str]:
    """A server that speaks one revision, and not the one the client starts with."""
    async with serving(http_registry(), adapters=AdapterSet([Adapter2025_11_25()])) as url:
        yield url


async def test_stdio_round_trip(stdio_config: MCPServerConfig) -> None:
    session = MCPSession(stdio_config)
    await session.connect()
    try:
        assert session.is_connected
        assert session.protocol_version == DEFAULT_PROTOCOL_VERSION
        assert {tool.name for tool in await session.list_tools()} == {"echo", "token", "explode"}
        result = await session.call_tool("echo", {"message": "hi"})
        assert result.is_error is False
        assert result_text(result) == "hi"
    finally:
        await session.close()
    assert not session.is_connected


async def test_stdio_passes_extra_environment() -> None:
    config = MCPServerConfig(
        name="fixture",
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env={"AXIO_MCP_TEST_TOKEN": "secret"},
    )
    session = MCPSession(config)
    await session.connect()
    try:
        result = await session.call_tool("token", {})
        assert result_text(result) == "secret"
    finally:
        await session.close()


async def test_stdio_standard_error_is_logged(stdio_config: MCPServerConfig, caplog: pytest.LogCaptureFixture) -> None:
    session = MCPSession(stdio_config)
    with caplog.at_level(logging.WARNING, logger="axio_tools_mcp.session"):
        await session.connect()
        await session.close()
    messages = [record.getMessage() for record in caplog.records]
    assert any("[mcp:fixture] fixture server ready" in message for message in messages)


async def test_stdio_serializes_concurrent_calls(stdio_config: MCPServerConfig) -> None:
    """One channel carries every request, so answers must not be mixed up."""
    session = MCPSession(stdio_config)
    await session.connect()
    try:
        calls = [session.call_tool("echo", {"message": str(number)}) for number in range(8)]
        results = await asyncio.gather(*calls)
    finally:
        await session.close()
    assert [result_text(result) for result in results] == [str(n) for n in range(8)]


async def test_failing_tool_returns_an_error_result(stdio_config: MCPServerConfig) -> None:
    session = MCPSession(stdio_config)
    await session.connect()
    try:
        result = await session.call_tool("explode", {})
    finally:
        await session.close()
    assert result.is_error is True
    assert "tool exploded" in result_text(result)


async def test_connect_failure_leaves_nothing_open() -> None:
    config = MCPServerConfig(name="missing", command="/nonexistent/mcp-server")
    session = MCPSession(config)
    with pytest.raises(FileNotFoundError):
        await session.connect()
    assert not session.is_connected
    assert session.process is None


async def test_http_round_trip(http_url: str) -> None:
    session = MCPSession(MCPServerConfig(name="remote", url=http_url))
    await session.connect()
    try:
        assert {tool.name for tool in await session.list_tools()} == {"echo", "explode"}
        result = await session.call_tool("echo", {"message": "hi"})
        assert result_text(result) == "hi"
    finally:
        await session.close()
    assert not session.is_connected


async def test_http_sends_configured_headers(http_url: str) -> None:
    config = MCPServerConfig(name="remote", url=http_url, headers={"Authorization": "Bearer token"})
    session = MCPSession(config)
    await session.connect()
    try:
        await session.list_tools()
    finally:
        await session.close()
    assert AUTHORIZATION and set(AUTHORIZATION) == {"Bearer token"}


async def test_connect_follows_the_revision_the_server_requires(legacy_url: str) -> None:
    session = MCPSession(MCPServerConfig(name="legacy", url=legacy_url))
    await session.connect()
    try:
        assert session.protocol_version == "2025-11-25"
        assert {tool.name for tool in await session.list_tools()} == {"echo", "explode"}
    finally:
        await session.close()


async def test_pinned_revision_is_not_replaced(legacy_url: str) -> None:
    config = MCPServerConfig(name="legacy", url=legacy_url, protocol_version="2025-06-18")
    session = MCPSession(config)
    with pytest.raises(ClientError):
        await session.connect()
    assert not session.is_connected


async def test_unknown_revision_is_refused() -> None:
    config = MCPServerConfig(name="fixture", command="true", protocol_version="1999-01-01")
    session = MCPSession(config)
    with pytest.raises(ValueError, match="Unsupported MCP protocol version"):
        await session.connect()


async def test_calls_need_a_connection(stdio_config: MCPServerConfig) -> None:
    session = MCPSession(stdio_config)
    with pytest.raises(RuntimeError, match="Not connected"):
        await session.list_tools()
    with pytest.raises(RuntimeError, match="Not connected"):
        await session.call_tool("echo", {})
