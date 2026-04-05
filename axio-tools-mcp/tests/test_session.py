"""Tests for MCPSession."""

from __future__ import annotations

import logging
import os
from io import TextIOWrapper
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.session import MCPSession


@pytest.fixture
def stdio_config() -> MCPServerConfig:
    return MCPServerConfig(name="test", command="echo", args=["hello"])


@pytest.fixture
def http_config() -> MCPServerConfig:
    return MCPServerConfig(name="remote", url="http://localhost:8000/mcp")


def _mock_client_session() -> MagicMock:
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=ListToolsResult(
            tools=[
                MCPTool(name="add", description="Add numbers", inputSchema={"type": "object"}),
            ],
        ),
    )
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="3")],
            isError=False,
        ),
    )
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


async def test_connect_stdio(stdio_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        mock_read, mock_write = MagicMock(), MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()

        assert session.is_connected
        mock_session.initialize.assert_awaited_once()
        await session.close()
        assert not session.is_connected


async def test_connect_http(http_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.streamable_http_client") as mock_http,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        mock_read, mock_write = MagicMock(), MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, lambda: "sid"))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_http.return_value = ctx

        session = MCPSession(http_config)
        await session.connect()

        assert session.is_connected
        mock_session.initialize.assert_awaited_once()
        await session.close()


async def test_list_tools(stdio_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()
        tools = await session.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "add"
        await session.close()


async def test_call_tool(stdio_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()
        result = await session.call_tool("add", {"a": 1, "b": 2})

        assert not result.isError
        assert result.content[0].text == "3"  # type: ignore[union-attr]
        await session.close()


async def test_call_tool_not_connected(stdio_config: MCPServerConfig) -> None:
    session = MCPSession(stdio_config)
    with pytest.raises(RuntimeError, match="Not connected"):
        await session.call_tool("add", {})


async def test_list_tools_not_connected(stdio_config: MCPServerConfig) -> None:
    session = MCPSession(stdio_config)
    with pytest.raises(RuntimeError, match="Not connected"):
        await session.list_tools()


async def test_stdio_client_receives_errlog(stdio_config: MCPServerConfig) -> None:
    """stdio_client must be called with a writable errlog pipe, not sys.stderr."""
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()

        _, kwargs = mock_stdio.call_args
        errlog = kwargs.get("errlog")
        assert errlog is not None, "errlog should be passed to stdio_client"
        assert isinstance(errlog, TextIOWrapper), "errlog should be a file object (pipe write-end)"
        assert not errlog.closed

        await session.close()
        assert errlog.closed


async def test_stderr_lines_logged(stdio_config: MCPServerConfig, caplog: pytest.LogCaptureFixture) -> None:
    """Lines written to the MCP process stderr appear as logger.warning records."""
    read_fd, write_fd = os.pipe()

    with (
        patch("axio_tools_mcp.session.os.pipe", return_value=(read_fd, write_fd)),
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=_mock_client_session()),
        caplog.at_level(logging.WARNING, logger="axio_tools_mcp.session"),
    ):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()

        # Write data via the raw fd (still valid since errlog hasn't been closed yet)
        os.write(write_fd, b"something went wrong\nanother line\n")

        # close() flushes and closes errlog → pipe EOF → _read_stderr task exits
        await session.close()

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("something went wrong" in m for m in messages)
    assert any("another line" in m for m in messages)
    assert all("[mcp:test]" in m for m in messages)
