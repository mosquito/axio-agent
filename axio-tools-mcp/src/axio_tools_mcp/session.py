"""MCPSession: wraps mcp.ClientSession with explicit lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import os
from asyncio import Task
from contextlib import AsyncExitStack

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPSession:
    """Manages connection lifecycle for a single MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._stderr_task: Task[None] | None = None

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @staticmethod
    async def _read_stderr(read_fd: int, server_name: str) -> None:
        """Read lines from pipe and forward to logger as warnings."""
        loop = asyncio.get_running_loop()
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as pipe_r:
                while True:
                    line = await loop.run_in_executor(None, pipe_r.readline)
                    if not line:
                        break
                    line = line.rstrip("\n\r")
                    if line:
                        logger.warning("[mcp:%s] %s", server_name, line)
        except Exception:
            pass

    async def connect(self) -> None:
        """Connect to the MCP server and initialize the session."""
        if self._session is not None:
            return

        if self._config.command is not None:
            read_fd, write_fd = os.pipe()
            errlog = os.fdopen(write_fd, "w", buffering=1, encoding="utf-8", errors="replace")
            self._exit_stack.callback(errlog.close)
            self._stderr_task = asyncio.create_task(self._read_stderr(read_fd, self._config.name))
            params = StdioServerParameters(
                command=self._config.command,
                args=self._config.args,
                env=self._config.env,
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(params, errlog=errlog))
        elif self._config.url is not None:
            http_client = httpx.AsyncClient(
                headers=self._config.headers or None,
                timeout=self._config.timeout,
            )
            read, write, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(self._config.url, http_client=http_client),
            )
        else:
            raise ValueError("No command or url configured")

        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("Connected to MCP server %r", self._config.name)

    async def list_tools(self) -> list[Tool]:
        """List available tools from the connected MCP server."""
        if self._session is None:
            raise RuntimeError("Not connected")
        result = await self._session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, object]) -> CallToolResult:
        """Call a tool on the connected MCP server."""
        if self._session is None:
            raise RuntimeError("Not connected")
        return await self._session.call_tool(name, arguments=arguments)

    async def close(self) -> None:
        """Close the session and release resources."""
        self._session = None
        await self._exit_stack.aclose()  # closes errlog write-end → pipe EOF → _read_stderr exits
        self._exit_stack = AsyncExitStack()
        if self._stderr_task is not None:
            try:
                await asyncio.wait_for(self._stderr_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                self._stderr_task.cancel()
            self._stderr_task = None
        logger.info("Disconnected from MCP server %r", self._config.name)
