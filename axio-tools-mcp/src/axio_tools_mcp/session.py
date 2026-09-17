"""MCPSession: one connection to an MCP server, over stdio or HTTP."""

from __future__ import annotations

import asyncio
import logging
import os
from asyncio import Task
from contextlib import AbstractAsyncContextManager, nullcontext
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import aiohttp
from aiohttp_tiny_mcp import CallToolResult, Client, ClientError, StdioClient
from aiohttp_tiny_mcp.adapter import Adapter
from aiohttp_tiny_mcp.client_base import BaseClient
from aiohttp_tiny_mcp.models import Implementation, ToolDef
from aiohttp_tiny_mcp.protocol.selection import AdapterSet

from .config import MCPServerConfig

logger = logging.getLogger(__name__)

ADAPTERS = AdapterSet.default()

# The handshake starts on this revision because every current server answers it.
# 2026-07-28 replaces `initialize` with `server/discover`, so it cannot open one.
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

# One line holds one JSON-RPC message, and a tool result can be large.
STDIO_LINE_LIMIT = 16 * 1024 * 1024

# How long to wait for a stdio server to stop before killing it.
TERMINATE_TIMEOUT = 5.0

# How long to wait for the standard error reader to drain a closed pipe.
STDERR_DRAIN_TIMEOUT = 2.0


def supported_version(error: ClientError) -> str | None:
    """Return the newest revision a refusal names and this client also speaks."""
    data = error.data if isinstance(error.data, dict) else {}
    offered = data.get("supported")
    if not isinstance(offered, list):
        return None
    known = sorted(v for v in offered if isinstance(v, str) and v in ADAPTERS.by_version)
    return known[-1] if known else None


def client_info() -> Implementation:
    """Return the identity this client reports to the server."""
    try:
        release = version("axio-tools-mcp")
    except PackageNotFoundError:
        release = "0"
    return Implementation(name="axio", version=release)


async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Ask a stdio server to stop, then kill it if it does not."""
    if process.returncode is not None:
        return
    if process.stdin is not None and not process.stdin.is_closing():
        process.stdin.close()  # EOF tells a stdio server to shut down.
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=TERMINATE_TIMEOUT)
    except TimeoutError:
        process.kill()
        await process.wait()


async def stop_reader(task: Task[None]) -> None:
    """Let a reader task drain its closed pipe, then drop it."""
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=STDERR_DRAIN_TIMEOUT)
    except (TimeoutError, asyncio.CancelledError):
        task.cancel()
    except Exception:
        logger.debug("MCP stderr reader failed", exc_info=True)


class MCPSession:
    """Manages connection lifecycle for a single MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.client: BaseClient | None = None
        self.http: aiohttp.ClientSession | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.stderr_task: Task[None] | None = None
        # stdio carries every request on one channel, so calls must not interleave.
        self.lock: asyncio.Lock | None = None

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    @property
    def protocol_version(self) -> str | None:
        """Return the revision in use, or None while disconnected."""
        return self.client.adapter.version if self.client is not None else None

    async def connect(self) -> None:
        """Connect to the MCP server and complete the handshake."""
        if self.client is not None:
            return
        wanted = self.config.protocol_version or DEFAULT_PROTOCOL_VERSION
        if wanted not in ADAPTERS.by_version:
            raise ValueError(f"Unsupported MCP protocol version {wanted!r}")
        try:
            await self.open(wanted)
        except ClientError as error:
            # A server that speaks neither revision names the ones it does.
            second = None if self.config.protocol_version else supported_version(error)
            await self.close()
            if second is None:
                raise
            logger.info("MCP server %r refused %s and requires %s", self.config.name, wanted, second)
            await self.open(second)
        except BaseException:
            await self.close()
            raise
        logger.info("Connected to MCP server %r on %s", self.config.name, self.protocol_version)

    async def open(self, protocol_version: str) -> None:
        """Start the transport and complete the handshake on one revision."""
        adapter = ADAPTERS.by_version[protocol_version]
        if self.config.command is not None:
            self.client = await self.connect_stdio(adapter)
        elif self.config.url is not None:
            self.client = self.connect_http(adapter)
        else:
            raise ValueError("No command or url configured")
        await asyncio.wait_for(self.handshake(), timeout=self.config.timeout)

    async def connect_stdio(self, adapter: Adapter) -> StdioClient:
        """Start the server process and talk to its standard streams."""
        assert self.config.command is not None
        process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.environment(),
            limit=STDIO_LINE_LIMIT,
        )
        self.process = process
        assert process.stdin is not None and process.stdout is not None
        assert process.stderr is not None
        self.stderr_task = asyncio.create_task(self.log_stderr(process.stderr))
        self.lock = asyncio.Lock()
        return StdioClient(process.stdout, process.stdin, adapter, client_info=client_info())

    def connect_http(self, adapter: Adapter) -> Client:
        """Open an HTTP session for the Streamable HTTP transport."""
        assert self.config.url is not None
        timeout = aiohttp.ClientTimeout(connect=self.config.timeout, sock_read=self.config.timeout)
        self.http = aiohttp.ClientSession(headers=self.config.headers or None, timeout=timeout)
        return Client(self.config.url, adapter, session=self.http, client_info=client_info())

    async def handshake(self) -> None:
        """Initialize, then follow the server to the revision it answered with."""
        assert self.client is not None
        result = await self.client.initialize()
        agreed = result.get("protocolVersion")
        if not isinstance(agreed, str) or agreed == self.client.adapter.version:
            return
        adapter = ADAPTERS.by_version.get(agreed)
        if adapter is None:
            logger.warning(
                "MCP server %r answered with unknown protocol version %r, keeping %s",
                self.config.name,
                agreed,
                self.client.adapter.version,
            )
            return
        logger.info("MCP server %r selected protocol version %s", self.config.name, agreed)
        self.client.adapter = adapter

    def environment(self) -> dict[str, str]:
        """Return the child environment: this process's, plus the configured extras."""
        env = dict(os.environ)
        if self.config.env:
            env.update(self.config.env)
        return env

    async def log_stderr(self, stream: asyncio.StreamReader) -> None:
        """Forward the server's standard error to the logger as warnings."""
        while True:
            try:
                line = await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                continue  # An over-long line is noise, not a reason to stop reading.
            except (asyncio.CancelledError, ConnectionResetError):
                return
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if text:
                logger.warning("[mcp:%s] %s", self.config.name, text)

    def exclusive(self) -> AbstractAsyncContextManager[Any]:
        """Serialize requests on stdio; HTTP requests are independent."""
        return self.lock if self.lock is not None else nullcontext()

    async def list_tools(self) -> list[ToolDef]:
        """List available tools from the connected MCP server."""
        if self.client is None:
            raise RuntimeError("Not connected")
        async with self.exclusive():
            return await self.client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call a tool on the connected MCP server."""
        if self.client is None:
            raise RuntimeError("Not connected")
        async with self.exclusive():
            result = await self.client.call_tool(name, arguments)
        if not isinstance(result, CallToolResult):
            # The server asked a question, which an agent tool call cannot answer.
            raise RuntimeError(f"MCP tool {name!r} requires input this client cannot supply")
        return result

    async def close(self) -> None:
        """Close the session and release resources."""
        connected = self.client is not None
        self.client = None
        self.lock = None
        if self.http is not None:
            await self.http.close()
            self.http = None
        if self.process is not None:
            await stop_process(self.process)
            self.process = None
        if self.stderr_task is not None:
            await stop_reader(self.stderr_task)
            self.stderr_task = None
        if connected:
            logger.info("Disconnected from MCP server %r", self.config.name)
