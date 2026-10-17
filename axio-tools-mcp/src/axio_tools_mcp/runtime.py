"""Connect configured MCP servers and expose their tools for one REPL session."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from aiohttp_tiny_mcp import CallToolResult, ClientError
from aiohttp_tiny_mcp.protocol.models import ReadResourceResult, ResourceDef, ResourceTemplateDef
from axio.tool import Tool

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.handler import build_tools
from axio_tools_mcp.inspection import ServerInspection, connection_details
from axio_tools_mcp.resources import ResourceCatalog
from axio_tools_mcp.session import MCPSession
from axio_tools_mcp.skills import SkillCatalog
from axio_tools_mcp.timeouts import error_summary, mcp_timeout

from .config import ConfigError


class Session(MCPSession):
    """Retain failure state and keep raw transport errors out of model diagnostics."""

    failed: str | None = None

    async def resource_request[T](self, request: Callable[[], Awaitable[T]]) -> T:
        """Bound resource requests and invalidate interrupted stdio connections."""
        if self.failed is not None:
            raise RuntimeError(f"MCP server {self.config.name!r} is unavailable; use /mcp reload")
        try:
            async with asyncio.timeout(self.config.timeout):
                return await request()
        except ClientError:
            # A completed protocol error leaves the connection usable.
            raise RuntimeError(f"MCP server {self.config.name!r} rejected the resource request") from None
        except BaseException as error:
            self.failed = type(error).__name__
            if isinstance(error, asyncio.CancelledError):
                raise
            if not isinstance(error, Exception):
                raise
            if isinstance(error, TimeoutError):
                raise RuntimeError(
                    f"MCP server {self.config.name!r}: resource request timed out after "
                    f"{self.config.timeout:g}s; use /mcp reload"
                ) from None
            raise RuntimeError(f"MCP server {self.config.name!r}: {self.failed}; use /mcp reload") from None

    async def list_resources(self) -> list[ResourceDef]:
        return await self.resource_request(super().list_resources)

    async def list_resource_templates(self) -> list[ResourceTemplateDef]:
        return await self.resource_request(super().list_resource_templates)

    async def read_resource(self, uri: str) -> ReadResourceResult:
        read = super().read_resource
        return await self.resource_request(lambda: read(uri))

    async def log_stderr(self, stream: asyncio.StreamReader) -> None:
        # Server stderr can contain tokens. Drain it without forwarding raw text.
        while await stream.read(65536):
            pass

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        if self.failed is not None:
            raise RuntimeError(f"MCP server {self.config.name!r} is unavailable; use /mcp reload")
        try:
            async with asyncio.timeout(self.config.timeout):
                return await super().call_tool(name, arguments)
        except ClientError:
            # A completed protocol error leaves the connection usable.
            raise RuntimeError(f"MCP server {self.config.name!r} rejected the tool request") from None
        except asyncio.CancelledError:
            # An interrupted stdio request can leave its response in the stream.
            self.failed = "cancelled request"
            raise
        except TimeoutError:
            self.failed = "TimeoutError"
            raise RuntimeError(
                f"MCP server {self.config.name!r}: tool request timed out after "
                f"{self.config.timeout:g}s; use /mcp reload"
            ) from None
        except Exception as error:
            self.failed = type(error).__name__
            raise RuntimeError(f"MCP server {self.config.name!r}: {self.failed}; use /mcp reload") from None


@dataclass(slots=True)
class Server:
    config: MCPServerConfig
    status: str = "configured"
    session: Session | None = None
    exposed: list[Tool[Any]] = field(default_factory=list)
    error: str | None = None
    skills: SkillCatalog | None = None
    skill_error: str | None = None
    resources: ResourceCatalog | None = None
    resource_error: str | None = None

    @property
    def alias(self) -> str:
        return self.config.name

    @property
    def state(self) -> str:
        if self.status == "connected" and self.session is not None:
            process = self.session.process
            if self.session.failed is not None or not self.session.is_connected:
                return "error"
            if process is not None and process.returncode is not None:
                return "error"
        return self.status

    @property
    def tools(self) -> list[Tool[Any]]:
        return self.exposed if self.state == "connected" else []


class MCPRuntime:
    """Own all MCP connections, including partial startup and reload failures."""

    def __init__(self) -> None:
        self.servers: list[Server] = []
        self.on_status: Callable[[str], None] = lambda message: print(message, file=sys.stderr)

    @property
    def tools(self) -> list[Tool[Any]]:
        return [tool for server in self.servers for tool in server.tools]

    async def reload(self, configs: tuple[MCPServerConfig, ...], *, reserved: set[str]) -> None:
        names = [config.name for config in configs]
        if len(names) != len(set(names)):
            raise ConfigError("Duplicate MCP server name")
        await self.close()
        self.servers = [Server(config, status="configured" if config.enabled else "disabled") for config in configs]
        used = set(reserved)
        try:
            for server in self.servers:
                if server.status != "configured":
                    continue
                await self.connect(server)
                for tool in server.tools:
                    if tool.name in used:
                        raise ConfigError(f"Duplicate MCP tool name {tool.name!r}")
                    if re.fullmatch(r"[A-Za-z0-9_-]+", tool.name) is None:
                        raise ConfigError(f"Invalid MCP tool name {tool.name!r}")
                    used.add(tool.name)
        except BaseException:
            await self.close()
            raise

    async def connect(self, server: Server) -> None:
        self.on_status(f"MCP {server.alias!r}: initializing (timeout {server.config.timeout:g}s)...")
        try:
            async with mcp_timeout(server.config.timeout, "Initialization (handshake and catalogs)"):
                await self._connect(server)
        except TimeoutError as error:
            server.status = "error"
            server.error = error_summary(error)
            server.exposed = []
            server.skills = None
            server.resources = None
            if server.session is not None:
                await server.session.close()
        detail = f"; {server.error}" if server.error else ""
        retry = "; use /mcp reload" if server.state == "error" else ""
        self.on_status(f"MCP {server.alias!r}: {server.state}{detail}{retry}")

    async def _connect(self, server: Server) -> None:
        config = server.config
        try:
            session = await config.connect(Session)
            assert isinstance(session, Session)
            server.session = session
            definitions = await session.list_tools() if "tools" in session.server_capabilities else []
            server.exposed = build_tools(session, definitions)
            server.status = "connected"
        except Exception as error:
            if server.session is not None:
                await server.session.close()
            server.status = "error"
            server.error = error_summary(error)
            return
        if "resources" in session.server_capabilities:
            server.resources = ResourceCatalog(session)
            server.exposed.extend(
                [
                    Tool(name=f"{server.alias}__list_resources", handler=server.resources.list_resources),
                    Tool(name=f"{server.alias}__read_resource", handler=server.resources.read_resource),
                ]
            )
            try:
                await server.resources.refresh()
            except Exception as error:
                server.resource_error = error_summary(error)
                self.on_status(
                    f"MCP server {config.name!r}: resource discovery failed ({server.resource_error}). "
                    "Use /mcp reload."
                )
        try:
            resources = server.resources.resources if server.resources is not None else []
            server.skills = await SkillCatalog.discover(session, resources)
            if server.skills.skills:
                server.exposed.append(Tool(name=f"{server.alias}__read_skill", handler=server.skills.read))
        except Exception as error:
            server.skill_error = error_summary(error)
            self.on_status(
                f"MCP server {config.name!r}: skill discovery failed ({server.skill_error}). Use /mcp reload."
            )

        server.exposed = [
            replace(tool, permission_identity=session.config.permission_identity) for tool in server.exposed
        ]

    async def close(self) -> None:
        # Attempt every close even when one transport cannot shut down cleanly.
        sessions = [server.session for server in self.servers if server.session is not None]
        for server in self.servers:
            server.exposed = []
            server.skills = None
            server.resources = None
            if server.status == "connected":
                server.status = "disconnected"
        results = await asyncio.gather(*(session.close() for session in sessions), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                print(f"MCP cleanup failed ({type(result).__name__})", file=sys.stderr)

    def inspect_servers(self) -> list[ServerInspection]:
        """Snapshot current state and cached catalogs without reconnecting servers."""
        result = []
        for server in self.servers:
            config, session = server.config, server.session
            details = [("Enabled", str(config.enabled))]
            details.extend(
                connection_details(
                    transport=config.transport or "Not configured",
                    url=config.url,
                    cwd=str(config.cwd) if config.cwd is not None else None,
                    argument_count=len(config.args),
                    environment_count=len(config.env or {}),
                    header_count=len(config.headers),
                    timeout=config.timeout,
                    requested_protocol=config.protocol_version,
                    negotiated_protocol=session.protocol_version if session is not None else None,
                )
            )
            details.extend(
                [
                    ("Environment references", str(len(config.env_from))),
                    ("Header references", str(len(config.headers_from))),
                    (
                        "Capabilities",
                        ", ".join(sorted(session.server_capabilities)) if session is not None else "None",
                    ),
                ]
            )
            errors = [error for error in (server.error, session.failed if session else None) if error]
            if server.resource_error:
                errors.append(f"Resource discovery failed: {server.resource_error}")
            if server.skill_error:
                errors.append(f"Skill discovery failed: {server.skill_error}")
            if server.state == "error" and not errors:
                errors.append("Connection is no longer available. Use /mcp reload.")
            connected = server.state == "connected"
            result.append(
                ServerInspection(
                    name=server.alias,
                    state=server.state,
                    origin="Configured",
                    details=tuple(details),
                    tools=tuple(server.tools),
                    resources=tuple(server.resources.resources) if connected and server.resources else (),
                    templates=tuple(server.resources.templates) if connected and server.resources else (),
                    skills=tuple(server.skills.skills.values()) if connected and server.skills else (),
                    errors=tuple(errors),
                )
            )
        return result
