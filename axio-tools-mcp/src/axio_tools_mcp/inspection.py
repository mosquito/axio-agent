"""Read-only MCP inspection data shared by application interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from aiohttp_tiny_mcp.protocol.models import ResourceDef, ResourceTemplateDef
from axio.tool import Tool

from .skills import Skill


@dataclass(frozen=True, slots=True)
class ServerInspection:
    name: str
    state: str
    origin: str
    details: tuple[tuple[str, str], ...]
    tools: tuple[Tool[Any], ...] = ()
    resources: tuple[ResourceDef, ...] = ()
    templates: tuple[ResourceTemplateDef, ...] = ()
    skills: tuple[Skill, ...] = ()
    errors: tuple[str, ...] = ()


def connection_details(
    *,
    transport: str,
    url: str | None,
    cwd: str | None,
    argument_count: int,
    environment_count: int,
    header_count: int,
    timeout: float,
    requested_protocol: str | None,
    negotiated_protocol: str | None,
) -> tuple[tuple[str, str], ...]:
    """Describe connections without exposing command, argument or credential values."""
    fields = [("Transport", transport)]
    if transport == "http":
        try:
            parts = urlsplit(url or "")
            host = parts.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            port = f":{parts.port}" if parts.port is not None else ""
            origin = f"{parts.scheme}://{host}{port}"
        except ValueError:
            origin = "Invalid endpoint"
        fields.extend(
            [
                ("Endpoint origin", origin),
                ("Endpoint path / query / credentials", "[redacted]"),
                ("Headers", f"{header_count} values [redacted]"),
            ]
        )
    elif transport == "stdio":
        fields.extend(
            [
                ("Working directory", cwd or "Inherited from application"),
                ("Command", "[redacted]"),
                ("Arguments", f"{argument_count} values [redacted]"),
                ("Environment overrides", f"{environment_count} values [redacted]"),
            ]
        )
    fields.extend(
        [
            ("Timeout", f"{timeout:g}s"),
            ("Requested protocol", requested_protocol or "Negotiate"),
            ("Negotiated protocol", negotiated_protocol or "Not connected"),
        ]
    )
    return tuple(fields)
