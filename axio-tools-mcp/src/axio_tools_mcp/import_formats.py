"""Convert explicit client configuration data into the MCP configuration format."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from .config import ConfigError, MCPServerConfig, _string

type Client = Literal["claude", "cursor"]
type Scope = Literal["user", "project", "local"]
type Destination = Literal["user", "project", "private"]


def source_entries(data: dict[str, Any], client: Client, scope: Scope, root: Path, source: Path) -> dict[str, Any]:
    section = data
    if client == "claude" and scope == "local":
        projects = data.get("projects", {})
        if not isinstance(projects, dict):
            raise fail(source, None, "projects must be an object")
        section = projects.get(str(root), {})
    if not isinstance(section, dict) or not isinstance(section.get("mcpServers", {}), dict):
        raise fail(source, None, "mcpServers must be an object in the selected scope")
    return dict(section.get("mcpServers", {}))


def fail(source: Path, name: str | None, reason: str) -> ConfigError:
    return ConfigError(f"MCP config {source}, server {name!r}: {reason}")


def canonical_path(path: Path, base: Path) -> Path:
    return (base / path).resolve()


def string(value: Any, source: Path, name: str, key: str) -> str:
    try:
        return _string(value, key)
    except ConfigError as error:
        raise fail(source, name, str(error)) from None


def serialize(server: MCPServerConfig) -> dict[str, Any]:
    return server.to_dict()


def reject_templates(value: Any, source: Path, name: str) -> None:
    if isinstance(value, str) and "${" in value:
        raise fail(source, name, "unsupported variable expression; use literal paths or envFrom/headersFrom")
    if isinstance(value, list):
        for item in value:
            reject_templates(item, source, name)
    if isinstance(value, dict):
        for item in value.values():
            reject_templates(item, source, name)


def convert(name: str, entry: Any, client: Client, source: Path, server_cwd: Path | None) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise fail(source, name, "server entry must be an object")
    data = dict(entry)
    for key in ("envFile", "auth", "oauth", "headersHelper"):
        if key in data:
            raise fail(source, name, f"{key} is not supported by this importer")
    if client == "claude" and "url" in data and "type" not in data:
        raise fail(source, name, "Claude Code URL entries must specify type")
    if data.get("type") == "streamable-http":
        data["type"] = "http"
    expression = r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}" if client == "claude" else r"\$\{env:([^{}]+)\}"
    for key, ref_key in (("env", "envFrom"), ("headers", "headersFrom")):
        values = data.get(key)
        if not isinstance(values, dict):
            continue
        literal = dict(values)
        refs = data.get(ref_key, {})
        if not isinstance(refs, dict):
            raise fail(source, name, f"{ref_key} must be an object")
        refs = dict(refs)
        for variable, value in values.items():
            match = re.fullmatch(expression, value) if isinstance(value, str) else None
            if match is not None:
                if variable in refs:
                    raise fail(source, name, f"duplicate {ref_key} reference")
                refs[variable] = match[1]
                del literal[variable]
        data[key] = literal
        if refs:
            data[ref_key] = refs
    reject_templates(data, source, name)
    if "command" in data and "cwd" not in data:
        if server_cwd is None:
            raise fail(source, name, "stdio import needs --server-cwd because the source does not declare cwd")
        data["cwd"] = str(server_cwd)
    if "command" in data and isinstance(data["command"], str) and "/" in data["command"]:
        # Relative executable paths use the explicitly chosen runtime directory.
        cwd = canonical_path(Path(string(data.get("cwd", "."), source, name, "cwd")), source.parent)
        data["command"] = os.path.abspath(cwd / data["command"])
    try:
        spec = MCPServerConfig.from_dict(name, data)
    except ConfigError as error:
        raise fail(source, name, str(error)) from None
    return serialize(spec)
