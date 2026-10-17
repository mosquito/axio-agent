"""Application policy for MCP JSON files, layers, and source diagnostics."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from axio_tools_mcp.config import ConfigError as ConfigError
from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.inspection import ServerInspection

from .paths import ConfigurationPaths, canonical_path, config_directory

type Layer = Literal["global", "project", "private", "explicit"]


def fail(source: Path, name: str | None, reason: str) -> ConfigError:
    server = f", server {name!r}" if name is not None else ""
    return ConfigError(f"MCP config {source}{server}: {reason}")


class JSONPairs(list[tuple[str, Any]]):
    """Keep duplicate keys until their enclosing server is known."""


def unique_objects(value: Any, source: Path, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, JSONPairs):
        result: dict[str, Any] = {}
        for key, item in value:
            if key in result:
                location = (*path, key)
                name = location[1] if len(location) > 1 and location[0] == "mcpServers" else None
                raise fail(source, name, f"duplicate key at {'/'.join(location)!r}")
            result[key] = unique_objects(item, source, (*path, key))
        return result
    if isinstance(value, list):
        return [unique_objects(item, source, (*path, str(i))) for i, item in enumerate(value)]
    return value


@dataclass(frozen=True, slots=True)
class ConfiguredServer(MCPServerConfig):
    """Add application provenance without changing the MCP connection contract."""

    source: Path = field(kw_only=True)
    layer: Layer = field(kw_only=True)


def parse_server(name: str, data: Any, source: Path, layer: Layer) -> ConfiguredServer:
    try:
        config = MCPServerConfig.from_dict(name, data)
        if config.command is not None:
            command = config.command
            if "/" in command:
                command = os.path.abspath(source.parent / command)
            config = replace(config, command=command, cwd=str(canonical_path(Path(config.cwd or "."), source.parent)))
        config = replace(config, inherit_env=False)
    except ValueError as error:
        raise fail(source, name, str(error)) from None
    return ConfiguredServer(**asdict(config), source=source, layer=layer)


@dataclass(frozen=True, slots=True)
class ConfigFile:
    source: Path
    layer: Layer
    servers: tuple[ConfiguredServer, ...]


@dataclass(frozen=True, slots=True)
class MCPConfiguration:
    origin: str
    files: tuple[ConfigFile, ...]

    @property
    def effective(self) -> dict[str, ConfiguredServer]:
        return {server.name: server for config in self.files for server in config.servers}

    def history(self, name: str) -> tuple[ConfiguredServer, ...]:
        return tuple(server for config in self.files for server in config.servers if server.name == name)


def parse_config(content: str, source: Path, layer: Layer) -> ConfigFile:
    """Validate configuration content supplied by a caller; perform no file discovery."""
    try:
        raw = json.loads(content, object_pairs_hook=JSONPairs)
    except (json.JSONDecodeError, RecursionError):
        raise fail(source, None, "invalid JSON") from None
    data = unique_objects(raw, source)
    if not isinstance(data, dict) or set(data) != {"mcpServers"} or not isinstance(data["mcpServers"], dict):
        raise fail(source, None, "expected an object with one mcpServers object")
    servers = tuple(parse_server(name, record, source, layer) for name, record in data["mcpServers"].items())
    return ConfigFile(source, layer, servers)


def read_config(path: Path, layer: Layer, *, required: bool = False) -> ConfigFile | None:
    """Read exactly one file. Missing automatic files are the only skipped errors."""
    try:
        path.lstat()
    except FileNotFoundError:
        if required:
            raise fail(path, None, "file does not exist") from None
        return None
    except OSError:
        raise fail(path, None, "cannot access file") from None
    try:
        source = path.resolve(strict=True)
        if not stat.S_ISREG(source.stat().st_mode):
            raise fail(path, None, "source must be a regular file")
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError, RuntimeError):
        raise fail(path, None, "cannot read file as UTF-8") from None
    return parse_config(content, source, layer)


def load_mcp_config(
    workspace: ConfigurationPaths, *, files: Iterable[Path] = (), isolated: bool = False
) -> tuple[MCPConfiguration, ...]:
    """Apply whole-entry replacement independently for each workspace root."""
    global_config = None if isolated else read_config(config_directory() / "mcp.json", "global")
    explicit = tuple(read_config(workspace.resolve(path), "explicit", required=True) for path in files)
    projects: list[MCPConfiguration] = []
    for root in workspace.roots:
        layers = [global_config]
        if not isolated:
            layers.extend(
                [
                    read_config(root / ".mcp.json", "project"),
                    read_config(root / ".axio" / "mcp.local.json", "private"),
                ]
            )
        layers.extend(explicit)
        projects.append(MCPConfiguration(str(root), tuple(layer for layer in layers if layer is not None)))
    return tuple(projects)


def connection_configs(projects: tuple[MCPConfiguration, ...]) -> tuple[MCPServerConfig, ...]:
    """Give servers unique application names before passing them to the provider."""
    return tuple(
        replace(server, name=server.name if len(projects) == 1 else f"w{index}__{server.name}")
        for index, project in enumerate(projects, 1)
        for server in project.effective.values()
    )


def inspect_configurations(
    projects: tuple[MCPConfiguration, ...], servers: list[ServerInspection]
) -> list[ServerInspection]:
    """Attach application file provenance to provider connection state."""
    sources = {
        server.name if len(projects) == 1 else f"w{index}__{server.name}": (project, server)
        for index, project in enumerate(projects, 1)
        for server in project.effective.values()
    }
    result = []
    for info in servers:
        entry = sources.get(info.name)
        if entry is None:
            result.append(info)
            continue
        project, server = entry
        details = [("Project", project.origin), ("Configuration", str(server.source))]
        details.extend(
            ("Replaced configuration", f"{old.layer}: {old.source}") for old in project.history(server.name)[:-1]
        )
        result.append(replace(info, origin=f"Automatic · {server.layer}", details=(*details, *info.details)))
    return result
