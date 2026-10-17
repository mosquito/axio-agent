"""Terminal presentation of local sources and MCP state."""

from __future__ import annotations

import json
import shutil
import textwrap
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from .paths import ConfigurationPaths

if TYPE_CHECKING:
    from axio_tools_local.instructions import Instructions
    from axio_tools_local.paths import LocalPaths
    from axio_tools_local.skills import LocalSkills
    from axio_tools_mcp.runtime import MCPRuntime

    from .configuration import MCPConfiguration


def format_skill(name: str, source: str, description: str) -> str:
    width = max(20, min(100, shutil.get_terminal_size().columns))
    lines = ["─" * width, name, ""]
    # Keep names and locations intact so they can be copied from terminal output.
    lines.append(f"  {source}")
    lines.append("")
    lines.append(textwrap.fill(description, width, initial_indent="  ", subsequent_indent="  "))
    return "\n".join(lines)


def describe_local_skills(skills: LocalSkills) -> str:
    blocks = [f"Local skills ({len(skills.catalog.skills)})"] if skills.catalog.skills else []
    blocks.extend(
        format_skill(f"local:{skill.name}", str(skill.source), skill.description)
        for skill in skills.catalog.skills.values()
    )
    if skills.catalog.diagnostics:
        blocks.append("Discovery notes:\n" + "\n".join(f"  - {message}" for message in skills.catalog.diagnostics))
    return "\n\n".join(blocks) if blocks else "No local skills available."


def describe_mcp(runtime: MCPRuntime, projects: tuple[MCPConfiguration, ...] = ()) -> str:
    if not runtime.servers:
        return "No MCP servers configured."
    origins = {
        name if len(projects) == 1 else f"w{index}__{name}": project.origin
        for index, project in enumerate(projects, 1)
        for name in project.effective
    }
    lines = ["SERVER  TOOLS  STATE  PROJECT"]
    for server in runtime.servers:
        detail = server.error or (server.session.failed if server.session is not None else None)
        state = server.state + (f" ({detail})" if detail else "")
        lines.append(f"{server.alias!r}  {len(server.tools)}  {state}  {origins.get(server.alias, 'Configured')}")
    lines.append(f"Active: {sum(s.state == 'connected' for s in runtime.servers)}; tools: {len(runtime.tools)}")
    return "\n".join(lines)


def describe_mcp_skills(runtime: MCPRuntime) -> str:
    lines = []
    for server in runtime.servers:
        if server.skill_error:
            lines.append(f"{server.alias}: skill discovery failed ({server.skill_error}); use /mcp reload")
        if server.state != "connected" or server.skills is None:
            continue
        if server.skills.skills:
            lines.append(f"MCP skills: {server.alias} ({len(server.skills.skills)})")
        for skill in server.skills.skills.values():
            lines.append(format_skill(f"{server.alias}:{skill.name}", skill.uri, skill.description))
    return "\n\n".join(lines) if lines else "No MCP skills available."


def explain_config(
    workspace: ConfigurationPaths,
    projects: tuple[MCPConfiguration, ...],
    instructions: Instructions | None,
    local_paths: LocalPaths | None = None,
) -> str:
    lines = [
        instructions.explain(local_paths, workspace.cwd)
        if instructions is not None and local_paths is not None
        else f"Working directory: {workspace.cwd}",
        "",
        "MCP configuration:",
    ]
    for project in projects:
        lines.append(f"\nProject: {project.origin}")
        for index, config in enumerate(project.files, 1):
            lines.append(f"  Layer {index}: {config.layer} {config.source}")
        if not project.effective:
            lines.append("  No MCP servers configured.")
        for name, server in project.effective.items():
            lines.append(f"  Server {json.dumps(name)}: {'enabled' if server.enabled else 'disabled'}")
            history = project.history(name)
            for index, entry in enumerate(history):
                label = "effective" if index == len(history) - 1 else "shadowed (whole entry replaced)"
                lines.append(f"    {entry.layer}: {entry.source} [{label}]")
            if not server.enabled:
                continue
            lines.append(f"    All configured fields come from: {server.source}")
            lines.append(f"    Transport: {server.transport}")
            if server.transport == "stdio":
                lines.append(f"    cwd: {server.cwd} (defaults to source directory when omitted)")
                lines.append(f"    command: [redacted]; args: {len(server.args)} values [redacted]")
                lines.append(
                    f"    env: {len(server.env or {})} values [redacted]; envFrom: {len(server.env_from)} references"
                )
                lines.append(
                    "    Inherited environment: PATH, HOME, USER, LOGNAME, LANG, LC_ALL, LC_CTYPE, TMPDIR, TMP, TEMP"
                )
            else:
                url = urlsplit(server.url or "")
                lines.append(f"    Endpoint origin: {url.scheme}://{url.netloc}; path/query: [redacted]")
                lines.append(
                    f"    headers: {len(server.headers)} values [redacted]; "
                    f"headersFrom: {len(server.headers_from)} references"
                )
            lines.append(f"    Timeout: {server.timeout:g}s (default: 30s)")
            lines.append(f"    Protocol: {server.protocol_version or 'negotiate (default)'}")
            try:
                server.resolve()
            except ValueError as error:
                lines.append(f"    Resolution error: {error}")
            else:
                lines.append("    Connection parameters resolve locally.")
    return "\n".join(lines)
