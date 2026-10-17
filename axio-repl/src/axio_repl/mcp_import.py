"""Explicit, offline copying of supported Claude Code and Cursor MCP entries.

Source contracts: https://code.claude.com/docs/en/mcp and
https://cursor.com/docs/mcp. Unknown source features are errors, not guesses.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from axio_tools_mcp.import_formats import Client, Destination, Scope, convert, source_entries

from .configuration import JSONPairs, fail, parse_server, unique_objects
from .paths import ConfigPaths, ConfigurationPaths, config_directory


def read_document(path: Path, *, optional: bool = False) -> tuple[dict[str, Any], bytes | None]:
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise fail(path, None, "expected a regular file")
        content = path.read_bytes()
    except FileNotFoundError:
        if optional and not path.is_symlink():
            return {"mcpServers": {}}, None
        raise fail(path, None, "file does not exist") from None
    except OSError:
        raise fail(path, None, "cannot read file") from None
    try:
        data = unique_objects(json.loads(content.decode("utf-8"), object_pairs_hook=JSONPairs), path)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise fail(path, None, "invalid UTF-8 JSON") from None
    if not isinstance(data, dict):
        raise fail(path, None, "expected a JSON object")
    return data, content


def source_file(client: Client, scope: Scope, root: Path, home: Path) -> Path:
    if client == "claude":
        return root / ".mcp.json" if scope == "project" else home / ".claude.json"
    if scope == "local":
        raise ValueError("Cursor has user and project scopes; local is a Claude Code scope")
    return (home if scope == "user" else root) / ".cursor/mcp.json"


@dataclass(frozen=True, slots=True)
class ImportPlan:
    source: Path
    destination: Path
    client: Client
    scope: Scope
    target: Destination
    imported: tuple[str, ...]
    conflicts: tuple[str, ...]
    data: dict[str, Any] = field(repr=False)
    previous: bytes | None = field(repr=False)

    def describe(self) -> str:
        lines = [
            f"Source: {self.source} ({self.client}, {self.scope})",
            f"Destination: {self.destination} ({self.target})",
            "Source files remain unchanged. No servers are started.",
            "Server values are redacted; inspect the source before writing.",
        ]
        for name in self.imported:
            entry = self.data["mcpServers"][name]
            action = "replace" if name in self.conflicts else "add"
            lines.append(f"  {json.dumps(name)}: {action}; fields: {', '.join(sorted(entry))}")
            if "cwd" in entry:
                lines.append(f"    cwd: {entry['cwd']}")
        return "\n".join(lines)

    def write(self, *, replace: bool = False) -> None:
        if self.conflicts and not replace:
            raise ValueError("Import would replace existing servers; review the preview and add --replace")
        self.destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        locks = config_directory() / "locks"
        locks.mkdir(parents=True, exist_ok=True, mode=0o700)
        identity = hashlib.sha256(str(self.destination.resolve()).encode()).hexdigest()
        lock_path = locks / f"mcp-import-{identity}.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _, current = read_document(self.destination, optional=True)
            if current != self.previous or self.destination.is_symlink():
                raise ValueError("Import destination changed after preview; run the import again")
            fd, temporary = tempfile.mkstemp(dir=self.destination.parent, prefix=".mcp-import-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(self.data, stream, ensure_ascii=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


def plan_import(
    workspace: ConfigurationPaths,
    client: Client,
    scope: Scope,
    target: Destination,
    *,
    names: tuple[str, ...] = (),
    server_cwd: Path | None = None,
    home: Path | None = None,
) -> ImportPlan:
    root = workspace.project_root
    source = source_file(client, scope, root, Path.home() if home is None else home)
    data, _ = read_document(source)
    entries = source_entries(data, client, scope, root, source)
    selected = names or tuple(entries)
    if not selected:
        raise fail(source, None, "no MCP servers in the selected scope")
    cwd = workspace.resolve(server_cwd) if server_cwd is not None else None
    imported: dict[str, Any] = {}
    for name in selected:
        if name not in entries:
            raise fail(source, name, "not found in the selected scope")
        imported[name] = convert(name, entries[name], client, source, cwd)
        entry = imported[name]
        if target == "project" and (entry.get("env") or entry.get("headers") or urlsplit(entry.get("url", "")).query):
            raise fail(source, name, "literal env, headers or URL queries require --target private or user")
    destination = {
        "user": config_directory() / "mcp.json",
        "project": root / ".mcp.json",
        "private": root / ".axio/mcp.local.json",
    }[target]
    if source.resolve() == destination.resolve() or destination.is_symlink():
        raise ValueError("Import destination must differ from the source and must not be a symlink")
    existing, previous = read_document(destination, optional=True)
    if set(existing) != {"mcpServers"} or not isinstance(existing["mcpServers"], dict):
        raise fail(destination, None, "expected an object with one mcpServers object")
    for name, entry in existing["mcpServers"].items():
        parse_server(name, entry, destination, "explicit")
    conflicts = tuple(name for name in imported if name in existing["mcpServers"])
    merged = {"mcpServers": existing["mcpServers"] | imported}
    return ImportPlan(source, destination, client, scope, target, tuple(imported), conflicts, merged, previous)


def import_main(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="axio-repl mcp import", description="Preview or copy MCP configurations")
    parser.add_argument("--from", dest="client", required=True, choices=("claude", "cursor"))
    parser.add_argument("--scope", required=True, choices=("user", "project", "local"), help="Source client scope")
    parser.add_argument(
        "--target", required=True, choices=("user", "project", "private"), help="REPL destination scope"
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--server", action="append", default=[], help="Copy only this named server; repeatable")
    parser.add_argument("--server-cwd", type=Path, help="Required for stdio entries without an explicit cwd")
    parser.add_argument("--write", action="store_true", help="Write the previewed configuration")
    parser.add_argument("--replace", action="store_true", help="Allow whole-entry replacement of conflicting names")
    args = parser.parse_args(arguments)
    try:
        workspace = ConfigPaths.discover(Path.cwd(), args.project_root)
        plan = plan_import(
            workspace, args.client, args.scope, args.target, names=tuple(args.server), server_cwd=args.server_cwd
        )
        print(plan.describe())
        if args.write:
            plan.write(replace=args.replace)
            print(f"Copied {len(plan.imported)} MCP server(s) to {plan.destination}")
        else:
            print("Preview only. Add --write to copy; add --replace to replace conflicting server names.")
    except (OSError, ValueError) as error:
        parser.error(str(error))
