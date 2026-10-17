"""Read and atomically edit one MCP configuration without losing other entries."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .configuration import Layer, parse_config
from .paths import config_directory


@dataclass(frozen=True, slots=True)
class MCPDocument:
    path: Path
    layer: Layer
    content: bytes | None = field(repr=False)
    servers: dict[str, Any] = field(repr=False)

    @classmethod
    def read(cls, path: Path, layer: Layer) -> MCPDocument:
        if path.is_symlink():
            raise ValueError("Cannot edit a symlink configuration. Edit its target explicitly.")
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return cls(path, layer, None, {})
        parse_config(content.decode("utf-8"), path, layer)
        return cls(path, layer, content, json.loads(content)["mcpServers"])

    def write(self, name: str, entry: dict[str, Any] | None, *, original: str | None = None) -> None:
        """Reject stale edits and duplicate names; commit a private, complete JSON file."""
        servers = dict(self.servers)
        if entry is not None and name != original and name in servers:
            raise ValueError("A server with this name already exists in this file.")
        if original is not None:
            servers.pop(original, None)
        if entry is not None:
            servers[name] = entry
        content = json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False) + "\n"
        parse_config(content, self.path, self.layer)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        locks = config_directory() / "locks"
        locks.mkdir(parents=True, exist_ok=True, mode=0o700)
        identity = hashlib.sha256(str(self.path.absolute()).encode()).hexdigest()
        fd = os.open(locks / f"mcp-edit-{identity}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = MCPDocument.read(self.path, self.layer)
            if current.content != self.content:
                raise ValueError("Configuration changed while editing. Reopen it and try again.")
            fd, temporary = tempfile.mkstemp(prefix=".mcp-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
