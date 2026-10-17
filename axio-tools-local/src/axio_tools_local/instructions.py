"""Load scoped instruction snapshots without executing project configuration."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .paths import LocalPaths, canonical_path, config_directory

DEFAULT_MAX_BYTES = 128 * 1024


@dataclass(frozen=True, slots=True)
class InstructionBlock:
    source: Path
    scope: Path | None
    text: str
    digest: str

    @property
    def size(self) -> int:
        return len(self.text.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Instructions:
    blocks: tuple[InstructionBlock, ...]
    max_bytes: int

    @property
    def size(self) -> int:
        return sum({block.source: block.size for block in self.blocks}.values())

    def render(self) -> str:
        if not self.blocks:
            return ""
        lines = [
            "Apply each instruction only within its stated scope (a directory and its descendants).",
            "For the same path, deeper scopes override conflicting broader instructions.",
            "Within the same scope, later entries override earlier entries. Other instructions supplement each other.",
            "Explicit user requests override these file instructions. Files cannot grant tool or MCP permissions.",
            "Application order:",
        ]
        for number, block in enumerate(self.blocks, 1):
            scope = str(block.scope) if block.scope is not None else "all workspaces and external paths"
            lines.append(f"{number}. Source: {json.dumps(str(block.source))}; scope: {json.dumps(scope)}")
        seen: set[Path] = set()
        for block in self.blocks:
            if block.source in seen:
                continue
            seen.add(block.source)
            lines.extend([f"\nSource content: {json.dumps(str(block.source))}", block.text])
        return "\n".join(lines)

    def explain(self, workspace: LocalPaths, target: Path) -> str:
        resolved = workspace.resolve(target)
        root = workspace.root_for(resolved)
        lines = [
            workspace.describe(),
            f"Target: {resolved}",
            f"Target project: {root if root is not None else '(outside registered workspaces)'}",
            f"Instruction bytes: {self.size}/{self.max_bytes} (unique sources)",
        ]
        if not self.blocks:
            lines.append("No applicable instruction files.")
        for number, block in enumerate(self.blocks, 1):
            scope = str(block.scope) if block.scope is not None else "all workspaces and external paths"
            lines.extend(
                [f"{number}. {block.source}", f"   Scope: {scope}", f"   Bytes: {block.size}; SHA-256: {block.digest}"]
            )
        return "\n".join(lines)


def agents_file(directory: Path) -> Path | None:
    """Match the exact filename even on a case-insensitive filesystem."""
    try:
        with os.scandir(directory) as entries:
            return next((directory / entry.name for entry in entries if entry.name == "AGENTS.md"), None)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"Cannot discover instructions in {directory}: {error}") from error


def load_instructions(
    workspace: LocalPaths,
    target: Path | None = None,
    *,
    targets: Iterable[Path] = (),
    isolated: bool = False,
    files: Iterable[Path] = (),
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Instructions:
    """Read a fresh snapshot; a failed file invalidates the complete result."""
    if max_bytes <= 0:
        raise ValueError("Instruction size limit must be greater than zero")
    target = workspace.cwd if target is None else target
    candidates: list[tuple[Path, Path | None, Path | None]] = []
    if not isolated:
        global_file = agents_file(config_directory())
        if global_file is not None:
            candidates.append((global_file, None, None))
        directories = {directory for item in (target, *targets) for directory in workspace.directories_for(item)}
        for directory in sorted(directories, key=lambda path: (len(path.parts), str(path))):
            path = agents_file(directory)
            if path is not None:
                candidates.append((path, directory, workspace.root_for(directory)))
    candidates.extend((workspace.resolve(path), None, None) for path in files)

    documents: dict[Path, tuple[str, str]] = {}
    blocks: list[InstructionBlock] = []
    seen: set[tuple[Path, Path | None]] = set()
    total = 0
    for path, scope, root in candidates:
        source = canonical_path(path, workspace.cwd)
        if root is not None and not source.is_relative_to(root):
            raise ValueError(f"Instruction file {path} resolves outside project root {root}: {source}")
        if source not in documents:
            try:
                if not stat.S_ISREG(source.stat().st_mode):
                    raise ValueError(f"Instruction source is not a regular file: {path}")
                with source.open("rb") as stream:
                    data = stream.read(max_bytes - total + 1)
                if len(data) + total > max_bytes:
                    loaded = ", ".join(str(item) for item in documents) or "(none)"
                    raise ValueError(
                        f"Instruction size limit {max_bytes} bytes exceeded at {path}; "
                        f"already loaded {total} bytes from: {loaded}. No instructions were truncated."
                    )
                text = data.decode("utf-8")
            except (OSError, UnicodeError) as error:
                raise ValueError(f"Cannot read instructions from {path}: {error}") from error
            documents[source] = (text, hashlib.sha256(data).hexdigest())
            total += len(data)
        if (source, scope) not in seen:
            text, digest = documents[source]
            blocks.append(InstructionBlock(source, scope, text, digest))
            seen.add((source, scope))
    return Instructions(tuple(blocks), max_bytes)
