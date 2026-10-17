"""Resolve project boundaries without changing the process working directory."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def canonical_path(path: Path, base: Path) -> Path:
    """Resolve a path relative to an explicit base, including existing symlinks."""
    try:
        return (base / path).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot resolve {path}: {error}") from error


def config_directory() -> Path:
    """Return the Axio configuration directory; ignore a relative XDG_CONFIG_HOME."""
    configured = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(configured) if configured else Path.home() / ".config"
    if not base.is_absolute():
        base = Path.home() / ".config"
    return canonical_path(base / "axio", Path.cwd())


def project_directory(path: Path, base: Path) -> Path:
    result = canonical_path(path, base)
    if not result.is_dir():
        raise ValueError(f"Workspace root is not an existing directory: {result}")
    return result


def discover_root(cwd: Path) -> Path:
    """Find the nearest Git marker, stopping before an implicit home or filesystem root."""
    home = Path.home().resolve()
    for directory in (cwd, *cwd.parents):
        if directory == home or directory == directory.parent:
            break
        marker = directory / ".git"
        if marker.is_file() or marker.is_dir():
            return directory
    return cwd


@dataclass(frozen=True, slots=True)
class LocalPaths:
    cwd: Path
    roots: tuple[Path, ...]

    @classmethod
    def discover(
        cls,
        cwd: Path,
        project_root: Path | None = None,
        additional_roots: Iterable[Path] = (),
    ) -> LocalPaths:
        current = project_directory(cwd, Path.cwd())
        primary = discover_root(current) if project_root is None else project_directory(project_root, current)
        if not current.is_relative_to(primary):
            raise ValueError(f"Working directory {current} is outside project root {primary}")
        roots = [primary]
        for path in additional_roots:
            root = project_directory(path, current)
            if root in roots:
                continue
            for other in roots:
                if root.is_relative_to(other) or other.is_relative_to(root):
                    raise ValueError(f"Workspace roots must not overlap: {other} and {root}")
            roots.append(root)
        return cls(cwd=current, roots=tuple(roots))

    @property
    def project_root(self) -> Path:
        return self.roots[0]

    def resolve(self, path: Path) -> Path:
        return canonical_path(path, self.cwd)

    def root_for(self, path: Path) -> Path | None:
        target = self.resolve(path)
        return next((root for root in self.roots if target.is_relative_to(root)), None)

    def directories_for(self, path: Path) -> tuple[Path, ...]:
        """Return root-to-parent directories for a file, or root-to-directory for a directory."""
        target = self.resolve(path)
        root = self.root_for(target)
        if root is None:
            return ()
        directory = target if target.is_dir() else target.parent
        if not directory.is_relative_to(root):
            return ()
        parts = directory.relative_to(root).parts
        return (root, *(root.joinpath(*parts[:end]) for end in range(1, len(parts) + 1)))

    def describe(self) -> str:
        return "\n".join([f"Working directory: {self.cwd}", *(f"Workspace root: {root}" for root in self.roots)])
