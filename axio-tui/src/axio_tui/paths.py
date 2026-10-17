"""Select application configuration roots without requiring a tool provider."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ConfigurationPaths(Protocol):
    @property
    def cwd(self) -> Path: ...

    @property
    def roots(self) -> tuple[Path, ...]: ...

    @property
    def project_root(self) -> Path: ...

    def describe(self) -> str: ...

    def resolve(self, path: Path) -> Path: ...


def canonical_path(path: Path, base: Path) -> Path:
    return (base / path).resolve()


def config_directory() -> Path:
    configured = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    if not configured.is_absolute():
        configured = Path.home() / ".config"
    return (configured / "axio").resolve()


@dataclass(frozen=True, slots=True)
class ConfigPaths:
    cwd: Path
    roots: tuple[Path, ...]

    @property
    def project_root(self) -> Path:
        return self.roots[0]

    def resolve(self, path: Path) -> Path:
        return canonical_path(path, self.cwd)

    def describe(self) -> str:
        return "\n".join([f"Working directory: {self.cwd}", *(f"Workspace root: {root}" for root in self.roots)])

    @classmethod
    def discover(
        cls, cwd: Path, project_root: Path | None = None, additional_roots: Iterable[Path] = ()
    ) -> ConfigPaths:
        cwd = cwd.resolve()
        primary = cwd
        if project_root is None:
            for directory in (cwd, *cwd.parents):
                if directory == Path.home().resolve() or directory == directory.parent:
                    break
                if (directory / ".git").exists():
                    primary = directory
                    break
        else:
            primary = canonical_path(project_root, cwd)
        if not cwd.is_relative_to(primary):
            raise ValueError(f"Working directory {cwd} is outside project root {primary}")
        roots = [primary]
        for path in additional_roots:
            root = canonical_path(path, cwd)
            if root in roots:
                continue
            if any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots):
                raise ValueError(f"Workspace roots must not overlap: {root}")
            roots.append(root)
        for root in roots:
            if not root.is_dir():
                raise ValueError(f"Workspace root is not an existing directory: {root}")
        return cls(cwd, tuple(roots))
