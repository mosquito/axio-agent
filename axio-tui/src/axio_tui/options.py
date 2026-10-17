"""Application startup options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_BYTES = 128 * 1024


@dataclass(frozen=True, slots=True)
class StartupOptions:
    isolated: bool = False
    instructions_files: tuple[Path, ...] = ()
    mcp_files: tuple[Path, ...] = ()
    skill_directories: tuple[Path, ...] = ()
    instructions_max_bytes: int = DEFAULT_MAX_BYTES
