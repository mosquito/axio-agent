from __future__ import annotations

import asyncio
import os
import stat as stat_module
from datetime import datetime
from pathlib import Path

from axio.field import StrictStr
from axio.tool_access import file_access, tool_access


@tool_access(file_access("read", "directory", default="."))
async def list_files(directory: StrictStr = ".") -> str:
    """List files and directories. Shows permissions, size, modification time,
    and name for each entry. Directories are listed first and marked with
    a trailing slash. Use this to explore the project structure before
    reading or editing files."""

    def _blocking() -> str:
        path = Path(os.getcwd()) / directory
        if not path.is_dir():
            raise FileNotFoundError(f"{directory} is not a valid directory")
        lines: list[str] = []
        # Listing a symlink describes the link; it must not stat its target.
        for fpath in sorted(path.glob("*"), key=lambda p: (not p.is_dir() if not p.is_symlink() else True, p.name)):
            try:
                st = fpath.lstat()
            except OSError:
                lines.append(f"{'?':10} {'?':>8} {'?':12} {fpath.name} [broken symlink]")
                continue
            mode = stat_module.filemode(st.st_mode)
            size = st.st_size
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%b %d %H:%M")
            name = fpath.name + ("/" if stat_module.S_ISDIR(st.st_mode) else "")
            lines.append(f"{mode} {size:>8} {mtime} {name}")
        return "\n".join(lines) or "(empty directory)"

    return await asyncio.to_thread(_blocking)
