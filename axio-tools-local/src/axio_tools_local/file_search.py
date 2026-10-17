"""Search project files, with a shared candidate enumeration for preflight checks."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Iterator
from pathlib import Path

from axio.field import StrictStr
from axio.tool_access import Access, tool_access


@tool_access(
    lambda arguments: (
        Access("read", str(arguments.get("path", "."))),
        *(Access("read", str(path)) for path in candidate_files(Path(arguments.get("path", ".")).resolve())),
    )
)
async def search_files(
    query: StrictStr,
    path: StrictStr = ".",
    regex: bool = False,
    max_results: int = 100,
) -> str:
    """Search for text or regex patterns in files under a directory.
    Returns matching lines with file paths and line numbers."""

    def _search() -> str:
        base = Path(path).resolve()
        if not base.exists():
            return f"error: path not found: {path}"

        try:
            pattern = re.compile(query) if regex else None
        except re.error as exc:
            return f"error: invalid regex: {exc}"

        matches: list[str] = []

        files = candidate_files(base)
        for file_path in files:
            if len(matches) >= max_results:
                break
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                found = pattern.search(line) if pattern else (query in line)
                if found:
                    matches.append(f"{file_path}:{idx}: {line}")
                    if len(matches) >= max_results:
                        break

        if not matches:
            return f"No matches for {query!r}"
        return "\n".join(matches)

    return await asyncio.to_thread(_search)


def _iter_files(base: Path, skip: set[str]) -> Iterator[Path]:
    for current_dir, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in sorted(files):
            if not name.startswith("."):
                yield Path(current_dir) / name


def candidate_files(base: Path) -> list[Path]:
    """Enumerate the files search_files can read, without reading their contents."""
    skip = {".git", ".venv", "__pycache__", "node_modules"}
    return [base] if base.is_file() else list(_iter_files(base, skip))
