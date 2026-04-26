import asyncio
import os
from typing import Any

from axio.tool import ToolHandler
from pydantic import StrictStr


class ReadFile(ToolHandler[Any]):
    """Read file contents. Returns text for text files, hex for binaries.
    Lines are 1-indexed: start_line=1 is the first line, end_line=3 includes
    line 3. Pass line_numbers=True to prefix each line with its 1-based line
    number (tab-separated) — required before calling patch_file. Large files
    are truncated to max_chars. Always read the file before editing it with
    write_file or patch_file."""

    filename: StrictStr
    max_chars: int = 32768
    binary_as_hex: bool = True
    start_line: int | None = None
    end_line: int | None = None
    line_numbers: bool = False

    def __repr__(self) -> str:
        return f"ReadFile(filename={self.filename!r})"

    def _blocking(self) -> str:
        path = os.path.join(os.getcwd(), self.filename)
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode()
        except UnicodeDecodeError:
            if self.binary_as_hex:
                return "Encoded binary data HEX: " + raw[: self.max_chars].hex()
            raise
        all_lines = text.splitlines(keepends=True)
        start = 0 if self.start_line is None else self.start_line - 1
        end = len(all_lines) if self.end_line is None else self.end_line
        lines = all_lines[start:end]
        if self.line_numbers:
            result = "".join(f"{start + 1 + i}\t{line}" for i, line in enumerate(lines))
        else:
            result = "".join(lines)
        if len(result) > self.max_chars:
            return result[: self.max_chars] + "\n...[truncated]"
        return result

    async def __call__(self, context: Any) -> str:
        return await asyncio.to_thread(self._blocking)
