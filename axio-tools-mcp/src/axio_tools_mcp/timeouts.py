"""Bound MCP operations and provide diagnostics without connection secrets."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class MCPTimeoutError(TimeoutError):
    """A timeout message that is safe to show in a frontend."""


@asynccontextmanager
async def mcp_timeout(seconds: float, operation: str) -> AsyncIterator[None]:
    """Apply one deadline, including all requests made by the operation."""
    try:
        async with asyncio.timeout(seconds):
            yield
    except MCPTimeoutError:
        raise
    except TimeoutError:
        raise MCPTimeoutError(f"{operation} timed out after {seconds:g}s (timeout setting)") from None


def error_summary(error: BaseException) -> str:
    """Keep raw exceptions, which can contain credentials, out of diagnostics."""
    return str(error) if isinstance(error, MCPTimeoutError) else type(error).__name__
