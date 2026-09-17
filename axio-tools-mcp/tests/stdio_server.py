"""MCP server over stdio, spawned by the session tests."""

from __future__ import annotations

import asyncio
import os
import sys

from aiohttp_tiny_mcp import Registry, run_stdio
from pydantic import BaseModel


class Nothing(BaseModel):
    pass


class Message(BaseModel):
    message: str


registry = Registry("fixture", "1.0")


@registry.tool
async def echo(args: Message) -> str:
    """Return the message."""
    return args.message


@registry.tool
async def token(args: Nothing) -> str:
    """Return the AXIO_MCP_TEST_TOKEN value of the server environment."""
    return os.environ.get("AXIO_MCP_TEST_TOKEN", "")


@registry.tool
async def explode(args: Nothing) -> str:
    """Always fail."""
    raise RuntimeError("tool exploded")


if __name__ == "__main__":
    # The session forwards this line to the logger, which the tests check.
    print("fixture server ready", file=sys.stderr, flush=True)
    asyncio.run(run_stdio(registry))
