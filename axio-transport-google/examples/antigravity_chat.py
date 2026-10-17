"""Run a Gemini tool loop with a saved Antigravity account.

First: uv run axio-antigravity login
Then: uv run --directory axio-transport-google python examples/antigravity_chat.py
"""

from __future__ import annotations

import asyncio

import aiohttp
from axio import Agent, MemoryContextStore, Tool

from axio_transport_google.antigravity import AntigravityTransport


async def add_one(value: int) -> str:
    """Return the number plus one."""
    return str(value + 1)


async def main() -> None:
    async with aiohttp.ClientSession() as http:
        transport = AntigravityTransport(session=http)
        await transport.fetch_models()
        print(f"Selected model: {transport.model.id}")
        agent = Agent(
            system="Use the requested tool and explain its result.",
            transport=transport,
            tools=[Tool(name="add_one", handler=add_one)],
        )
        context = MemoryContextStore()
        print(await agent.run("Use add_one for 3 and for 7.", context))
        print(await agent.run("What were the two results?", context))


if __name__ == "__main__":
    asyncio.run(main())
