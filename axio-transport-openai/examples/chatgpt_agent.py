"""Run an agent with a saved ChatGPT registration and a local arithmetic tool."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
from axio import Agent, MemoryContextStore, Tool
from axio_responses import ResponsesAPIError

from axio_transport_openai.accounts import open_chatgpt
from axio_transport_openai.cli import error_message
from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.credentials import CredentialError
from axio_transport_openai.oauth import OAuthError


async def multiply(left: int, right: int) -> str:
    """Multiply two integers and return the product."""
    return str(left * right)


async def run(args: argparse.Namespace) -> str:
    store = FileCredentialStore(args.credentials)
    async with aiohttp.ClientSession() as http:
        async with open_chatgpt(store, http, client_id=args.account, model=args.model) as transport:
            agent = Agent(
                system="Use multiply for calculations. Return a concise answer.",
                transport=transport,
                tools=[Tool(name="multiply", handler=multiply)],
                max_iterations=4,
            )
            return await agent.run(args.prompt, MemoryContextStore())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Multiply 17 by 19 using the multiply tool.")
    parser.add_argument("--account", help="Override the saved account selection.")
    parser.add_argument("--model", default="gpt-6-luna", help="A visible ChatGPT model.")
    parser.add_argument("--credentials", type=Path)
    args = parser.parse_args()
    try:
        print(asyncio.run(run(args)))
    except (OAuthError, CredentialError, ResponsesAPIError, aiohttp.ClientError, OSError) as error:
        print(error_message(error), file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
