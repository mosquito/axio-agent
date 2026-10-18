"""Explicit paid STT -> selected text provider -> paid TTS, using an input WAV file.

Run from this directory with ``uv run python speech_pipeline.py --help``.
No microphone, PortAudio, subscription audio endpoints, or automatic billing fallback.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
from pathlib import Path

import aiohttp
from axio import Agent, MemoryContextStore
from axio.transport import CompletionTransport
from axio_transport_openai import ChatGPTTransport, OpenAISTTTransport, OpenAITransport, OpenAITTSTransport


async def respond(audio: bytes, text_transport: CompletionTransport, http: aiohttp.ClientSession) -> bytes:
    transcript = await OpenAISTTTransport(session=http).transcribe(audio)
    answer = await Agent(system="Answer briefly. Your response will be spoken.", transport=text_transport).run(
        transcript, MemoryContextStore()
    )
    async with aclosing(OpenAITTSTransport(session=http).synthesize(answer)) as speech:
        return b"".join([chunk async for chunk in speech])


async def main(args: argparse.Namespace) -> None:
    async with aiohttp.ClientSession() as http:
        audio = Path(args.input).read_bytes()
        if args.text_provider == "chatgpt":
            # Existing native OAuth/account handling only; credentials never enter API audio transports.
            async with ChatGPTTransport.open_for_app(session=http, account=args.account) as text:
                output = await respond(audio, text, http)
        else:
            output = await respond(audio, OpenAITransport(session=http), http)
        Path(args.output).write_bytes(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Encoded WAV input, at most 25 MB")
    parser.add_argument("output", help="Output MP3 path")
    parser.add_argument("--text-provider", choices=("openai", "chatgpt"), required=True)
    parser.add_argument("--account", help="Existing ChatGPT registration; use axio-chatgpt login first")
    parser.add_argument(
        "--allow-paid-audio",
        action="store_true",
        required=True,
        help="Explicitly authorize API-billed STT/TTS using OPENAI_API_KEY",
    )
    asyncio.run(main(parser.parse_args()))
