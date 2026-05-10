"""Realtime smoke test: tool calls + voice interruption against a real provider.

Two scenarios:

* ``tool``     — sends a question that should make the model call a tool;
                 dispatches the tool, captures the spoken response, exits.
* ``interrupt``— sends a question that triggers a long answer; once the model
                 starts streaming audio, fires ``interrupt()`` and verifies
                 the audio stream stops within a short grace period.

Both scenarios save the captured PCM16 mono audio (24 kHz) to a file so you
can verify audibly with ``aplay -f S16_LE -r 24000 -c 1 out.pcm`` (Linux) or
similar.

Usage::

    OPENAI_API_KEY=...  uv run python smoke.py --provider openai --scenario tool
    GEMINI_API_KEY=...  uv run python smoke.py --provider gemini --scenario interrupt

The exit status is 0 when the scenario's expectations are met; non-zero
otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from axio import (
    AudioOutputDelta,
    RealtimeAgent,
    RealtimeTransport,
    SpeechStarted,
    SpeechStopped,
    TextBlock,
    TextDelta,
    Tool,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from axio.blocks import AudioBlock
from axio.events import Error
from axio.types import StopReason

WEATHER_DATA = {
    "tokyo": ("13C", "rain"),
    "paris": ("8C", "cloudy"),
    "san francisco": ("18C", "fog"),
}


async def _tts_user_speech(text: str, *, voice: str = "echo") -> bytes:
    """Synthesize ``text`` as 24 kHz PCM16 mono via OpenAI's TTS endpoint.

    Real synthesized speech reliably trips the realtime server VAD, where
    procedurally-generated noise / tone bursts do not (VAD looks at the
    speech-band envelope, not raw amplitude).
    """
    import aiohttp

    api_key = os.environ["OPENAI_API_KEY"]
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "tts-1",
                "voice": voice,
                "input": text,
                "response_format": "pcm",  # raw PCM16 mono @ 24 kHz
            },
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            return await resp.read()


async def get_weather(city: str) -> str:
    """Return the current weather for a city (canned data)."""
    info = WEATHER_DATA.get(city.lower())
    if not info:
        return f"No data for {city}"
    temp, condition = info
    return f"{city}: {temp}, {condition}"


@dataclass
class Result:
    scenario: str
    provider: str
    audio_bytes: int = 0
    audio_chunks: int = 0
    tool_calls: list[str] = field(default_factory=list)
    transcript_assistant: str = ""
    transcript_user: str = ""
    text_chunks: int = 0
    turns: int = 0
    interrupted_at: float | None = None
    silence_after_interrupt_ms: float | None = None
    speech_started_after_interrupt_ms: float | None = None
    expectations: dict[str, bool] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pcm_buffer: bytearray = field(default_factory=bytearray, repr=False)

    def passed(self) -> bool:
        return all(self.expectations.values())


def make_transport(provider: str) -> RealtimeTransport:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        from axio_transport_openai import OpenAIRealtimeTransport

        return OpenAIRealtimeTransport()
    if provider == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        from axio_transport_google.realtime import GeminiLiveTransport

        return GeminiLiveTransport()
    raise ValueError(f"Unknown provider: {provider}")


async def _drain_until(
    agent: RealtimeAgent,
    result: Result,
    *,
    stop_after_turns: int = 1,
    interrupt_after_audio_ms: float | None = None,
    max_seconds: float = 30.0,
    silence_window_seconds: float = 1.5,
) -> None:
    """Consume events from the agent and update ``result``.

    Stops when one of:
      * ``stop_after_turns`` TurnComplete events have been seen,
      * ``max_seconds`` of wall time elapsed,
      * an interrupt was requested AND ``silence_window_seconds`` have passed
        without any new AudioOutputDelta.

    If ``interrupt_after_audio_ms`` is set, fires ``interrupt()`` once that many
    milliseconds of audio (assuming 24 kHz mono int16, i.e. 48 bytes/ms) have
    been received and records ``silence_after_interrupt_ms`` — the time from
    interrupt fire to the last audio chunk that still leaked through.
    """
    started = time.monotonic()
    last_audio_at: float | None = None
    interrupt_fired_at: float | None = None
    pcm_per_ms = 24 * 2

    events_iter = agent.events().__aiter__()
    while True:
        deadline = max_seconds - (time.monotonic() - started)
        if deadline <= 0:
            result.timeline.append({"t": time.monotonic() - started, "label": "max_seconds_reached"})
            return
        # Once interrupted, bound each wait by the silence window so a quiet
        # WebSocket can be detected as "interrupt was honoured".
        per_step_timeout = silence_window_seconds if interrupt_fired_at is not None else deadline
        try:
            event = await asyncio.wait_for(events_iter.__anext__(), timeout=per_step_timeout)
        except StopAsyncIteration:
            return
        except TimeoutError:
            if interrupt_fired_at is not None and last_audio_at is not None:
                last_leak = max(last_audio_at - interrupt_fired_at, 0.0)
                result.silence_after_interrupt_ms = last_leak * 1000
                result.timeline.append({"t": time.monotonic() - started, "label": "post_interrupt_silence_confirmed"})
            else:
                result.timeline.append({"t": time.monotonic() - started, "label": "no_events_timeout"})
            return

        now = time.monotonic()
        elapsed = now - started

        if isinstance(event, Error):
            result.errors.append(str(event.exception))
            result.timeline.append({"t": elapsed, "label": "error", "detail": str(event.exception)})
            return
        if isinstance(event, AudioOutputDelta):
            result.audio_chunks += 1
            result.audio_bytes += len(event.data)
            result.pcm_buffer.extend(event.data)
            last_audio_at = now
            if result.audio_chunks == 1:
                result.timeline.append({"t": elapsed, "label": "first_audio"})
            if (
                interrupt_after_audio_ms is not None
                and interrupt_fired_at is None
                and result.audio_bytes >= interrupt_after_audio_ms * pcm_per_ms
            ):
                interrupt_fired_at = now
                result.interrupted_at = elapsed
                result.timeline.append({"t": elapsed, "label": "fire_interrupt"})
                await agent.interrupt()
        elif isinstance(event, ToolUseStart):
            result.tool_calls.append(event.name)
            result.timeline.append({"t": elapsed, "label": "tool_call", "name": event.name, "id": event.tool_use_id})
        elif isinstance(event, TranscriptDelta):
            if event.role == "assistant":
                result.transcript_assistant += event.delta
            else:
                result.transcript_user += event.delta
        elif isinstance(event, TextDelta):
            result.text_chunks += 1
        elif isinstance(event, SpeechStarted):
            result.timeline.append({"t": elapsed, "label": "speech_started"})
        elif isinstance(event, SpeechStopped):
            result.timeline.append({"t": elapsed, "label": "speech_stopped"})
        elif isinstance(event, TurnComplete):
            result.turns += 1
            result.timeline.append({"t": elapsed, "label": "turn_complete", "stop_reason": event.stop_reason.name})
            if result.turns >= stop_after_turns:
                return
            if event.stop_reason is StopReason.tool_use:
                continue


async def scenario_tool(transport: RealtimeTransport, provider: str) -> Result:
    """Asks for weather; expects a tool call followed by an audio answer."""
    result = Result(scenario="tool", provider=provider)
    async with RealtimeAgent(
        system=(
            "You are a weather assistant. Always call the get_weather function for "
            "any weather question; do not invent answers. Reply concisely."
        ),
        transport=transport,
        tools=[Tool(name="get_weather", handler=get_weather, description="Get current weather for a city")],
        voice="marin" if provider == "openai" else None,
    ) as agent:
        await agent.send(TextBlock(text="What's the weather in Tokyo right now?"))
        await agent.commit()
        await _drain_until(agent, result, stop_after_turns=2, max_seconds=30.0)

    result.expectations = {
        "tool_was_called": "get_weather" in result.tool_calls,
        "audio_received": result.audio_bytes > 0,
        "two_turns_observed": result.turns >= 2,
    }
    return result


async def scenario_cancel(transport: RealtimeTransport, provider: str) -> Result:
    """Triggers a long answer, calls ``agent.interrupt()`` mid-stream, verifies
    the audio actually stops within ~1.5 s.  This exercises the **programmatic**
    cancel path (``response.cancel`` to the server)."""
    result = Result(scenario="cancel", provider=provider)
    async with RealtimeAgent(
        system="You are a verbose poet. Compose a long, detailed monologue when asked.",
        transport=transport,
        voice="marin" if provider == "openai" else None,
    ) as agent:
        await agent.send(
            TextBlock(text="Tell me a long story about a samurai cat exploring the cosmos. Take your time.")
        )
        await agent.commit()
        await _drain_until(
            agent,
            result,
            stop_after_turns=5,
            interrupt_after_audio_ms=1500.0,
            max_seconds=20.0,
        )

    result.expectations = {
        "audio_received_before_interrupt": result.audio_bytes > 0,
        "interrupt_fired": result.interrupted_at is not None,
        "silence_within_1500ms": (
            result.silence_after_interrupt_ms is None or result.silence_after_interrupt_ms < 1500
        ),
    }
    return result


async def scenario_voice_interrupt(transport: RealtimeTransport, provider: str) -> Result:
    """Triggers a long answer, then injects fake user speech (audio) mid-stream
    so the **server VAD** sees a barge-in.  Validates that:

      * ``SpeechStarted`` is emitted (server VAD detected the 'user'),
      * audio output stops within ~1.5 s (turn_detection.interrupt_response
        actually cancelled the response).
    """
    result = Result(scenario="voice_interrupt", provider=provider)
    async with RealtimeAgent(
        system="You are a verbose poet. Compose a long, detailed monologue when asked.",
        transport=transport,
        voice="marin" if provider == "openai" else None,
    ) as agent:
        await agent.send(
            TextBlock(text="Tell me a long story about a samurai cat exploring the cosmos. Take your time.")
        )
        await agent.commit()

        # ONE iterator drains the session for the whole scenario — calling
        # agent.events() twice would leave two consumers racing for the same
        # underlying WebSocket and split the event stream unpredictably.
        events_iter = agent.events().__aiter__()

        await _drain_until_first_audio(events_iter, result, max_seconds=10.0, min_audio_ms=500.0)
        if result.audio_bytes == 0:
            result.expectations = {"audio_received_before_barge_in": False}
            return result

        # Inject real synthesized speech to force a server-VAD-driven interrupt.
        speech = await _tts_user_speech("Stop, please. I changed my mind.")
        chunk_bytes = 24000 // 1000 * 2 * 100  # 100 ms at 24 kHz mono int16
        barge_in_started = time.monotonic()
        result.timeline.append({"t": 0.0, "label": "barge_in_audio_start"})
        for offset in range(0, len(speech), chunk_bytes):
            await agent.send(AudioBlock(media_type="audio/pcm", data=speech[offset : offset + chunk_bytes]))
            await asyncio.sleep(0.05)
        result.timeline.append({"t": time.monotonic() - barge_in_started, "label": "barge_in_audio_done"})

        await _watch_post_barge_in(events_iter, result, silence_window_seconds=1.5, max_seconds=12.0)

    result.expectations = {
        "audio_received_before_barge_in": result.audio_bytes > 0,
        "speech_started_observed": result.speech_started_after_interrupt_ms is not None,
        "silence_within_1500ms": (
            result.silence_after_interrupt_ms is not None and result.silence_after_interrupt_ms < 1500
        ),
    }
    return result


async def _drain_until_first_audio(
    events_iter: Any, result: Result, *, max_seconds: float, min_audio_ms: float
) -> None:
    """Read events until at least ``min_audio_ms`` of audio has accumulated."""
    started = time.monotonic()
    pcm_per_ms = 24 * 2
    threshold = int(min_audio_ms * pcm_per_ms)
    while True:
        if time.monotonic() - started > max_seconds:
            return
        try:
            event = await asyncio.wait_for(events_iter.__anext__(), timeout=max_seconds)
        except (TimeoutError, StopAsyncIteration):
            return
        if isinstance(event, AudioOutputDelta):
            result.audio_chunks += 1
            result.audio_bytes += len(event.data)
            result.pcm_buffer.extend(event.data)
            if result.audio_chunks == 1:
                result.timeline.append({"t": time.monotonic() - started, "label": "first_audio"})
            if result.audio_bytes >= threshold:
                return
        elif isinstance(event, TranscriptDelta) and event.role == "assistant":
            result.transcript_assistant += event.delta
        elif isinstance(event, ToolUseStart):
            result.tool_calls.append(event.name)


async def _watch_post_barge_in(
    events_iter: Any,
    result: Result,
    *,
    silence_window_seconds: float,
    max_seconds: float,
) -> None:
    """After fake user audio is injected, watch for SpeechStarted and confirm
    the model actually goes quiet within the silence window."""
    barge_in_at = time.monotonic()
    last_audio_at: float | None = None
    speech_started_at: float | None = None
    while True:
        elapsed = time.monotonic() - barge_in_at
        if elapsed > max_seconds:
            return
        try:
            event = await asyncio.wait_for(events_iter.__anext__(), timeout=silence_window_seconds)
        except (TimeoutError, StopAsyncIteration):
            if speech_started_at is not None:
                last_leak = max((last_audio_at or speech_started_at) - speech_started_at, 0.0)
                result.silence_after_interrupt_ms = last_leak * 1000
            return
        now = time.monotonic()
        if isinstance(event, AudioOutputDelta):
            result.audio_chunks += 1
            result.audio_bytes += len(event.data)
            result.pcm_buffer.extend(event.data)
            last_audio_at = now
        elif isinstance(event, SpeechStarted):
            if speech_started_at is None:
                speech_started_at = now
                result.speech_started_after_interrupt_ms = (now - barge_in_at) * 1000
                result.timeline.append({"t": now - barge_in_at, "label": "speech_started"})
        elif isinstance(event, TurnComplete):
            result.turns += 1
            result.timeline.append(
                {"t": now - barge_in_at, "label": "turn_complete", "stop_reason": event.stop_reason.name}
            )
            if speech_started_at is not None and last_audio_at is not None:
                result.silence_after_interrupt_ms = max(last_audio_at - speech_started_at, 0.0) * 1000
            return


def write_audio(path: str, pcm: bytes) -> None:
    if pcm:
        with open(path, "wb") as f:
            f.write(pcm)


def serialise(result: Result) -> str:
    # Drop the raw bytes from the JSON payload so stdout stays human-readable.
    d = dataclasses.asdict(result)
    d.pop("pcm_buffer", None)
    return json.dumps(d, indent=2, default=str)


async def main_async(args: argparse.Namespace) -> int:
    transport = make_transport(args.provider)
    if args.scenario == "tool":
        result = await scenario_tool(transport, args.provider)
    elif args.scenario == "cancel":
        result = await scenario_cancel(transport, args.provider)
    elif args.scenario == "voice_interrupt":
        result = await scenario_voice_interrupt(transport, args.provider)
    else:
        raise ValueError(args.scenario)

    if args.audio_out:
        write_audio(args.audio_out, bytes(result.pcm_buffer))
    print(serialise(result))
    return 0 if result.passed() else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    ap.add_argument(
        "--scenario",
        choices=["tool", "cancel", "voice_interrupt"],
        default="tool",
        help="tool: tool-call round trip; cancel: programmatic agent.interrupt(); "
        "voice_interrupt: inject fake user speech to trigger server-VAD barge-in.",
    )
    ap.add_argument("--audio-out", default="out.pcm", help="Path to dump received PCM16 mono audio")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
