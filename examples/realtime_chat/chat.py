"""Interactive realtime voice chat with a TUI volume meter.

Two parallel coroutines:
  * ``_push_mic`` — pumps Microphone chunks into the agent and computes a peak
    level for the dashboard.
  * ``_consume_events`` — feeds AudioOutputDelta into the Speaker, updates a
    speaker-side peak level, prints transcripts, and drops the speaker buffer
    on SpeechStarted (server VAD detected the user — barge-in).

A rich.live display shows mic / speaker meters, transcript, status line.

Usage::

    OPENAI_API_KEY=...  uv run python chat.py --provider openai
    GEMINI_API_KEY=...  uv run python chat.py --provider gemini

Press Ctrl+C to exit.  ``--no-tui`` falls back to plain stdout for terminals
that don't render TUIs well.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import math
import os
import sys
import termios
import tty
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import webrtcvad  # type: ignore[import-untyped]
from axio import (
    AudioOutputDelta,
    RealtimeAgent,
    RealtimeTransport,
    SpeechStarted,
    SpeechStopped,
    Tool,
    ToolInputDelta,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from axio.blocks import AudioBlock
from axio.events import Error
from axio.exceptions import GuardError
from axio.permission import PermissionGuard
from axio_audio import DuplexAudio, Microphone, Speaker
from axio_tools_local.list_files import list_files
from axio_tools_local.read_file import read_file
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aec import WebRtcAECProcessor


class _ResampleSpeaker:
    """Quacks like ``Speaker`` / ``DuplexAudio.speaker`` but resamples
    incoming PCM from ``src_rate`` to ``dst_rate`` before forwarding.

    Used when the duplex device is forced to a rate the realtime model
    doesn't emit at — typically AEC3 needing 48 kHz while OpenAI/Gemini
    Live emit 24 kHz.  Without this, ``feed(pcm_24k)`` on a 48 kHz
    device plays back at 2× speed — the classic cartoon-voice symptom.
    """

    __slots__ = ("_inner", "_src_rate", "_dst_rate", "_resample")

    def __init__(
        self,
        inner: Any,
        *,
        src_rate: int,
        dst_rate: int,
        resample: Callable[[bytes, int, int], bytes],
    ) -> None:
        self._inner = inner
        self._src_rate = src_rate
        self._dst_rate = dst_rate
        self._resample = resample

    async def feed(self, pcm: bytes) -> None:
        if self._src_rate != self._dst_rate:
            pcm = self._resample(pcm, self._src_rate, self._dst_rate)
        await self._inner.feed(pcm)

    async def stop(self) -> None:
        await self._inner.stop()

    def pending_bytes(self) -> int:
        # Translate device-rate bytes back into model-rate equivalents
        # so callers' "queued ms" calculations stay correct.
        n = self._inner.pending_bytes()
        if self._src_rate == self._dst_rate:
            return n
        return n * self._src_rate // self._dst_rate


# Friendly language names for the few BCP-47 codes Gemini Live supports.
# The model understands the BCP-47 tag fine on its own, but a natural-language
# instruction like "Respond in Russian." steers it noticeably better than
# "Respond in ru-RU.", and the user gets to see what we asked for in the
# status line.
_LANGUAGE_NAMES: dict[str, str] = {
    "ar-XA": "Arabic",
    "bn-IN": "Bengali",
    "cmn-CN": "Mandarin Chinese",
    "de-DE": "German",
    "en-AU": "English (Australia)",
    "en-GB": "English (United Kingdom)",
    "en-IN": "English (India)",
    "en-US": "English",
    "es-ES": "Spanish (Spain)",
    "es-US": "Spanish (US)",
    "fr-CA": "French (Canada)",
    "fr-FR": "French",
    "gu-IN": "Gujarati",
    "hi-IN": "Hindi",
    "id-ID": "Indonesian",
    "it-IT": "Italian",
    "ja-JP": "Japanese",
    "kn-IN": "Kannada",
    "ko-KR": "Korean",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "nl-NL": "Dutch",
    "pl-PL": "Polish",
    "pt-BR": "Portuguese (Brazil)",
    "ru-RU": "Russian",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "th-TH": "Thai",
    "tr-TR": "Turkish",
    "vi-VN": "Vietnamese",
}


def build_voice_prompt(
    persona: str,
    tools: list[Tool[Any]],
    language_code: str | None,
    fs_root: Path | None = None,
) -> str:
    """Compose a voice-tailored system prompt.

    Inspired by ``axio_repl.build_system_prompt``.  Tool list is
    rendered from the runtime ``tools`` so this stays in sync when the
    example grows new tools.

    ``language_code`` overrides the model's default output language —
    Gemini's ``speechConfig.languageCode`` only steers TTS voice
    selection, not the language the model writes in.
    """
    lines: list[str] = [persona, ""]

    lines += [
        "How to work:",
        "- Don't stop at the first tool call.  If the user asks you to "
        "explore, look around, summarise, or compare, keep calling tools "
        "until you actually have the answer.",
        "- Before replying, check that every part of the user's request is "
        "satisfied — for compound asks, mentally tick each item off.  If "
        "something is still missing, do the next tool call instead of "
        "guessing or hand-waving.",
        "- Never refuse a safe request, never claim inability when a tool "
        "could answer it, never give a half answer because the full one "
        "would take a few more steps.",
        "- If a tool call fails, look at the error and try a different "
        "approach.  Stuck after a few tries — say what you tried and ask "
        "for guidance, don't silently give up.",
        "",
    ]

    if tools:
        lines.append(
            "Tools available — use them when the user is actually asking for "
            "that information, then answer conversationally rather than "
            "reading raw tool output:"
        )
        for tool in tools:
            blurb = (tool.description or "").strip().splitlines()[0] if tool.description else ""
            if blurb:
                lines.append(f"- {tool.name}: {blurb}")
            else:
                lines.append(f"- {tool.name}")
        if fs_root is not None:
            lines.append(f"Filesystem tools are read-only and restricted to {fs_root}.")
        lines.append("")

    if language_code:
        name = _LANGUAGE_NAMES.get(language_code, language_code)
        lines.append(f"Always respond in {name}.")

    return "\n".join(lines).rstrip() + "\n"


async def get_weather(city: str) -> str:
    """Look up current weather for ``city``.  Returns a one-line summary
    with temperature, conditions, humidity, and wind speed."""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"weather lookup failed: HTTP {resp.status}"
                data = await resp.json(content_type=None)
    except Exception as exc:
        return f"weather lookup failed: {type(exc).__name__}: {exc}"
    try:
        cur = data["current_condition"][0]
        nearest = data.get("nearest_area", [{}])[0]
        place = nearest.get("areaName", [{}])[0].get("value", city)
        country = nearest.get("country", [{}])[0].get("value", "")
        desc = cur.get("weatherDesc", [{}])[0].get("value", "")
        temp_c = cur.get("temp_C", "?")
        feels_c = cur.get("FeelsLikeC", "?")
        humidity = cur.get("humidity", "?")
        wind_kmph = cur.get("windspeedKmph")
        try:
            wind = f"{float(wind_kmph) / 3.6:.1f}"
        except (TypeError, ValueError):
            wind = "?"
        loc = f"{place}, {country}".strip(", ")
        return f"{loc}: {temp_c}°C ({desc.lower()}), feels like {feels_c}°C, humidity {humidity}%, wind {wind} m/s"
    except (KeyError, IndexError, TypeError) as exc:
        return f"weather lookup parse failed: {exc}"


class CwdGuard(PermissionGuard):
    """Confine any path-typed argument to a single root directory.

    The root is captured once at construction time (so subsequent ``os.chdir``
    calls don't widen the jail) and every listed argument is resolved against
    it.  Symlinks are followed via ``Path.resolve(strict=False)``: if the
    resolved target is outside the root the call is denied with
    ``GuardError``.  Non-string args and missing keys are passed through —
    the guard only validates what it can interpret as a path.
    """

    def __init__(self, *path_args: str, root: Path | None = None) -> None:
        self.root = (root or Path.cwd()).resolve()
        self._path_args = path_args or ("filename", "directory", "path")

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        for name in self._path_args:
            value = kwargs.get(name)
            if not isinstance(value, str):
                continue
            candidate = (self.root / value).resolve()
            try:
                candidate.relative_to(self.root)
            except ValueError as exc:
                raise GuardError(
                    f"{tool.name}: path {value!r} resolves to {candidate}, "
                    f"which is outside the allowed root {self.root}"
                ) from exc
        return kwargs


async def find_files(pattern: str, directory: str = ".", max_matches: int = 100) -> str:
    """Recursively find files whose name matches a glob ``pattern`` (e.g.
    ``*.py``, ``test_*.txt``).  ``directory`` is relative to the project
    root.  Returns one path per line, capped at ``max_matches``."""

    def _blocking() -> str:
        root = Path(os.getcwd()) / directory
        if not root.is_dir():
            return f"not a directory: {directory}"
        hits: list[str] = []
        for path in root.rglob("*"):
            if path.is_file() and fnmatch.fnmatch(path.name, pattern):
                rel = os.path.relpath(path, os.getcwd())
                hits.append(rel)
                if len(hits) >= max_matches:
                    break
        if not hits:
            return f"no files match {pattern!r} under {directory}"
        return "\n".join(hits)

    return await asyncio.to_thread(_blocking)


async def grep_files(
    pattern: str,
    directory: str = ".",
    file_glob: str = "*",
    max_matches: int = 50,
) -> str:
    """Recursively search file contents for a substring ``pattern`` (case
    sensitive).  ``file_glob`` filters which file names are searched
    (default ``*``).  Returns ``path:line:content`` lines, capped at
    ``max_matches`` total matches across all files."""

    def _blocking() -> str:
        root = Path(os.getcwd()) / directory
        if not root.is_dir():
            return f"not a directory: {directory}"
        hits: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file() or not fnmatch.fnmatch(path.name, file_glob):
                continue
            try:
                with path.open("r", encoding="utf-8", errors="strict") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if pattern in line:
                            rel = os.path.relpath(path, os.getcwd())
                            hits.append(f"{rel}:{lineno}:{line.rstrip()}")
                            if len(hits) >= max_matches:
                                return "\n".join(hits)
            except (OSError, UnicodeDecodeError):
                # Binary or unreadable — skip silently, the model doesn't
                # need a flood of "skipped" lines polluting its result.
                continue
        if not hits:
            return f"no matches for {pattern!r} in {file_glob} under {directory}"
        return "\n".join(hits)

    return await asyncio.to_thread(_blocking)


_METER_FLOOR_DB = -60.0
"""Quietest level the meter renders.  Anything below this prints an empty bar."""


@dataclass
class _LevelMeter:
    """Rolling-average level meter.

    Holds a small deque of RMS levels (one per audio chunk) so the meter
    averages over ~``window_ms`` instead of jumping to whatever the current
    chunk's instantaneous peak happens to be.
    """

    window_ms: int = 250
    chunk_ms: int = 50
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=max(1, self.window_ms // max(self.chunk_ms, 1)))

    def push(self, pcm: bytes) -> None:
        self.samples.append(_rms_pcm16(pcm))

    def decay(self, factor: float = 0.6) -> None:
        """Pull the meter toward 0 between chunks so it visibly drops on silence."""
        if self.samples:
            self.samples.append(self.samples[-1] * factor)

    @property
    def rms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    @property
    def dbfs(self) -> float:
        rms = self.rms
        if rms <= 1e-6:
            return _METER_FLOOR_DB
        return max(_METER_FLOOR_DB, 20.0 * math.log10(rms))


class _VadStrip:
    """Rolling-window WebRTC VAD on whatever PCM we send to the agent.

    Shows the user what the server-side VAD is most likely seeing — if
    the strip lights up while the model is talking and the user is
    silent, echo is leaking through and the model will self-interrupt.
    If the strip stays dark during model speech and lights up the moment
    the user opens their mouth, AEC + ducking are doing their job.

    webrtcvad operates on 8 / 16 / 32 / 48 kHz int16 mono frames of
    10/20/30 ms.  Mic chunks at unsupported rates (e.g. OpenAI's 24 kHz)
    are linearly resampled to 16 kHz for VAD only — the bytes sent to
    the agent are unaffected.
    """

    def __init__(
        self,
        src_rate: int,
        *,
        aggressiveness: int = 2,
        window_frames: int = 80,
        hangover_frames: int = 5,
    ) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)
        # webrtcvad rejects rates outside its supported set; resample if needed.
        self._vad_rate = src_rate if src_rate in (8000, 16000, 32000, 48000) else 16000
        self._src_rate = src_rate
        self._frame_ms = 30
        self._bytes_per_frame = self._vad_rate * self._frame_ms // 1000 * 2
        self._pending = bytearray()
        self.history: deque[bool] = deque(maxlen=window_frames)
        self._hangover = hangover_frames

    def feed(self, pcm: bytes) -> None:
        if not pcm:
            return
        if self._src_rate != self._vad_rate:
            from aec import _linear_resample_pcm16

            pcm = _linear_resample_pcm16(pcm, self._src_rate, self._vad_rate)
        self._pending.extend(pcm)
        while len(self._pending) >= self._bytes_per_frame:
            frame = bytes(self._pending[: self._bytes_per_frame])
            del self._pending[: self._bytes_per_frame]
            try:
                active = self._vad.is_speech(frame, self._vad_rate)
            except Exception:
                # webrtcvad raises on weird payloads; skip the frame.
                active = False
            self.history.append(active)

    @property
    def trigger_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def is_open(self) -> bool:
        """Return ``True`` when the local VAD considers voice activity to
        be present right now (with a short hangover for clean releases).

        The empty-history return is ``True`` so that gating doesn't drop
        the very first chunks before we have any evidence one way or
        the other.
        """
        if not self.history:
            return True
        # Tail of the history covers ``hangover_frames * frame_ms`` of
        # decision context — defaults to 5 × 30 ms = 150 ms.  Any voice
        # activity within that window keeps the gate open.
        recent = list(self.history)[-self._hangover :]
        return any(recent)


@dataclass
class Dashboard:
    """Shared mutable state rendered by rich.live."""

    mic: _LevelMeter = field(default_factory=_LevelMeter)
    speaker: _LevelMeter = field(default_factory=_LevelMeter)
    speaker_pending_bytes: int = 0
    user_transcript: str = ""
    assistant_transcript: str = ""
    status: str = "connecting…"
    tool_calls: int = 0
    speech_active: bool = False
    mic_ducked: bool = False
    last_audio_output_at: float = 0.0
    log: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    # AEC diagnostics: (far, near, out) dBFS rolling-1s.  Updated by
    # ``_consume_events`` if AEC is enabled; ``None`` otherwise.
    aec_levels: tuple[float, float, float] | None = None
    aec_overflow_bytes: int = 0
    # Local mirror of the server's VAD on whatever bytes we send to the
    # agent — populated by ``_push_mic`` when ``--show-vad`` is on.
    vad: _VadStrip | None = None

    def _render_meters(self, console_width: int) -> Group:
        # Panel frame (4) + label (3) + " " (1) + bar + "  -XX.X dBFS" (13) = 21
        meter_width = max(8, console_width - 21)
        # Plain string, no rich-markup brackets — "[muted]" would be parsed
        # as an unknown style tag and silently swallowed.
        # Mic panel border encodes VAD state (when --show-vad is active):
        # bright yellow if the local VAD mirrored on the bytes we send
        # to the agent is currently firing, dim cyan otherwise.  This is
        # the same signal the server-side VAD operates on, so the panel
        # going yellow during model output is a strong signal that
        # echo cancellation is failing and the model is about to
        # self-interrupt.
        mic_border = "cyan"
        mic_title = "Microphone (muted)" if self.mic_ducked else "Microphone"
        if self.vad is not None:
            current = bool(self.vad.history and self.vad.history[-1])
            mic_border = "bold yellow" if current else "grey50"
            mic_title += f"   VAD {self.vad.trigger_rate * 100:.0f}%"
        panels: list[Any] = [
            Panel(
                _meter(self.mic.dbfs, width=meter_width, label="Mic"),
                title=mic_title,
                border_style=mic_border,
            ),
            Panel(
                _meter(self.speaker.dbfs, width=meter_width, label="Spk"),
                title=f"Speaker (queued {self.speaker_pending_bytes // 48:>5} ms)",
                border_style="magenta",
            ),
        ]
        if self.aec_levels is not None:
            far, near, out = self.aec_levels
            suppression = near - out
            text = Text()
            text.append(f"far  {far:>6.1f} dBFS   ", style="grey70")
            text.append(f"near {near:>6.1f} dBFS   ", style="grey70")
            text.append(f"out  {out:>6.1f} dBFS   ", style="grey70")
            colour = "green" if suppression >= 15 else ("yellow" if suppression >= 5 else "red")
            text.append(f"suppress {suppression:>+5.1f} dB", style=f"bold {colour}")
            if self.aec_overflow_bytes:
                text.append(f"   drift drops {self.aec_overflow_bytes:>6d} B", style="grey50")
            panels.append(Panel(text, title="AEC", border_style="grey42"))
        return Group(*panels)

    def _render_status(self) -> Panel:
        return Panel(
            Text(
                "  ".join(
                    [
                        f"status: {self.status}",
                        f"tool calls: {self.tool_calls}",
                        "Esc: interrupt",
                        "Ctrl+C: exit",
                    ]
                )
            ),
            border_style="grey50",
            height=3,
        )

    def render(self, *, console_width: int = 80, console_height: int = 24) -> Layout:
        layout = Layout()
        # Each Panel takes 3 rows (top border + content + bottom border).
        meters_rows = 3 * (3 if self.aec_levels is not None else 2)
        layout.split_column(
            Layout(self._render_meters(console_width), name="meters", size=meters_rows),
            Layout(
                Panel(_transcript_table(self), title="Conversation", border_style="green"),
                name="conversation",
                ratio=1,
            ),
            Layout(self._render_status(), name="status", size=3),
        )
        return layout


def _meter(dbfs: float, *, width: int, label: str) -> Text:
    """dBFS-scaled meter: fully filled at 0 dBFS, empty at ``_METER_FLOOR_DB``."""
    norm = (dbfs - _METER_FLOOR_DB) / -_METER_FLOOR_DB  # 0..1
    norm = min(max(norm, 0.0), 1.0)
    filled = int(norm * width)
    bar = "█" * filled + "░" * (width - filled)
    if dbfs >= -3.0:
        colour = "red"
    elif dbfs >= -18.0:
        colour = "yellow"
    else:
        colour = "green"
    text = Text()
    text.append(label + " ", style="bold")
    text.append(bar[:filled], style=colour)
    text.append(bar[filled:], style="grey42")
    text.append(f"  {dbfs:6.1f} dBFS", style="grey70")
    return text


def _rms_pcm16(data: bytes) -> float:
    """Return the RMS amplitude of int16 mono PCM bytes, normalised to 0..1.

    Uses numpy because this function runs on the audio callback thread
    (via ``Speaker.playback_tap``), where a Python-level for-loop over
    480-sample chunks at ~50 Hz is enough to push past the audio
    callback's deadline on slower machines and cause buffer underruns.
    """
    if not data:
        return 0.0
    samples = np.frombuffer(data, dtype="<i2")
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0


def _transcript_table(d: Dashboard) -> Table:
    t = Table.grid(expand=True)
    t.add_column(no_wrap=False)
    if d.user_transcript:
        t.add_row(Text("you  ▶ ", style="cyan bold") + Text(d.user_transcript))
    if d.assistant_transcript:
        t.add_row(Text("model▶ ", style="magenta bold") + Text(d.assistant_transcript))
    for line in d.log:
        t.add_row(Text(line, style="grey70"))
    if not d.user_transcript and not d.assistant_transcript and not d.log:
        t.add_row(Text("(waiting for first turn…)", style="grey42 italic"))
    return t


# Per-transport metadata: env vars that signal "credentials present" for
# auto-detection (axio-repl style), plus the audio rates the provider's
# realtime stream actually uses.  Keyed by the entry-point name from the
# ``axio.transport.realtime`` group.
_TRANSPORT_ENV_VARS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "vertex": ["GOOGLE_CLOUD_PROJECT"],
}
_TRANSPORT_RATES: dict[str, tuple[int, int]] = {
    "openai": (24000, 24000),
    "gemini": (16000, 24000),
    "vertex": (16000, 24000),
}


def _discover_realtime_transports() -> dict[str, Callable[..., RealtimeTransport]]:
    from importlib.metadata import entry_points

    result: dict[str, Callable[..., RealtimeTransport]] = {}
    for ep in entry_points(group="axio.transport.realtime"):
        try:
            result[ep.name] = ep.load()
        except Exception:
            pass
    return result


def _select_transport(name: str | None) -> str:
    """Return a chosen transport name. Falls back to env-var sniffing."""
    available = _discover_realtime_transports()
    if name:
        if name not in available:
            raise SystemExit(f"Unknown transport {name!r}. Available: {', '.join(sorted(available))}")
        return name
    for transport_name in available:
        envs = _TRANSPORT_ENV_VARS.get(transport_name, [])
        if any(os.environ.get(v) for v in envs):
            return transport_name
    msg = ["No realtime transport credentials found. Set one of:"]
    for transport_name in available:
        envs = _TRANSPORT_ENV_VARS.get(transport_name, [])
        if envs:
            msg.append(f"  {', '.join(envs)}  ({transport_name})")
    raise SystemExit("\n".join(msg))


def make_transport(name: str, args: argparse.Namespace) -> tuple[RealtimeTransport, int, int, str | None]:
    """Returns ``(transport, mic_rate, speaker_rate, language_code)``."""
    available = _discover_realtime_transports()
    cls = available[name]
    mic_rate, spk_rate = _TRANSPORT_RATES.get(name, (24000, 24000))
    # Provider-specific kwargs.  We can't pass everything blindly because
    # the OpenAI and Gemini transports expose different VAD knobs and
    # auth surfaces — this dispatch is the small price of letting the
    # transport classes stay independent.
    if name == "openai":
        return (
            cls(
                vad_threshold=args.vad_threshold,
                vad_silence_duration_ms=args.vad_silence_ms,
                vad_interrupt_response=not args.no_auto_interrupt,
                input_noise_reduction=None if args.no_noise_reduction else args.noise_reduction,
            ),
            mic_rate,
            spk_rate,
            args.language_code,
        )
    # gemini / vertex
    from axio_transport_google.realtime import detect_system_language

    language = args.language_code or detect_system_language()
    return (
        cls(language_code=language, auto_region=args.auto_region),
        mic_rate,
        spk_rate,
        language,
    )


async def _push_mic(
    agent: RealtimeAgent,
    mic: Microphone,
    speaker: Speaker,
    dash: Dashboard,
    aec: WebRtcAECProcessor | None,
    *,
    duck_during_playback: bool,
    duck_release_ms: int,
    local_vad_gate: bool,
    pause_when_muted: bool,
    vad_close_tail_ms: int,
) -> None:
    """Forward mic chunks to the agent, with optional ducking and VAD gating.

    AEC always sees the **real** mic signal — even when we end up
    sending silence to the server.  Feeding AEC silence during gated
    periods would let its adaptive echo-path estimate decay (no echo
    to subtract from → nothing to learn), and on gate release the
    canceller would have to cold-restart its filter against the
    next live frame.  Pumping the real signal keeps AEC continuously
    locked onto the room.

    The decision tree is:

    * ``duck_during_playback`` and the model is playing back through
      the speaker → mute (send zero PCM the same length the cleaned
      chunk would have been).  Hard-blocks bleed-back regardless of
      AEC quality.
    * ``local_vad_gate`` and the local VAD doesn't currently see
      voice → mute.  Stricter than the server's VAD, blocks AEC
      residual + background noise that would otherwise trigger
      self-interrupt.
    * Otherwise → send the AEC-cleaned (or raw, when AEC is off) PCM.

    The local VAD always sees the cleaned signal (not the muted
    output), otherwise it would never reopen the gate once it
    closed.

    ``vad_close_tail_ms`` is the trailing silence window after the
    local VAD closes the gate: with ``pause_when_muted=True`` we'd
    otherwise stop sending frames cold, but the server VAD's
    silence-duration timer can't advance without frames — so it
    never commits the turn.  During the tail we keep streaming zero
    PCM so the server sees a clean silence-end transition; after
    the tail expires we go fully silent.  The tail does not apply
    to duck-mute (model owns the speaker; user-end-of-speech is
    not the relevant event there).
    """
    import time as _time

    vad_gate_closed_at: float | None = None

    async for chunk in mic:
        # Step 1: AEC sees the real mic regardless of gating decisions.
        if aec is not None:
            cleaned = aec.process_mic(chunk.data)
            if not cleaned:
                # AEC buffers partial frames internally; nothing to
                # emit yet for this tick.
                continue
        else:
            cleaned = chunk.data

        dash.mic.push(cleaned)
        if dash.vad is not None:
            dash.vad.feed(cleaned)

        # Step 2: figure out whether to mute, and which mute applies.
        now = _time.monotonic()
        duck_muted = False
        if duck_during_playback:
            if speaker.pending_bytes() > 0 or (
                dash.last_audio_output_at > 0 and (now - dash.last_audio_output_at) * 1000 < duck_release_ms
            ):
                duck_muted = True
        gate_muted = local_vad_gate and dash.vad is not None and not dash.vad.is_open()

        # Track the rising edge of the VAD-gate close so we can drive
        # a fixed-duration silence tail through the server VAD.
        if gate_muted and vad_gate_closed_at is None:
            vad_gate_closed_at = now
        elif not gate_muted:
            vad_gate_closed_at = None

        muted = duck_muted or gate_muted
        dash.mic_ducked = muted

        if muted and pause_when_muted:
            in_tail = (
                gate_muted
                and not duck_muted
                and vad_gate_closed_at is not None
                and (now - vad_gate_closed_at) * 1000 < vad_close_tail_ms
            )
            if not in_tail:
                # Skip sending entirely — both OpenAI Realtime and
                # Gemini Live accept gaps in the audio stream; the
                # server VAD's silence timer just doesn't advance
                # while no frames arrive.
                continue

        outgoing = b"\x00" * len(cleaned) if muted else cleaned
        await agent.send(AudioBlock(media_type="audio/pcm", data=outgoing))


async def _consume_events(
    agent: RealtimeAgent,
    speaker: Speaker,
    dash: Dashboard,
    aec: WebRtcAECProcessor | None,
) -> None:
    # Buffer per-tool-call partial JSON so we can show the assembled
    # arguments in the conversation log on the next turn-complete.
    pending_tools: dict[str, dict[str, Any]] = {}

    async for event in agent.events():
        if isinstance(event, AudioOutputDelta):
            # Don't push to dash.speaker here — the model bursts the whole
            # response at us in <100 ms and the meter would briefly spike
            # then sit at zero for the entire real playback.  The speaker's
            # playback_tap (wired in main_async) feeds the meter at the
            # rate samples actually hit the audio device.
            await speaker.feed(event.data)
            dash.speaker_pending_bytes = speaker.pending_bytes()
            if aec is not None:
                dash.aec_levels = aec.levels_dbfs()
                dash.aec_overflow_bytes = aec.far_overflow_bytes
        elif isinstance(event, SpeechStarted):
            dash.speech_active = True
            dash.log.append("[barge-in] flushing speaker buffer")
            await speaker.stop()
            dash.speaker_pending_bytes = 0
        elif isinstance(event, SpeechStopped):
            dash.speech_active = False
        elif isinstance(event, ToolUseStart):
            dash.tool_calls += 1
            pending_tools[event.tool_use_id] = {"name": event.name, "parts": []}
        elif isinstance(event, ToolInputDelta):
            tracker = pending_tools.get(event.tool_use_id)
            if tracker is not None:
                tracker["parts"].append(event.partial_json)
        elif isinstance(event, TranscriptDelta):
            if event.role == "user":
                dash.user_transcript += event.delta
            else:
                dash.assistant_transcript += event.delta
        elif isinstance(event, TurnComplete):
            for tracker in pending_tools.values():
                raw = "".join(tracker["parts"])
                try:
                    args = json.loads(raw) if raw else {}
                    pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
                except json.JSONDecodeError:
                    pretty = raw
                dash.log.append(f"[tool→] {tracker['name']}({pretty})")
            pending_tools.clear()
            if dash.user_transcript or dash.assistant_transcript:
                if dash.user_transcript:
                    dash.log.append(f"you: {dash.user_transcript}")
                if dash.assistant_transcript:
                    dash.log.append(f"model: {dash.assistant_transcript}")
                dash.user_transcript = ""
                dash.assistant_transcript = ""
            dash.status = f"turn complete ({event.stop_reason.name})"
        elif isinstance(event, Error):
            dash.status = f"error: {event.exception}"
            return


async def _decay_meters(
    dash: Dashboard,
    aec: WebRtcAECProcessor | None = None,
) -> None:
    """Push a decaying value into each meter so silent periods visibly drop."""
    while True:
        await asyncio.sleep(0.05)
        if aec is not None:
            dash.aec_levels = aec.levels_dbfs()
            dash.aec_overflow_bytes = aec.far_overflow_bytes
        dash.mic.decay()
        dash.speaker.decay()


@contextmanager
def _raw_stdin_keyhandler(on_esc: Callable[[], Coroutine[Any, Any, Any]]) -> Iterator[None]:
    """Put stdin in cbreak mode and dispatch ``on_esc`` when the user hits
    Escape.  Other keys (including arrow sequences that start with ESC) are
    swallowed silently so they don't pollute the terminal.

    The implementation handles the classic Escape vs ESC-prefix-of-sequence
    ambiguity by deferring the ``on_esc`` call by 50 ms; if a follow-up byte
    arrives in that window we treat the ESC as the start of an escape
    sequence (e.g. ``ESC [ A`` for arrow-up) and cancel the deferred call.
    """
    if not sys.stdin.isatty():
        # Detached stdin (pipe, redirect) — nothing to do.
        yield
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    loop = asyncio.get_running_loop()
    pending_esc: asyncio.TimerHandle | None = None

    def fire() -> None:
        nonlocal pending_esc
        pending_esc = None
        loop.create_task(on_esc())

    def reader() -> None:
        nonlocal pending_esc
        try:
            data = os.read(fd, 64)
        except BlockingIOError:
            return
        if pending_esc is not None:
            # Follow-up byte → ESC was the prefix of an escape sequence.
            pending_esc.cancel()
            pending_esc = None
            if data.startswith(b"\x1b"):
                # Sequence with a NEW lone ESC at its head — re-arm.
                pending_esc = loop.call_later(0.05, fire)
            return
        if data == b"\x1b":
            pending_esc = loop.call_later(0.05, fire)

    loop.add_reader(fd, reader)
    try:
        yield
    finally:
        if pending_esc is not None:
            pending_esc.cancel()
        try:
            loop.remove_reader(fd)
        except ValueError:
            pass
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _print_devices() -> None:
    import sounddevice as sd  # type: ignore[import-untyped]

    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    print(f"{'idx':>3}  {'host':<8}  {'in':>3}  {'out':>3}  {'rate':>7}  name")
    for i, d in enumerate(devices):
        host = sd.query_hostapis(d["hostapi"])["name"][:8]
        marker = ""
        if i == default_in:
            marker += "I"
        if i == default_out:
            marker += "O"
        marker = f" [{marker}]" if marker else ""
        print(
            f"{i:>3}  {host:<8}  {d['max_input_channels']:>3}  "
            f"{d['max_output_channels']:>3}  {int(d['default_samplerate']):>7}  "
            f"{d['name']}{marker}"
        )


def _resolve_device(spec: str | None) -> int | str | None:
    if spec is None:
        return None
    try:
        return int(spec)
    except ValueError:
        return spec


async def main_async(args: argparse.Namespace) -> int:
    transport_name = _select_transport(args.transport)
    transport, mic_rate, spk_rate, language = make_transport(transport_name, args)
    voice: str | None = args.voice or ("marin" if transport_name == "openai" else None)
    dash = Dashboard()
    if args.show_vad:
        # Hangover converted from ms to webrtcvad's 30 ms frame steps.
        dash.vad = _VadStrip(
            mic_rate,
            aggressiveness=args.vad_aggressiveness,
            hangover_frames=max(1, args.vad_hangover_ms // 30),
        )
    dash.status = f"connecting to {transport_name} (mic {mic_rate} Hz / spk {spk_rate} Hz)"
    if language:
        dash.log.append(f"language: {_LANGUAGE_NAMES.get(language, language)} ({language})")
    console = Console()

    in_device: int | str | None = _resolve_device(args.input_device)
    out_device: int | str | None = _resolve_device(args.output_device)

    # functools.wraps preserves __doc__ + signature so axio.Tool picks them up
    # automatically — its __post_init__ falls back to handler.__doc__ when no
    # explicit description= is passed, and derives the JSON schema from
    # inspect.signature + get_type_hints.  No need to repeat the description
    # string at the Tool() call site.
    from functools import wraps

    @wraps(get_weather)
    async def get_weather_logged(city: str) -> str:
        result = await get_weather(city)
        short = result if len(result) <= 90 else result[:87] + "…"
        dash.log.append(f"[tool←] {short}")
        return result

    # File-system tools share one cwd-jail guard.  ``read_file`` /
    # ``list_files`` accept ``filename`` / ``directory``; ``find_files`` and
    # ``grep_files`` only expose ``directory``.  Every path-typed argument
    # name they use must appear in the guard's path_args list.
    cwd_guard = CwdGuard("filename", "directory", root=Path(os.getcwd()))
    fs_tools: list[Tool[Any]] = [
        Tool(name="read_file", handler=read_file, guards=(cwd_guard,)),
        Tool(name="list_files", handler=list_files, guards=(cwd_guard,)),
        Tool(name="find_files", handler=find_files, guards=(cwd_guard,)),
        Tool(name="grep_files", handler=grep_files, guards=(cwd_guard,)),
    ]
    dash.log.append(f"fs tools jailed to {cwd_guard.root}")

    agent_tools: list[Tool[Any]] = [
        Tool(name="get_weather", handler=get_weather_logged),
        *fs_tools,
    ]
    system_prompt = build_voice_prompt(
        args.system,
        agent_tools,
        language,
        fs_root=cwd_guard.root,
    )

    aec: WebRtcAECProcessor | None = None
    aec_capture_rate = mic_rate  # the rate we actually drive the mic at
    if args.echo_cancel:
        # AEC3 only supports {16, 32, 48} kHz.  Pick the smallest one
        # that's >= max(mic_rate, spk_rate) so the device runs at a
        # native AEC3 rate without unnecessary sample loss; the
        # cleaned mic output is then resampled back to mic_rate
        # before going to the model.
        for candidate in (16000, 32000, 48000):
            if candidate >= max(mic_rate, spk_rate):
                aec_capture_rate = candidate
                break
        else:
            aec_capture_rate = 48000
        try:
            aec = WebRtcAECProcessor(
                sample_rate=aec_capture_rate,
                output_rate=mic_rate,
                latency_hint_ms=args.echo_cancel_latency_ms,
            )
        except (RuntimeError, ImportError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2
        rate_note = (
            ""
            if aec_capture_rate == mic_rate
            else f"; mic captured at {aec_capture_rate} Hz, resampled to {mic_rate} Hz"
        )
        dash.status = f"in-process AEC: webrtc-aec3 (AEC3 + NS + HPF + TS){rate_note}"

    # The speaker callback fires for every chunk that actually leaves the
    # device — that's the right clock for both the AEC reference (so the
    # echo path is bit-aligned) and the dashboard meter (so it tracks
    # real playback, not the burst at which the model emits).  Both
    # consumers run from a sounddevice audio thread, so they must stay
    # cheap and non-blocking.
    import time as _time_mod

    # The speaker pads short reads with silence so the device doesn't
    # starve, and the tap sees those silence ticks too.  Push them into
    # the meter (so it visibly drops between phrases) but DON'T treat
    # them as fresh playback for mic-duck — otherwise the duck never
    # releases because the silence-pad keeps refreshing the timer.
    _silent_threshold = b"\x00\x00"

    def _on_playback(pcm: bytes) -> None:
        dash.speaker.push(pcm)
        if pcm.replace(_silent_threshold, b""):
            dash.last_audio_output_at = _time_mod.monotonic()
        if aec is not None:
            aec.feed_speaker(pcm)

    # When AEC is on we use a single duplex PortAudio stream so mic and
    # speaker share one clock — separate input/output streams drift
    # ~10–25 ms/sec on consumer audio stacks, which after a few minutes
    # pushes the far reference past AEC3's internal search window and
    # cancellation collapses.  Without AEC, drift still happens but
    # nobody cares — keep the simpler split-streams setup so users
    # without compatible hardware aren't forced into duplex mode.
    @asynccontextmanager
    async def _audio() -> AsyncIterator[tuple[Any, Any]]:
        if args.echo_cancel:
            duplex_device: Any
            if in_device is not None or out_device is not None:
                duplex_device = (in_device, out_device)
            else:
                duplex_device = None
            async with DuplexAudio(
                sample_rate=aec_capture_rate,
                chunk_ms=args.chunk_ms,
                channels=1,
                device=duplex_device,
                playback_tap=_on_playback,
            ) as duplex:
                # When the duplex device runs faster than what the model
                # sends (e.g. AEC3 forces 48 kHz on the device but the
                # model emits 24 kHz audio), feeding the raw PCM through
                # would play it at the wrong speed — pitch-shifted up,
                # cartoon-voice symptom.  Resample on the way in so the
                # device clock and the model clock agree.
                speaker_view: Any = duplex.speaker
                if spk_rate != aec_capture_rate:
                    from aec import _linear_resample_pcm16

                    speaker_view = _ResampleSpeaker(
                        duplex.speaker,
                        src_rate=spk_rate,
                        dst_rate=aec_capture_rate,
                        resample=_linear_resample_pcm16,
                    )
                yield duplex.mic, speaker_view
        else:
            async with (
                Microphone(sample_rate=aec_capture_rate, chunk_ms=args.chunk_ms, device=in_device) as mic,
                Speaker(sample_rate=spk_rate, device=out_device, playback_tap=_on_playback) as speaker,
            ):
                yield mic, speaker

    async with (
        RealtimeAgent(
            system=system_prompt,
            transport=transport,
            tools=agent_tools,
            voice=voice,
        ) as agent,
        _audio() as (mic, speaker),
    ):
        if not args.echo_cancel:
            dash.status = "connected — start speaking"
        if args.no_tui:
            return await _run_plain(agent, mic, speaker, dash, aec, args)

        async def _on_esc() -> None:
            dash.log.append("[esc] interrupting model")
            try:
                await agent.interrupt()
            except Exception as exc:
                dash.log.append(f"[esc] interrupt failed: {exc}")
            await speaker.stop()

        with (
            _raw_stdin_keyhandler(_on_esc),
            Live(
                dash.render(console_width=console.width, console_height=console.height),
                console=console,
                # 10 Hz is enough for level meters: faster updates make
                # numbers blur into a smear of digits and burn CPU
                # re-rendering the whole layout, while slower drops
                # below the speed of speech transients.  Matched against
                # ``_refresh`` below — keep the two values in sync.
                refresh_per_second=10,
                screen=True,
            ) as live,
        ):

            async def _refresh() -> None:
                while True:
                    await asyncio.sleep(0.1)
                    live.update(dash.render(console_width=console.width, console_height=console.height))

            tasks = [
                asyncio.create_task(
                    _push_mic(
                        agent,
                        mic,
                        speaker,
                        dash,
                        aec,
                        duck_during_playback=args.mic_duck,
                        duck_release_ms=args.mic_duck_release_ms,
                        local_vad_gate=args.local_vad_gate,
                        pause_when_muted=args.pause_when_muted,
                        vad_close_tail_ms=args.vad_close_tail_ms,
                    ),
                    name="mic",
                ),
                asyncio.create_task(_consume_events(agent, speaker, dash, aec), name="events"),
                asyncio.create_task(_decay_meters(dash, aec), name="decay"),
                asyncio.create_task(_refresh(), name="refresh"),
            ]
            try:
                # Stop as soon as the events task ends (Error / EOF); the
                # mic / decay / refresh tasks are infinite loops and need
                # explicit cancel.
                await asyncio.wait(tasks[:2], return_when=asyncio.FIRST_COMPLETED)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            for t in tasks:
                t.cancel()
            # Cap the cleanup window — a single stuck task (e.g. websocket
            # send blocked on a half-closed socket) shouldn't keep the
            # whole process alive after the user has indicated they're
            # done.  Anything past the timeout gets dropped on the floor;
            # the OS will reclaim it on process exit.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0,
                )
            except TimeoutError:
                pass
    return 0


async def _run_plain(
    agent: RealtimeAgent,
    mic: Microphone,
    speaker: Speaker,
    dash: Dashboard,
    aec: WebRtcAECProcessor | None,
    args: argparse.Namespace,
) -> int:
    """Fallback when --no-tui is requested: plain stdout.

    Reuses ``_push_mic`` so the no-tui path picks up every gating /
    AEC behaviour the TUI path supports — earlier this was a
    duplicated loop that drifted out of sync (e.g. it didn't know
    about local-VAD gating or ``--pause-when-muted``).
    """

    async def consume() -> None:
        async for event in agent.events():
            if isinstance(event, AudioOutputDelta):
                await speaker.feed(event.data)
            elif isinstance(event, SpeechStarted):
                await speaker.stop()
                print("\n[barge-in]", flush=True)
            elif isinstance(event, ToolUseStart):
                print(f"[tool {event.name}]", flush=True)
            elif isinstance(event, TranscriptDelta):
                if event.role == "user":
                    print(f"\ryou: {event.delta}", end="", flush=True)
                else:
                    print(f"\rmodel: {event.delta}", end="", flush=True)
            elif isinstance(event, TurnComplete):
                print()
            elif isinstance(event, Error):
                print(f"\n[error: {event.exception}]", flush=True)
                return

    try:
        await asyncio.gather(
            _push_mic(
                agent,
                mic,
                speaker,
                dash,
                aec,
                duck_during_playback=args.mic_duck,
                duck_release_ms=args.mic_duck_release_ms,
                local_vad_gate=args.local_vad_gate,
                pause_when_muted=args.pause_when_muted,
                vad_close_tail_ms=args.vad_close_tail_ms,
            ),
            consume(),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--transport",
        default=None,
        help=(
            "Realtime transport name from the ``axio.transport.realtime`` "
            "entry-point group (e.g. openai, gemini, vertex).  When omitted, "
            "auto-selected by the first transport whose credentials are "
            "present in env (axio-repl-style)."
        ),
    )
    ap.add_argument(
        "--auto-region",
        action="store_true",
        help=(
            "Probe each supported Vertex Live region at connect time and pick "
            "the lowest-latency one for this network. Overrides GOOGLE_CLOUD_LOCATION. "
            "Adds ~1 s of startup latency."
        ),
    )
    ap.add_argument(
        "--language-code",
        default=None,
        help=(
            "BCP-47 language for Gemini speechConfig (e.g. ru-RU, en-US). "
            "Auto-detected from the system locale if it's on the supported "
            "list; otherwise the model's default (en-US) is used."
        ),
    )
    ap.add_argument(
        "--system",
        default=(
            "You are a friendly realtime voice assistant.  Treat the user as a "
            "conversation partner: listen closely, answer the question they "
            "actually asked, and keep the chat moving."
        ),
        help=(
            "Persona / opening line of the system prompt.  The runtime appends "
            "voice-output rules, the current tool list, and the language "
            "directive automatically — pass only the persona here."
        ),
    )
    ap.add_argument(
        "--voice",
        default=None,
        help=(
            "Voice name. OpenAI: alloy/ash/ballad/coral/echo/sage/shimmer/verse/"
            "marin/cedar (default marin if --provider=openai). "
            "Gemini: Puck/Charon/Kore/Fenrir/Aoede/etc (see Vertex Live docs)."
        ),
    )
    ap.add_argument("--chunk-ms", type=int, default=50)
    ap.add_argument("--no-tui", action="store_true", help="Plain stdout output (no rich live display).")
    ap.add_argument(
        "--list-devices",
        action="store_true",
        help="List sounddevice audio devices and exit.",
    )
    ap.add_argument(
        "--input-device",
        default=None,
        help="Input device index or name (e.g. 'pulse', 'pipewire').  Defaults to the system default.",
    )
    ap.add_argument(
        "--output-device",
        default=None,
        help="Output device index or name.  Defaults to the system default.",
    )
    ap.add_argument(
        "--echo-cancel",
        action="store_true",
        help=(
            "Run in-process AEC3 + NS + HPF + TS over the mic stream "
            "via the native ``webrtc_apm`` extension (same algorithm "
            "PipeWire's ``module-echo-cancel`` uses by default).  "
            "Forces a duplex audio stream so mic and speaker share one "
            "PortAudio clock — without that the in/out streams drift "
            "tens of ms per second on consumer audio stacks and AEC3's "
            "delay estimator loses lock."
        ),
    )
    ap.add_argument(
        "--echo-cancel-latency-ms",
        type=int,
        default=80,
        help=(
            "Initial mic ↔ speaker round-trip hint for AEC3's delay "
            "estimator.  AEC3 still adapts on its own; a roughly correct "
            "hint just speeds up the first second of convergence."
        ),
    )
    ap.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Server VAD activation threshold (0..1). Higher = less sensitive.",
    )
    ap.add_argument(
        "--vad-silence-ms",
        type=int,
        default=500,
        help="Silence (ms) required before the server marks a turn complete.",
    )
    ap.add_argument(
        "--no-auto-interrupt",
        action="store_true",
        help=(
            "Disable server-VAD-driven response cancellation. With AEC imperfect "
            "or in noisy environments this prevents the model from interrupting "
            "itself when its own audio bleeds back into the mic."
        ),
    )
    ap.add_argument(
        "--noise-reduction",
        choices=["near_field", "far_field"],
        default="near_field",
        help="Server-side input noise reduction profile (default near_field).",
    )
    ap.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="Disable server-side input noise reduction.",
    )
    ap.add_argument(
        "--mic-duck",
        action="store_true",
        help=(
            "Replace mic input with silence while the model is producing "
            "audio.  Hard-blocks the model's own voice from looping back "
            "and tripping server VAD — strictly stronger than AEC alone, "
            "at the cost of half-duplex turn-taking (you can't speak over "
            "the model with voice; use Ctrl+C / agent.interrupt() instead)."
        ),
    )
    ap.add_argument(
        "--mic-duck-release-ms",
        type=int,
        default=200,
        help="Hold the mic ducked this long after the last speaker chunk.",
    )
    ap.add_argument(
        "--show-vad",
        action="store_true",
        help=(
            "Run a local WebRTC VAD on the bytes we send to the agent and "
            "color the Microphone panel yellow while it's firing.  Useful "
            "for diagnosing AEC: yellow during model output (with user "
            "silent) means echo is getting through and the server VAD "
            "will probably self-interrupt.  Implied by --local-vad-gate."
        ),
    )
    ap.add_argument(
        "--local-vad-gate",
        action="store_true",
        help=(
            "Use the local WebRTC VAD as a hard gate: when it sees no voice "
            "activity, replace the outgoing mic chunk with silence so the "
            "server-side VAD never sees AEC residual.  Crank "
            "--vad-aggressiveness to 3 for the strictest gate (still passes "
            "real user voice, blocks most echo).  Implies --show-vad."
        ),
    )
    ap.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help=(
            "WebRTC VAD aggressiveness 0..3 (higher = stricter, fewer false "
            "positives — e.g. echo residual passes less often).  Used for "
            "both the dashboard indicator and --local-vad-gate."
        ),
    )
    ap.add_argument(
        "--vad-hangover-ms",
        type=int,
        default=150,
        help=(
            "Local VAD release time: keep the gate open this long after the "
            "last frame that detected voice.  Stops the user's voice from "
            "being chopped off at silence boundaries."
        ),
    )
    ap.add_argument(
        "--pause-when-muted",
        action="store_true",
        help=(
            "When the mic is muted (mic-duck during playback or local-VAD "
            "gate closed), skip ``agent.send`` entirely instead of padding "
            "with zero PCM.  Both OpenAI Realtime and Gemini Live accept "
            "stream gaps: the server VAD's silence-duration timer just "
            "doesn't advance while no frames arrive.  Saves bandwidth and "
            "stops the server from clocking 'silence elapsed' on our zero "
            "padding (useful when --vad-silence-ms is short and you don't "
            "want gated silence to count toward turn-end)."
        ),
    )
    ap.add_argument(
        "--vad-close-tail-ms",
        type=int,
        default=800,
        help=(
            "Trailing zero-PCM window after the local VAD gate closes, "
            "before --pause-when-muted starts dropping frames entirely.  "
            "Without this, a hard cut leaves the server VAD waiting for "
            "frames it never gets and the turn never commits.  Should be "
            "at least --vad-silence-ms with a little margin so the "
            "server's silence-duration timer reliably elapses."
        ),
    )
    args = ap.parse_args()
    if args.local_vad_gate:
        args.show_vad = True
    if args.list_devices:
        _print_devices()
        return 0

    # Two-strikes SIGINT escape hatch for the windows where asyncio is
    # NOT running.  asyncio.run installs its own SIGINT handler for
    # the lifetime of the loop (Runner.__enter__ → __exit__) — that one
    # already does cancel-then-KeyboardInterrupt on its own.  Our
    # handler covers the brief moments before asyncio.run starts and
    # after it returns; in particular, if PortAudio's audio thread or
    # the native ``webrtc_apm`` extension keeps the process alive past
    # ``asyncio.run`` and the user mashes Ctrl+C, we ``os._exit``
    # rather than make them reach for ``kill -9``.
    import signal

    sigint_count = [0]

    def _sigint_force_exit(signum: int, frame: Any) -> None:
        sigint_count[0] += 1
        if sigint_count[0] >= 2:
            print("\n[force exit on second Ctrl+C]", file=sys.stderr, flush=True)
            os._exit(130)
        # First press — restore the default behaviour and re-raise so
        # asyncio's loop sees a real KeyboardInterrupt and can run
        # graceful teardown.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_force_exit)

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
