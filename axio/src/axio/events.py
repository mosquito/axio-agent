"""Stream events: all variants emitted by AgentStream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .blocks import AudioMediaType, ImageMediaType, VideoMediaType
from .types import StopReason, ToolCallID, ToolName, Usage


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class ToolUseStart:
    index: int
    tool_use_id: ToolCallID
    name: ToolName


@dataclass(frozen=True, slots=True)
class ToolInputDelta:
    index: int
    tool_use_id: ToolCallID
    partial_json: str


@dataclass(frozen=True, slots=True)
class ToolFieldStart:
    index: int
    tool_use_id: ToolCallID
    key: str


@dataclass(frozen=True, slots=True)
class ToolFieldDelta:
    index: int
    tool_use_id: ToolCallID
    key: str
    text: str


@dataclass(frozen=True, slots=True)
class ToolFieldEnd:
    index: int
    tool_use_id: ToolCallID
    key: str


@dataclass(frozen=True, slots=True)
class ToolOutputDelta:
    tool_use_id: ToolCallID
    name: ToolName
    key: str
    delta: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_use_id: ToolCallID
    name: ToolName
    is_error: bool
    content: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageOutput:
    """Model generated an image inline (e.g. Nano Banana / Gemini Image)."""

    index: int
    data: bytes
    media_type: ImageMediaType


@dataclass(frozen=True, slots=True)
class AudioOutput:
    """Audio content from a tool result (e.g. read_file on an audio file)."""

    index: int
    data: bytes
    media_type: AudioMediaType


@dataclass(frozen=True, slots=True)
class VideoOutput:
    """Model generated a video inline."""

    index: int
    data: bytes
    media_type: VideoMediaType


@dataclass(frozen=True, slots=True)
class IterationEnd:
    iteration: int
    stop_reason: StopReason
    usage: Usage


@dataclass(frozen=True, slots=True)
class Error:
    exception: BaseException


@dataclass(frozen=True, slots=True)
class SessionEndEvent:
    stop_reason: StopReason
    total_usage: Usage


# ── Realtime (duplex) events ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AudioOutputDelta:
    """Streaming audio chunk from the assistant in a realtime session."""

    data: bytes
    media_type: str = "audio/pcm;rate=24000"


@dataclass(frozen=True, slots=True)
class TranscriptDelta:
    """Live transcript delta — server-side STT of user mic, or assistant
    speech transcription, depending on ``role``."""

    role: Literal["user", "assistant"]
    delta: str


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """Server VAD detected the user started speaking (realtime)."""


@dataclass(frozen=True, slots=True)
class SpeechStopped:
    """Server VAD detected the user stopped speaking (realtime)."""


@dataclass(frozen=True, slots=True)
class TurnComplete:
    """Assistant turn finished in a realtime session.  ``stop_reason`` may be
    :class:`StopReason.tool_use` to signal that pending tool calls should run
    before the next turn starts."""

    stop_reason: StopReason
    usage: Usage | None = None


type StreamEvent = (
    ReasoningDelta
    | TextDelta
    | ImageOutput
    | AudioOutput
    | VideoOutput
    | ToolUseStart
    | ToolInputDelta
    | ToolFieldStart
    | ToolFieldDelta
    | ToolFieldEnd
    | ToolOutputDelta
    | ToolResult
    | IterationEnd
    | Error
    | SessionEndEvent
    | AudioOutputDelta
    | TranscriptDelta
    | SpeechStarted
    | SpeechStopped
    | TurnComplete
)
