"""Keep the terminal usable when no transport has credentials."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from axio.events import StreamEvent
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import TransportError


class UnconfiguredTransport:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.model = ModelSpec("unconfigured", frozenset({Capability.text, Capability.tool_use}))
        self.models = ModelRegistry([])

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        raise TransportError(self.reason)
