"""Independent provider fixture with no imports from either frontend."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from axio.events import IterationEnd, IterationStart, StreamEvent, TextDelta
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import TransportAuth
from axio.types import StopReason, Usage

MODEL = ModelSpec("fixture-first", frozenset({Capability.text, Capability.tool_use}))
OTHER = ModelSpec("fixture-second", MODEL.capabilities)


@dataclass
class Transport:
    session: Any
    model: ModelSpec = MODEL
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry([MODEL, OTHER]))
    closed: bool = False

    def with_model(self, model: ModelSpec) -> Transport:
        return replace(self, model=model)

    async def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        yield IterationStart(0, model=self.model.id)
        yield TextDelta(0, f"fixture reply from {self.model.id}")
        yield IterationEnd(0, StopReason.end_turn, Usage(1, 2))


class Plugin:
    display_name = "Independent fixture"
    env_vars = ("AXIO_FIXTURE_API_KEY",)
    supports_temperature = False
    supports_max_tokens = False
    supports_debug = False
    supports_thinking = False
    auth: TransportAuth | None = None

    def __init__(self) -> None:
        self.opened: list[Transport] = []

    def has_credentials(self, config: Mapping[str, str] | None = None) -> bool:
        return bool(os.environ.get(self.env_vars[0]))

    @asynccontextmanager
    async def open(self, *, session: Any, **settings: Any) -> AsyncIterator[Transport]:
        transport = Transport(session)
        self.opened.append(transport)
        try:
            yield transport
        finally:
            transport.closed = True


plugin = Plugin()
