"""The TUI consumes the common discovery and factory contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from axio.models import Capability, ModelRegistry, ModelSpec

from axio_tui.transport_registry import TransportRegistry


async def test_failed_entry_remains_visible_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> object:
        raise ImportError("missing fixture dependency")

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda **kwargs: [SimpleNamespace(name="broken", load=broken)] if kwargs["group"] == "axio.transport" else [],
    )
    registry = TransportRegistry()
    await registry.init(AsyncMock(spec=aiohttp.ClientSession))
    assert registry.discovered == ["broken"] and not registry.available
    assert registry.errors == {"broken": "missing fixture dependency"}
    await registry.close()


async def test_provider_defaults_are_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = ModelSpec("preferred-chat", frozenset({Capability.text, Capability.tool_use}))

    class Provider:
        default_models = {"chat": chat.id}

        def __init__(self, **kwargs: object) -> None:
            self.models = ModelRegistry([ModelSpec("first"), chat])
            self.model = self.models.first()

        async def fetch_models(self) -> None:
            pass

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda **kwargs: (
            [SimpleNamespace(name="external", load=lambda: Provider)] if kwargs["group"] == "axio.transport" else []
        ),
    )
    registry = TransportRegistry()
    await registry.init(AsyncMock(spec=aiohttp.ClientSession))
    binding = registry.default_model("chat", Capability.text)
    assert binding is not None and binding.transport == "external" and binding.model is chat
    await registry.close()
