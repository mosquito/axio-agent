"""Tests for REPL model selection helpers."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.transport import TransportCatalog, TransportEntry

import axio_repl as repl
from axio_repl import REPLState, _resolve_model_arg

_BASE_MODEL = ModelSpec(
    id="z-ai/glm-4.7",
    capabilities=frozenset({Capability.text, Capability.tool_use}),
    context_window=256_000,
    max_output_tokens=16_384,
)


class _VariantTransport:
    def __init__(self) -> None:
        self.model = _BASE_MODEL
        self.models = ModelRegistry([_BASE_MODEL])

    def resolve_model(self, model_id: str) -> ModelSpec:
        base_id, sep, _variant = model_id.rpartition(":")
        if sep and base_id in self.models:
            return replace(self.models[base_id], id=model_id)
        return self.models[model_id]


def test_resolve_model_arg_uses_transport_resolver() -> None:
    transport = _VariantTransport()

    spec = _resolve_model_arg(transport, "z-ai/glm-4.7:nitro")

    assert spec.id == "z-ai/glm-4.7:nitro"
    assert spec.context_window == _BASE_MODEL.context_window


def test_change_model_accepts_resolved_variant(capsys: pytest.CaptureFixture[str]) -> None:
    transport = _VariantTransport()
    state = REPLState(Path("/tmp/test-workspace"), transport)

    state.change_model("z-ai/glm-4.7:nitro")

    assert state.transport.model.id == "z-ai/glm-4.7:nitro"
    assert state.agent.transport is state.transport
    assert transport.model is _BASE_MODEL
    assert "z-ai/glm-4.7:nitro" in state.agent.system
    captured = capsys.readouterr()
    assert "Switched to" in captured.out


async def test_startup_resolves_variant_before_selecting_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _VariantTransport()
    entry = TransportEntry.from_target("fixture", lambda **settings: transport)
    monkeypatch.setattr(repl, "discover_transports", lambda: TransportCatalog({entry.name: entry}))
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["axio-repl", "--isolated", "--transport", "fixture", "--model", "z-ai/glm-4.7:nitro", "hello"]
    )
    prompt = AsyncMock()
    monkeypatch.setattr(repl, "run_prompt", prompt)

    await repl.main()

    prompt.assert_awaited_once()
    agent = prompt.call_args.args[0]
    assert agent.transport.model.id == "z-ai/glm-4.7:nitro"
    assert agent.transport.model.context_window == _BASE_MODEL.context_window
    assert transport.model is _BASE_MODEL
