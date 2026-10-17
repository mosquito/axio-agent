"""Model selection errors must preserve the catalog and current session."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.transport import TransportCatalog, TransportEntry
from axio_tools_local.paths import LocalPaths as Workspace

import axio_repl as repl


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider", ["openai", "nebius", "openrouter", "anthropic", "google", "google-vertex", "codex", "plugin"]
)
@pytest.mark.parametrize("model_ids", [("api-first", "api-second"), ()])
async def test_unknown_startup_model_lists_fetched_catalog(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], provider: str, model_ids: tuple[str, ...]
) -> None:
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--isolated", "--transport", provider, "--model", "missing"])
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    transport = SimpleNamespace(models=ModelRegistry([ModelSpec("stale")]), model=ModelSpec("stale"))

    async def fetch_models() -> None:
        transport.models = ModelRegistry(ModelSpec(name) for name in model_ids)

    transport.fetch_models = AsyncMock(side_effect=fetch_models)
    sessions: list[aiohttp.ClientSession] = []

    def factory(*, session: aiohttp.ClientSession) -> SimpleNamespace:
        sessions.append(session)
        return transport

    monkeypatch.setattr(
        repl,
        "discover_transports",
        lambda: TransportCatalog({provider: TransportEntry.from_target(provider, factory)}),
    )
    prompt = AsyncMock()
    monkeypatch.setattr(repl, "run_prompt", prompt)
    with pytest.raises(SystemExit) as error:
        await repl.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "Unknown model 'missing'" in captured.err
    assert "Available models:" in captured.err
    for name in model_ids or ("(none)",):
        assert f"  {name}" in captured.err
    assert "stale" not in captured.err and "Traceback" not in captured.err
    assert "REPL ready" not in captured.out
    transport.fetch_models.assert_awaited_once()
    prompt.assert_not_awaited()
    assert sessions[0].closed


@pytest.mark.parametrize("query", ["missing", "cur"])
def test_interactive_invalid_model_keeps_current_model(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, query: str
) -> None:
    current = ModelSpec("current", frozenset({Capability.text, Capability.tool_use}))
    transport = SimpleNamespace(model=current, models=ModelRegistry([current, ModelSpec("current-alternative")]))
    workspace = Workspace(tmp_path, (tmp_path,))
    state = repl.REPLState(workspace.cwd, transport)
    original_system = state.agent.system
    state.change_model(query)
    assert state.transport is transport and state.agent.transport is transport
    assert transport.model is current and state.agent.system == original_system
    output = capsys.readouterr()
    if query == "missing":
        assert "No model matching 'missing'.\nAvailable models:\n  current\n  current-alternative" in output.err
    else:
        assert "Ambiguous — matches: current, current-alternative" in output.out
