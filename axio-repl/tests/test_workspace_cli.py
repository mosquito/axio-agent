from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from axio import Agent, MemoryContextStore
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.transport import TransportEntry

import axio_repl as repl


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    directory = root / "service"
    directory.mkdir(parents=True)
    (root / ".git").write_text("gitdir: elsewhere\n")
    (root / "AGENTS.md").write_text("project instructions")
    (directory / "AGENTS.md").write_text("service instructions")
    monkeypatch.chdir(directory)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return directory


@pytest.mark.asyncio
async def test_explain_needs_no_transport_or_history(
    cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--instructions-explain", "new.py"])
    forbidden = Mock(side_effect=AssertionError("must not start the REPL"))
    monkeypatch.setattr(repl, "_select_transport", forbidden)
    monkeypatch.setattr(repl, "setup_history", forbidden)
    monkeypatch.setattr(aiohttp, "ClientSession", forbidden)
    await repl.main()
    output = capsys.readouterr().out
    assert f"Working directory: {cwd}" in output
    assert f"Workspace root: {cwd.parent}" in output
    assert f"Target: {cwd / 'new.py'}" in output
    assert f"Scope: {cwd}" in output
    assert f"1. {cwd.parent / 'AGENTS.md'}" in output
    assert f"2. {cwd / 'AGENTS.md'}" in output
    assert "project instructions" not in output
    forbidden.assert_not_called()


@pytest.mark.asyncio
async def test_explain_other_workspace_and_isolated(
    cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    other = cwd.parent.parent / "other"
    other.mkdir()
    (other / "AGENTS.md").write_text("other instructions")
    explicit = cwd / "custom.md"
    explicit.write_text("session instructions")
    args = ["axio-repl", "--workspace-root", str(other), "--instructions-explain", str(other / "file.py")]
    monkeypatch.setattr(sys, "argv", args)
    await repl.main()
    output = capsys.readouterr().out
    assert f"1. {other / 'AGENTS.md'}" in output
    assert str(cwd / "AGENTS.md") not in output
    monkeypatch.setattr(sys, "argv", [*args, "--isolated", "--instructions-file", "custom.md"])
    await repl.main()
    output = capsys.readouterr().out
    assert f"1. {explicit}" in output
    assert str(other / "AGENTS.md") not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["limit", "encoding", "missing", "overlap"])
async def test_instruction_error_precedes_transport_selection(
    cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    args = ["axio-repl"]
    if bad == "limit":
        args += ["--instructions-max-bytes", "1"]
    elif bad == "encoding":
        (cwd / "AGENTS.md").write_bytes(b"\xff")
    elif bad == "missing":
        args += ["--instructions-file", "missing.md"]
    else:
        args += ["--workspace-root", str(cwd)]
    monkeypatch.setattr(sys, "argv", args)
    forbidden = Mock(side_effect=AssertionError("must not select a transport"))
    monkeypatch.setattr(repl, "_select_transport", forbidden)
    with pytest.raises(SystemExit) as error:
        await repl.main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    forbidden.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
async def test_startup_and_model_switch_preserve_scoped_snapshot(
    cwd: Path, monkeypatch: pytest.MonkeyPatch, interactive: bool
) -> None:
    models = ModelRegistry(
        ModelSpec(name, frozenset({Capability.text, Capability.tool_use})) for name in ("first", "second")
    )
    transport = SimpleNamespace(model=models["first"], models=models, fetch_models=AsyncMock())
    monkeypatch.setattr(
        repl,
        "_select_transport",
        AsyncMock(return_value=TransportEntry.from_target("fixture", lambda **kwargs: transport)),
    )
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    monkeypatch.setattr(sys, "argv", ["axio-repl", *([] if interactive else ["hello"])])
    inputs = iter(["hello", "/model second", "hello", "/quit"])
    monkeypatch.setattr(repl, "_read_input", lambda: next(inputs))
    prompts: list[str] = []

    async def run(agent: Agent, context: MemoryContextStore, text: str) -> None:
        await agent._prepare_instructions()
        assert text == "hello"
        assert Path.cwd() == cwd
        assert f"Current directory: {cwd}" in agent._effective_system()
        assert f"Workspace root: {cwd.parent}" in agent._effective_system()
        assert 'scope: "' + str(cwd) + '"' in agent._effective_system()
        assert agent._effective_system().count("project instructions") == 1
        assert agent._effective_system().count("service instructions") == 1
        prompts.append(agent._effective_system())

    monkeypatch.setattr(repl, "run_prompt", run)
    await repl.main()
    assert len(prompts) == (2 if interactive else 1)
    if interactive:
        assert "Model: first" in prompts[0]
        assert "Model: second" in prompts[1]
        assert (
            prompts[0].split('Loaded instruction "local:scoped-instructions":')[1]
            == prompts[1].split('Loaded instruction "local:scoped-instructions":')[1]
        )


@pytest.mark.asyncio
async def test_model_switch_keeps_dynamic_instructions_and_refreshes_changes(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import AsyncIterator
    from typing import Any

    from axio import Tool
    from axio.events import StreamEvent
    from axio.messages import Message
    from axio.testing import StubTransport, make_text_response, make_tool_use_response

    nested = cwd / "nested"
    nested.mkdir()
    source = nested / "AGENTS.md"
    source.write_text("FIRST_DYNAMIC_RULE")
    (nested / "data.txt").write_text("data")
    models = ModelRegistry(ModelSpec(name, frozenset({Capability.tool_use})) for name in ("first", "second"))
    observed: list[tuple[str, str]] = []

    class Transport(StubTransport):
        def __init__(self) -> None:
            super().__init__(
                [
                    make_tool_use_response("read_file", "read1", {"filename": "nested/data.txt"}),
                    make_tool_use_response("read_file", "read2", {"filename": "nested/data.txt"}),
                    make_text_response(),
                    make_text_response(),
                ]
            )
            self.model = models["first"]
            self.models = models

        async def fetch_models(self) -> None:
            pass

        def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
            observed.append((self.model.id, system))
            return super().stream(messages, tools, system)

    transport = Transport()
    monkeypatch.setattr(
        repl,
        "_select_transport",
        AsyncMock(return_value=TransportEntry.from_target("fixture", lambda **kwargs: transport)),
    )
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    monkeypatch.setattr(sys, "argv", ["axio-repl"])
    inputs = iter(["first request", "/model second", "second request", "/quit"])

    def read_input() -> str:
        text = next(inputs)
        if text == "/model second":
            source.write_text("UPDATED_DYNAMIC_RULE")
        return text

    monkeypatch.setattr(repl, "_read_input", read_input)
    await repl.main()
    assert "FIRST_DYNAMIC_RULE" not in observed[0][1]
    assert "FIRST_DYNAMIC_RULE" in observed[1][1]
    model, system = observed[-1]
    assert model == "second"
    assert "UPDATED_DYNAMIC_RULE" in system and "FIRST_DYNAMIC_RULE" not in system
    assert f'scope: "{nested}"' in system
