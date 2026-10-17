"""Provider failures and cancellation do not leak application resources."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from axio.instructions import Instruction
from axio.plugins import PluginCatalog
from axio.tool import Tool
from axio.transport import TransportCatalog

import axio_repl as repl


class Provider:
    def __init__(self, *, fail: bool = False, wait: asyncio.Event | None = None) -> None:
        self.fail = fail
        self.wait = wait
        self.started = False
        self.closed = 0
        self.refreshed = 0
        self.all_tools: list[Tool[Any]] = []

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        self.started = True
        if self.wait is not None:
            self.wait.set()
            await asyncio.Event().wait()
        if self.fail:
            raise ValueError("broken provider")

    async def close(self) -> None:
        self.closed += 1

    async def refresh(self, *, config: Any = None) -> None:
        self.refreshed += 1

    def get_instructions(self) -> Sequence[Instruction]:
        return ()


def configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, providers: dict[str, Any]) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--isolated"])
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    monkeypatch.setattr(repl, "discover_transports", lambda: TransportCatalog())
    monkeypatch.setattr(repl, "discover_tool_providers", lambda: PluginCatalog(entries=providers.copy()))
    monkeypatch.setattr(repl, "discover_tools", lambda: PluginCatalog())


@pytest.mark.asyncio
async def test_failure_does_not_block_independent_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken, healthy = Provider(fail=True), Provider()
    configure(monkeypatch, tmp_path, {"broken": broken, "custom-name": healthy})
    monkeypatch.setattr(repl, "_read_input", iter(["/skills reload", "hello", "/quit"]).__next__)
    await repl.main()
    assert broken.closed == healthy.closed == 1
    assert healthy.started and healthy.refreshed == 1
    assert "broken provider" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_cancel_during_provider_startup_closes_partial_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = asyncio.Event()
    healthy, waiting = Provider(), Provider(wait=started)
    configure(monkeypatch, tmp_path, {"healthy": healthy, "waiting": waiting})
    task = asyncio.create_task(repl.main())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert healthy.closed == waiting.closed == 1
