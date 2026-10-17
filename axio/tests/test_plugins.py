"""Shared discovery, tool ownership and provider cleanup contracts."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from axio.instructions import Instruction
from axio.plugins import (
    PluginError,
    collect_tools,
    discover_guards,
    discover_selectors,
    discover_tool_providers,
    discover_tools,
    open_tool_providers,
)
from axio.testing import make_echo_tool
from axio.tool import Tool


class Provider:
    def __init__(self, name: str, calls: list[str], *, fail: bool = False, wait: asyncio.Event | None = None) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail
        self.wait = wait
        self.all_tools: list[Tool[Any]] = []

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        self.calls.append(f"open:{self.name}")
        if self.wait is not None:
            self.wait.set()
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("startup failed")

    def get_instructions(self) -> Sequence[Instruction]:
        return ()

    async def close(self) -> None:
        self.calls.append(f"close:{self.name}")


def test_failed_first_declaration_reserves_name() -> None:
    points = [
        EntryPoint(name="echo", value="missing_axio_fixture:handler", group="axio.tools"),
        EntryPoint(name="echo", value="axio.testing:_msg_input", group="axio.tools"),
        EntryPoint(name="other", value="axio.testing:_msg_input", group="axio.tools"),
    ]
    catalog = discover_tools(points=points)
    assert list(catalog.entries) == ["other"]
    assert catalog.sources["echo"].value == "missing_axio_fixture:handler"
    with pytest.raises(PluginError, match="echo.*failed to load"):
        catalog.require("echo")
    with pytest.raises(PluginError, match="Available: other"):
        catalog.require("absent")


def test_group_adapters_validate_their_own_contracts() -> None:
    point = EntryPoint(name="wrong", value="builtins:str", group="axio.tools")
    assert "wrong" in discover_tools(points=[point]).errors
    assert "wrong" in discover_guards(points=[point]).errors
    assert "wrong" in discover_selectors(points=[point]).errors
    assert "wrong" in discover_tool_providers(points=[point]).errors
    guards = discover_guards(
        points=[EntryPoint(name="deny", value="axio.permission:DenyAllGuard", group="axio.guards")]
    )
    assert "deny" in guards.entries


async def test_cleanup_on_normal_exit_and_body_failure() -> None:
    calls: list[str] = []
    providers = [Provider("first", calls), Provider("second", calls)]
    async with open_tool_providers(providers) as opened:
        assert opened == tuple(providers)
    assert calls == ["open:first", "open:second", "close:second", "close:first"]
    calls.clear()
    with pytest.raises(ValueError, match="body"):
        async with open_tool_providers(providers):
            raise ValueError("body")
    assert calls == ["open:first", "open:second", "close:second", "close:first"]


async def test_partial_startup_closes_failing_provider_and_prior_providers() -> None:
    calls: list[str] = []
    providers = [Provider("first", calls), Provider("failed", calls, fail=True), Provider("last", calls)]
    with pytest.raises(RuntimeError, match="startup"):
        async with open_tool_providers(providers):
            pytest.fail("Failed startup must not yield")
    assert calls == ["open:first", "open:failed", "close:failed", "close:first"]


async def test_cancelled_startup_closes_providers() -> None:
    calls: list[str] = []
    started = asyncio.Event()

    async def run() -> None:
        async with open_tool_providers([Provider("first", calls), Provider("waiting", calls, wait=started)]):
            pytest.fail("Cancelled startup must not yield")

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == ["open:first", "open:waiting", "close:waiting", "close:first"]


async def test_conflicting_tools_close_providers() -> None:
    calls: list[str] = []
    first, second = Provider("first", calls), Provider("second", calls)
    first.all_tools = [make_echo_tool()]
    second.all_tools = [make_echo_tool()]
    with pytest.raises(PluginError, match="Conflicting tool name.*echo"):
        async with open_tool_providers([first, second]):
            pytest.fail("Tool conflicts must not be hidden")
    assert calls[-2:] == ["close:second", "close:first"]
    second.all_tools = first.all_tools
    assert collect_tools([first, second]) == first.all_tools
    second.all_tools = []
    first.all_tools = []
    assert collect_tools([first, second]) == []


async def test_real_entry_point_loads_provider_and_instruction_without_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real importlib distribution discovery, without replacing entry_points."""
    module = tmp_path / "fixture_axio_provider.py"
    module.write_text(
        "from axio.instructions import Instruction, LoadedInstruction\n"
        "from axio.testing import make_echo_tool\n"
        "class Rules(Instruction):\n"
        "    async def load(self):\n"
        "        return LoadedInstruction('Repeat the supplied message.', (make_echo_tool(),))\n"
        "class Provider:\n"
        "    def __init__(self):\n"
        "        self.opened = self.closed = False\n"
        "        self.all_tools = []\n"
        "    async def init(self, config=None, global_config=None):\n"
        "        self.opened = True\n"
        "    def get_instructions(self):\n"
        "        return (Rules('fixture:rules', 'Repeat a message'),)\n"
        "    async def close(self):\n"
        "        self.closed = True\n",
        encoding="utf-8",
    )
    dist = tmp_path / "fixture_axio_provider-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: fixture-axio-provider\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text("[axio.tools.providers]\nfixture = fixture_axio_provider:Provider\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    catalog = discover_tool_providers()
    provider = catalog.require("fixture")
    assert catalog.sources["fixture"].distribution == "fixture-axio-provider"
    assert not getattr(provider, "opened")
    async with open_tool_providers([provider]):
        instruction = provider.get_instructions()[0]
        assert instruction.id == "fixture:rules"
        loaded = await instruction.load()
        assert "hello" in await loaded.tools[0](msg="hello")
    assert getattr(provider, "closed")
    script = textwrap.dedent(
        """
        import asyncio
        import importlib.abc
        import sys

        class NoOptionalDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'textual', 'axio_tui', 'aiohttp', 'yaml', 'axio_tools_mcp'}:
                    raise ImportError('Optional dependency blocked: ' + fullname)

        sys.meta_path.insert(0, NoOptionalDependencies())
        sys.path.insert(0, sys.argv[1])
        from axio.plugins import discover_tool_providers, open_tool_providers

        async def main():
            provider = discover_tool_providers().require('fixture')
            async with open_tool_providers([provider]):
                loaded = await provider.get_instructions()[0].load()
                assert 'hello' in await loaded.tools[0](msg='hello')
            assert provider.closed

        asyncio.run(main())
        """
    )
    result = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_failed_provider_does_not_expose_uninitialized_plain_handlers() -> None:
    from dataclasses import replace

    from axio.plugins import PluginCatalog, PluginSource, ToolProvider, standalone_tools

    unsafe, independent = replace(make_echo_tool(), name="local"), replace(make_echo_tool(), name="independent")
    catalog = PluginCatalog(
        entries={"local": unsafe, "independent": independent},
        sources={
            "local": PluginSource("axio.tools", "failed-distribution", "local"),
            "independent": PluginSource("axio.tools", "other-distribution", "independent"),
        },
    )
    providers: PluginCatalog[ToolProvider] = PluginCatalog(
        errors={"provider": "initialization failed"},
        sources={"provider": PluginSource("axio.tools.providers", "failed-distribution", "provider")},
    )
    assert standalone_tools(catalog, providers) == [independent]
