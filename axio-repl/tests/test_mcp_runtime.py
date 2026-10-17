"""Run MCP discovery and calls through the REPL, with real local transports."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp_tiny_mcp import Registry
from aiohttp_tiny_mcp.testing import serving
from axio import Agent, MemoryContextStore
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.testing import StubTransport
from axio.transport import TransportCatalog, TransportEntry
from axio_tools_local.paths import LocalPaths as Workspace
from axio_tools_local.provider import local_tools
from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.runtime import MCPRuntime, Session
from pydantic import BaseModel

import axio_repl as repl
from axio_repl.configuration import ConfigError, connection_configs, load_mcp_config
from axio_repl.presentation import describe_mcp

pytestmark = pytest.mark.asyncio
STDIO = Path(__file__).resolve().parents[2] / "axio-tools-mcp/tests/stdio_server.py"


class Message(BaseModel):
    message: str


def registry() -> Registry:
    result = Registry("runtime-test", "1")

    @result.tool
    async def echo(args: Message) -> str:
        """Return the supplied message."""
        return args.message

    return result


def configurations(root: Path, entries: dict[str, Any]) -> tuple[MCPServerConfig, ...]:
    path = root / ".mcp.json"
    path.write_text(json.dumps({"mcpServers": entries}))
    return connection_configs(load_mcp_config(Workspace.discover(root, root), files=[path], isolated=True))


def stdio() -> dict[str, Any]:
    return {"type": "stdio", "command": sys.executable, "args": [str(STDIO)]}


async def test_stdio_round_trip_reload_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXIO_MCP_TEST_TOKEN", "must-not-inherit")
    project = configurations(tmp_path, {"local": stdio(), "off": {"enabled": False}})
    runtime = MCPRuntime()
    try:
        await runtime.reload(project, reserved=set())
        session = runtime.servers[0].session
        assert session is not None and session.process is not None
        first = session.process
        assert session.config.cwd == str(tmp_path)
        tools = {tool.name: tool for tool in runtime.tools}
        assert await tools["local__echo"](message="hello") == "hello"
        assert await tools["local__token"]() == ""
        assert "'local'  3  connected" in describe_mcp(runtime)
        assert "'off'  0  disabled" in describe_mcp(runtime)
        await runtime.reload(project, reserved=set())
        assert first.returncode is not None
        session = runtime.servers[0].session
        assert session is not None and session.process is not None
        second = session.process
    finally:
        await runtime.close()
    assert second.returncode is not None and not runtime.tools


async def test_http_discovery_and_tool_call(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    async with serving(registry()) as url:
        try:
            await runtime.reload(configurations(tmp_path, {"remote": {"type": "http", "url": url}}), reserved=set())
            assert len(runtime.tools) == 1
            assert await runtime.tools[0](message="http works") == "http works"
            session = runtime.servers[0].session
            assert session is not None and session.http is not None
            http = session.http
        finally:
            await runtime.close()
        assert http.closed


async def test_stdio_tool_timeout_invalidates_session(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    try:
        await runtime.reload(configurations(tmp_path, {"s": stdio()}), reserved=set())
        server = runtime.servers[0]
        session = server.session
        assert session is not None
        session.config = replace(session.config, timeout=0.1)
        # The outer deadline catches a missing request timeout without waiting for the server.
        async with asyncio.timeout(2):
            with pytest.raises(RuntimeError, match="tool request timed out after 0.1s.*reload"):
                await session.call_tool("echo", {"message": "late", "delay": 10})
        assert session.failed == "TimeoutError"
        assert server.state == "error" and not runtime.tools
        with pytest.raises(RuntimeError, match="unavailable"):
            await session.call_tool("echo", {"message": "next"})
    finally:
        await runtime.close()


async def test_stdio_rpc_error_keeps_session_usable(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    try:
        await runtime.reload(configurations(tmp_path, {"s": stdio()}), reserved=set())
        server = runtime.servers[0]
        session = server.session
        assert session is not None
        with pytest.raises(RuntimeError, match="rejected the tool request"):
            await session.call_tool("missing", {})
        assert session.failed is None and server.state == "connected"
        tools = {tool.name: tool for tool in runtime.tools}
        assert await tools["s__echo"](message="still works") == "still works"
    finally:
        await runtime.close()


async def test_partial_failure_keeps_healthy_tools(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = {"type": "stdio", "command": sys.executable, "args": ["-c", "raise RuntimeError('secret')"]}
    runtime = MCPRuntime()
    try:
        await runtime.reload(configurations(tmp_path, {"broken": broken, "good": stdio()}), reserved=set())
        assert runtime.servers[0].state == "error"
        assert len(runtime.tools) == 3
        assert "'broken'  0  error" in describe_mcp(runtime)
        assert "secret" not in capsys.readouterr().err
    finally:
        await runtime.close()


async def test_resolution_failure_does_not_block_other_servers(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    try:
        await runtime.reload(
            configurations(
                tmp_path,
                {
                    "missing": {"type": "stdio", "command": "/nonexistent/mcp-server"},
                    "good": stdio(),
                },
            ),
            reserved=set(),
        )
        assert runtime.servers[0].state == "error"
        assert len(runtime.tools) == 3
    finally:
        await runtime.close()


async def test_changed_configuration_is_reloaded_without_confirmation(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    try:
        await runtime.reload(configurations(tmp_path, {"s": stdio()}), reserved=set())
        changed = stdio() | {"env": {"AXIO_MCP_TEST_TOKEN": "new"}}
        await runtime.reload(configurations(tmp_path, {"s": changed}), reserved=set())
        tools = {tool.name: tool for tool in runtime.tools}
        assert await tools["s__token"]() == "new"
    finally:
        await runtime.close()


async def test_collision_closes_all_sessions(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    with pytest.raises(ConfigError, match="Duplicate MCP tool"):
        await runtime.reload(configurations(tmp_path, {"s": stdio()}), reserved={"s__echo"})
    assert not runtime.tools
    assert all((server.session is None or server.session.process is None) for server in runtime.servers)


async def test_cancel_connect_closes_partial_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    original = Session.connect

    async def connect(session: Session) -> None:
        await original(session)
        if session.config.name == "second":
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(Session, "connect", connect)
    runtime = MCPRuntime()
    task = asyncio.create_task(
        runtime.reload(configurations(tmp_path, {"first": stdio(), "second": stdio()}), reserved=set())
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime.tools
    assert all((server.session is None or server.session.process is None) for server in runtime.servers)


async def test_multi_root_names_and_unexpected_process_exit(tmp_path: Path) -> None:
    first = configurations(tmp_path, {"s": stdio()})[0]
    other = tmp_path / "other"
    other.mkdir()
    second = replace(first, name="w2__s", cwd=str(other))
    first = replace(first, name="w1__s")
    runtime = MCPRuntime()
    try:
        await runtime.reload((first, second), reserved=set())
        assert {tool.name for tool in runtime.tools} >= {"w1__s__echo", "w2__s__echo"}
        session = runtime.servers[0].session
        assert session is not None and session.process is not None
        session.process.kill()
        await session.process.wait()
        assert runtime.servers[0].state == "error"
        assert len(runtime.tools) == 3
    finally:
        await runtime.close()


async def test_main_loads_project_mcp_and_keeps_tools_across_model_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configurations(tmp_path, {"s": stdio()})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--transport", "fixture"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(repl, "setup_history", lambda: None)
    models = ModelRegistry([ModelSpec(name, frozenset({Capability.tool_use})) for name in ("one", "two")])

    class Transport(StubTransport):
        def __init__(self, *, session: aiohttp.ClientSession) -> None:
            super().__init__([])
            self.model = models["one"]
            self.models = models

        async def fetch_models(self) -> None:
            pass

        def with_model(self, model: ModelSpec) -> Transport:
            self.model = model
            return self

    monkeypatch.setattr(
        repl,
        "discover_transports",
        lambda: TransportCatalog({"fixture": TransportEntry.from_target("fixture", Transport)}),
    )
    inputs = iter(["/mcp", "/model two", "call", "/mcp reload", "call", "/quit"])
    monkeypatch.setattr(repl, "_read_input", lambda: next(inputs))
    calls: list[str] = []

    async def run_prompt(agent: Agent, ctx: MemoryContextStore, text: str) -> None:
        tools = {tool.name: tool for tool in agent.get_tools()}
        assert set(tool.name for tool in local_tools(Path.cwd())) <= tools.keys()
        calls.append(await tools["s__echo"](message="loaded"))
        assert "s__echo" in agent.system

    monkeypatch.setattr(repl, "run_prompt", run_prompt)
    await repl.main()
    assert calls == ["loaded", "loaded"]
    output = capsys.readouterr().out
    assert "'s'  3  connected" in output
    assert "Active: 1; tools: 3" in output


@pytest.mark.parametrize("phase", ["connect", "list_tools"])
async def test_startup_timeout_is_clear_and_does_not_block_healthy_server(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = getattr(Session, phase)

    async def stalled(session: Session) -> Any:
        if session.config.name == "slow":
            await asyncio.Event().wait()
        return await original(session)

    monkeypatch.setattr(Session, phase, stalled)
    runtime = MCPRuntime()
    try:
        async with asyncio.timeout(5):
            await runtime.reload(
                configurations(tmp_path, {"slow": stdio() | {"timeout": 0.1}, "good": stdio()}), reserved=set()
            )
        failed = runtime.servers[0]
        assert failed.state == "error"
        assert failed.error is not None and "timed out after 0.1s" in failed.error
        assert failed.session is None or failed.session.process is None
        assert len(runtime.tools) == 3
        output = capsys.readouterr().err
        assert "MCP 'slow': initializing (timeout 0.1s)" in output
        assert "use /mcp reload" in output
        monkeypatch.setattr(Session, phase, original)
        await runtime.reload(configurations(tmp_path, {"slow": stdio()}), reserved=set())
        assert runtime.servers[0].state == "connected"
    finally:
        await runtime.close()
