"""Exercise skill discovery, lazy reads and REPL lifecycle through MCP."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp_tiny_mcp.testing import serving
from axio import Agent, MemoryContextStore
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.testing import StubTransport
from axio.transport import TransportCatalog, TransportEntry
from axio_tools_local.paths import LocalPaths as Workspace
from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.runtime import MCPRuntime

import axio_repl as repl
from axio_repl.configuration import ConfigError, connection_configs, load_mcp_config
from axio_repl.presentation import describe_mcp_skills

from .instruction_helpers import instruction_text
from .skills_server import INSTRUCTIONS, REFERENCE, SKILL_URI, skill_registry

pytestmark = pytest.mark.asyncio


def configurations(root: Path, entries: dict[str, Any]) -> tuple[MCPServerConfig, ...]:
    path = root / ".mcp.json"
    path.write_text(json.dumps({"mcpServers": entries}))
    return connection_configs(load_mcp_config(Workspace.discover(root, root), files=[path], isolated=True))


async def test_http_lazy_reads_namespace_reload_and_cleanup(tmp_path: Path) -> None:
    reads: list[str] = []
    runtime = MCPRuntime()
    async with serving(skill_registry(reads)) as url:
        projects = configurations(tmp_path, {name: {"url": url} for name in ("first", "second")})
        try:
            await runtime.reload(projects, reserved=set())
            assert reads == []
            assert "first:planning" in describe_mcp_skills(runtime)
            assert "second:planning" in describe_mcp_skills(runtime)
            assert INSTRUCTIONS not in (await instruction_text(runtime, "skills"))
            tools = {tool.name: tool for tool in runtime.tools}
            for name in ("first", "second"):
                result = json.loads(await tools[f"{name}__read_skill"](uri=SKILL_URI))
                assert result["server"] == name
                assert result["text"] == INSTRUCTIONS
                result = json.loads(await tools[f"{name}__read_skill"](uri=SKILL_URI, path="references/details.md"))
                assert result["text"] == REFERENCE
            assert reads == ["SKILL.md", "references/details.md"] * 2
            await runtime.reload(configurations(tmp_path, {"first": {"enabled": False}}), reserved=set())
            assert (await instruction_text(runtime, "skills")) == ""
            assert describe_mcp_skills(runtime) == "No MCP skills available."
            assert not runtime.tools
        finally:
            await runtime.close()


async def test_stdio_skill_and_tool_requests_share_connection(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    try:
        await runtime.reload(
            configurations(
                tmp_path,
                {"s": {"command": sys.executable, "args": [str(Path(__file__).with_name("skills_server.py"))]}},
            ),
            reserved=set(),
        )
        tools = {tool.name: tool for tool in runtime.tools}
        results = await asyncio.gather(
            tools["s__read_skill"](uri=SKILL_URI),
            tools["s__ping"](),
            tools["s__read_skill"](uri=SKILL_URI, path="references/details.md"),
        )
        assert json.loads(results[0])["text"] == INSTRUCTIONS
        assert results[1] == "pong"
        assert json.loads(results[2])["text"] == REFERENCE
        session = runtime.servers[0].session
        assert session is not None and session.process is not None
        process = session.process
    finally:
        await runtime.close()
    assert process.returncode is not None


async def test_bad_skill_catalog_keeps_server_tools(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runtime = MCPRuntime()
    async with serving(skill_registry([], broken=True)) as url:
        try:
            await runtime.reload(configurations(tmp_path, {"s": {"url": url}}), reserved=set())
            assert runtime.servers[0].state == "connected"
            assert {tool.name for tool in runtime.tools} == {"s__ping", "s__list_resources", "s__read_resource"}
            assert await runtime.tools[0]() == "pong"
            assert "skill discovery failed" in describe_mcp_skills(runtime)
            assert "skill discovery failed" in capsys.readouterr().err
            assert "notes://guide" in (await instruction_text(runtime, "resources"))
        finally:
            await runtime.close()


@pytest.mark.parametrize("name", ["read_skill", "read_resource", "list_resources"])
async def test_reader_name_collision_closes_connections(tmp_path: Path, name: str) -> None:
    runtime = MCPRuntime()
    async with serving(skill_registry([])) as url:
        with pytest.raises(ConfigError, match="Duplicate MCP tool"):
            await runtime.reload(configurations(tmp_path, {"s": {"url": url}}), reserved={f"s__{name}"})
        assert not runtime.tools
        assert not (await instruction_text(runtime, "skills"))
        assert not (await instruction_text(runtime, "resources"))


async def test_resource_cancellation_removes_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = MCPRuntime()
    async with serving(skill_registry([])) as url:
        try:
            await runtime.reload(configurations(tmp_path, {"s": {"url": url}}), reserved=set())
            session = runtime.servers[0].session
            assert session is not None and session.client is not None
            monkeypatch.setattr(session.client, "read_resource", AsyncMock(side_effect=asyncio.CancelledError))
            with pytest.raises(asyncio.CancelledError):
                await session.read_resource(SKILL_URI)
            assert runtime.servers[0].state == "error"
            assert not runtime.tools
            assert not (await instruction_text(runtime, "skills"))
        finally:
            await runtime.close()


@pytest.mark.parametrize("single", [False, True])
async def test_repl_skills_survive_model_change_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], single: bool
) -> None:
    reads: list[str] = []
    async with serving(skill_registry(reads)) as url:
        configurations(tmp_path, {"s": {"url": url}})
        local = tmp_path / "skills/local-plan/SKILL.md"
        local.parent.mkdir(parents=True)
        local.write_text("---\nname: local-plan\ndescription: Plan locally\n---\nLocal workflow.\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setattr(
            sys,
            "argv",
            ["axio-repl", "--transport", "fixture", "--skills-dir", "skills", *(["plan"] if single else [])],
        )
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
        inputs = iter(["/skills", "/model two", "plan", "/mcp reload", "/skills reload", "plan", "/quit"])
        monkeypatch.setattr(repl, "_read_input", lambda: next(inputs))

        async def run_prompt(agent: Agent, ctx: MemoryContextStore, text: str) -> None:
            await agent._prepare_instructions()
            prompt = agent._effective_system()
            assert "s:planning" in prompt and "local-plan" in prompt
            assert SKILL_URI in prompt and INSTRUCTIONS not in prompt
            assert "Local workflow." not in prompt
            resources = await agent.load_instruction("mcp:s:resources")
            assert "notes://guide" in resources.text and "notes://topics/{name}" in resources.text
            loaded = await agent.load_instruction(f"mcp:s:{SKILL_URI}")
            assert INSTRUCTIONS in loaded.text
            local_instruction = next(i for i in agent._catalog() if i.source == str(local))
            assert "Local workflow." in (await agent.load_instruction(local_instruction.id)).text
            await agent._prepare_instructions()
            resource_tool = next(tool for tool in agent.get_tools() if tool.name == "s__read_resource")
            result = json.loads(await resource_tool(uri="notes://guide"))
            assert result["contents"][0]["text"] == "A plain MCP resource."

        monkeypatch.setattr(repl, "run_prompt", run_prompt)
        await repl.main()
    assert reads == ["SKILL.md"] * (1 if single else 2)
    if not single:
        assert "s:planning" in capsys.readouterr().out
