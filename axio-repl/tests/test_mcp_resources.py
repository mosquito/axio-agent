"""Expose all MCP resources to the model and preserve their server-defined URIs."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp_tiny_mcp import Registry
from aiohttp_tiny_mcp.testing import serving
from axio import HandlerError
from axio_tools_mcp.runtime import MCPRuntime, Session
from axio_tools_mcp.skills import SKILLS_CATALOG, SKILLS_ROOT

from .instruction_helpers import instruction_text
from .skills_server import INSTRUCTIONS, SKILL_URI, Nothing, skill_registry
from .test_mcp_skills import configurations

pytestmark = pytest.mark.asyncio


async def check_resources(runtime: MCPRuntime) -> None:
    tools = {tool.name: tool for tool in runtime.tools}
    catalog = json.loads(await tools["s__list_resources"]())
    uris = {item["uri"] for item in catalog["resources"]}
    assert {SKILLS_CATALOG, SKILL_URI, f"{SKILLS_ROOT}/planning/SKILL.md", "notes://guide"} <= uris
    assert catalog["resourceTemplates"][0]["uriTemplate"] == "notes://topics/{name}"
    assert "A general guide" in (await instruction_text(runtime, "resources"))
    assert "notes://topics/{name}" in (await instruction_text(runtime, "resources"))
    assert INSTRUCTIONS not in (await instruction_text(runtime, "resources"))
    results = await asyncio.gather(
        tools["s__read_resource"](uri=SKILL_URI),
        tools["s__read_skill"](uri=SKILL_URI),
        tools["s__read_resource"](uri=f"{SKILLS_ROOT}/planning/SKILL.md"),
        tools["s__read_resource"](uri="notes://topics/example"),
        tools["s__ping"](),
    )
    assert json.loads(results[0])["contents"][0]["uri"] == SKILL_URI
    assert json.loads(results[0])["contents"][0]["text"] == INSTRUCTIONS
    assert json.loads(results[1])["text"] == INSTRUCTIONS
    assert json.loads(results[2])["contents"][0]["uri"] == f"{SKILLS_ROOT}/planning/SKILL.md"
    assert json.loads(results[3])["contents"][0]["text"] == "Topic: example"
    assert results[4] == "pong"


async def test_http_resources_include_skill_duplicates_and_refresh(tmp_path: Path) -> None:
    registry = skill_registry([])
    runtime = MCPRuntime()
    async with serving(registry) as url:
        try:
            await runtime.reload(configurations(tmp_path, {"s": {"url": url}}), reserved=set())
            await check_resources(runtime)

            async def added(args: Nothing) -> str:
                return "New resource"

            registry.resource("notes://added")(added)
            tool = next(tool for tool in runtime.tools if tool.name == "s__list_resources")
            assert "notes://added" in await tool()
            assert "notes://added" in (await instruction_text(runtime, "resources"))
            await runtime.reload(configurations(tmp_path, {"s": {"enabled": False}}), reserved=set())
            assert not (await instruction_text(runtime, "resources")) and not runtime.tools
        finally:
            await runtime.close()


async def test_stdio_resources_share_the_tool_connection(tmp_path: Path) -> None:
    runtime = MCPRuntime()
    try:
        await runtime.reload(
            configurations(
                tmp_path,
                {"s": {"command": sys.executable, "args": [str(Path(__file__).with_name("skills_server.py"))]}},
            ),
            reserved=set(),
        )
        await check_resources(runtime)
    finally:
        await runtime.close()
    assert not (await instruction_text(runtime, "resources"))


async def test_failed_skill_uri_does_not_fall_back_to_compatibility_uri(tmp_path: Path) -> None:
    reads: list[str] = []
    runtime = MCPRuntime()
    async with serving(skill_registry(reads, direct_skills=False)) as url:
        try:
            await runtime.reload(configurations(tmp_path, {"s": {"url": url}}), reserved=set())
            tools = {tool.name: tool for tool in runtime.tools}
            for tool_name in ("s__read_skill", "s__read_resource"):
                with pytest.raises(HandlerError, match="rejected the resource request"):
                    await tools[tool_name](uri=SKILL_URI)
            assert reads == []
            assert runtime.servers[0].state == "connected"
            uri = f"{SKILLS_ROOT}/planning/SKILL.md"
            assert uri in (await instruction_text(runtime, "resources"))
            result = json.loads(await tools["s__read_resource"](uri=uri))
            assert result["contents"][0]["text"] == INSTRUCTIONS
            assert reads == ["SKILL.md"]
        finally:
            await runtime.close()


async def test_resource_only_server_does_not_require_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = Registry("resources-only", "1")

    async def guide(args: Nothing) -> str:
        return "Only resources"

    async def forbidden_list_tools(session: Session) -> None:
        pytest.fail("A resource-only server must not receive tools/list")

    registry.resource("notes://guide")(guide)
    monkeypatch.setattr(Session, "list_tools", forbidden_list_tools)
    runtime = MCPRuntime()
    async with serving(registry) as url:
        try:
            await runtime.reload(configurations(tmp_path, {"s": {"url": url}}), reserved=set())
            assert runtime.servers[0].state == "connected"
            tools = {tool.name: tool for tool in runtime.tools}
            assert set(tools) == {"s__list_resources", "s__read_resource"}
            result = json.loads(await tools["s__read_resource"](uri="notes://guide"))
            assert result["contents"][0]["text"] == "Only resources"
        finally:
            await runtime.close()
