from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp_tiny_mcp.testing import serving
from axio.instructions import Instruction
from axio.plugins import ToolProvider, collect_tools
from instruction_server import INSTRUCTIONS, SKILL_URI, skill_registry

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.provider import MCPToolProvider


def configurations(tmp_path: Path, servers: dict[str, dict[str, Any]]) -> tuple[MCPServerConfig, ...]:
    return tuple(
        MCPServerConfig.from_dict(name, {**data, "cwd": str(tmp_path)} if "command" in data else data)
        for name, data in servers.items()
    )


def skill_instruction(provider: MCPToolProvider) -> Instruction:
    return next(item for item in provider.get_instructions() if item.source == SKILL_URI)


async def test_http_instruction_load_tools_resources_and_reload(tmp_path: Path) -> None:
    reads: list[str] = []
    provider = MCPToolProvider()
    contract: ToolProvider = provider
    async with serving(skill_registry(reads)) as url:
        await contract.init(configurations(tmp_path, {"s": {"url": url}}))
        try:
            instruction = skill_instruction(provider)
            assert reads == [] and INSTRUCTIONS not in instruction.summary
            assert skill_instruction(provider) is instruction
            loaded = await instruction.load()
            assert INSTRUCTIONS in loaded.text
            assert reads == ["SKILL.md"]
            tools = {tool.name: tool for tool in loaded.tools}
            assert await tools["s__ping"]() == "pong"
            resource = json.loads(await tools["s__read_resource"](uri=SKILL_URI))
            assert resource["contents"][0]["uri"] == SKILL_URI
            assert collect_tools([contract]) == list(loaded.tools)
            assert all(tool.permission_identity for tool in loaded.tools)
            await provider.reload(configurations(tmp_path, {"s": {"enabled": False}}))
            assert not provider.all_tools and not provider.get_instructions()
            with pytest.raises(RuntimeError, match="closed connection"):
                await instruction.load()
        finally:
            await contract.close()


async def test_stdio_instruction_uses_live_session_and_close(tmp_path: Path) -> None:
    provider = MCPToolProvider()
    await provider.init(
        configurations(
            tmp_path,
            {
                "s": {
                    "command": sys.executable,
                    "args": [str(Path(__file__).with_name("instruction_server.py"))],
                }
            },
        )
    )
    instruction = skill_instruction(provider)
    session = provider.runtime.servers[0].session
    assert session is not None and session.process is not None
    process = session.process
    try:
        loaded = await instruction.load()
        tool = next(tool for tool in loaded.tools if tool.name == "s__ping")
        assert await tool() == "pong"
    finally:
        await provider.close()
    assert process.returncode is not None
    with pytest.raises(RuntimeError, match="closed connection"):
        await instruction.load()


async def test_failed_load_drops_unusable_server_and_preserves_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = MCPToolProvider()
    async with serving(skill_registry([])) as url:
        await provider.init(configurations(tmp_path, {"s": {"url": url}, "good": {"url": url}}))
        try:
            instruction = skill_instruction(provider)
            session = provider.runtime.servers[0].session
            assert session is not None and session.client is not None
            monkeypatch.setattr(session.client, "read_resource", AsyncMock(side_effect=asyncio.CancelledError))
            with pytest.raises(asyncio.CancelledError):
                await instruction.load()
            assert provider.runtime.servers[0].state == "error"
            assert all(tool.name.startswith("good__") for tool in provider.all_tools)
            assert all(item.id.startswith("mcp:good:") for item in provider.get_instructions())
        finally:
            await provider.close()


def test_provider_import_does_not_require_local_or_ui() -> None:
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'axio_tools_local', 'axio_tui', 'textual'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from axio_tools_mcp.provider import MCPToolProvider
assert MCPToolProvider().all_tools == []
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
