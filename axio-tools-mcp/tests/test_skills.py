"""Validate compatibility catalogs and verify exact skill content bytes."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest
from aiohttp_tiny_mcp.protocol.models import ReadResourceResult, ResourceDef, TextResourceContents

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.session import MCPSession
from axio_tools_mcp.skills import SKILLS_CATALOG, SKILLS_LIST, SKILLS_ROOT, SkillCatalog

URI = f"{SKILLS_ROOT}/plan/SKILL.md"
TEXT = "---\nname: plan\ndescription: Plan work\n---\nKeep constraints. Условия.\n"


def entry(uri: str = URI) -> dict[str, Any]:
    return {
        "uri": uri,
        "frontmatter": {"name": "plan", "description": "Plan work"},
        "resources": [
            {"uri": uri, "size": len(TEXT.encode()), "digest": "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest()}
        ],
    }


def result(uri: str, text: str) -> ReadResourceResult:
    return ReadResourceResult(contents=[TextResourceContents(uri=uri, text=text)])


async def test_discovery_pages_and_duplicate_names(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    session.server_capabilities = {"resources": {}}
    resources = AsyncMock(return_value=[ResourceDef(uri=SKILLS_LIST, name="skills/list")])
    second_uri = f"{SKILLS_ROOT}/another/plan/SKILL.md"
    next_uri = SKILLS_LIST + "?params=" + quote(json.dumps({"cursor": "next &/?"}), safe="")
    read = AsyncMock(
        side_effect=[
            result(SKILLS_LIST, json.dumps({"skills": [entry()], "nextCursor": "next &/?"})),
            result(next_uri, json.dumps({"skills": [entry(second_uri)]})),
            result(second_uri, TEXT),
        ]
    )
    monkeypatch.setattr(session, "list_resources", resources)
    monkeypatch.setattr(session, "read_resource", read)
    catalog = await SkillCatalog.discover(session)
    assert list(catalog.skills) == [URI, second_uri]
    assert read.await_count == 2
    assert json.loads(await catalog.read(second_uri))["text"] == TEXT
    assert read.await_args_list[1].args == (next_uri,)


async def test_no_resources_capability_does_not_send_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    resources = AsyncMock()
    monkeypatch.setattr(session, "list_resources", resources)
    assert not (await SkillCatalog.discover(session)).skills
    resources.assert_not_awaited()


async def test_repeated_cursor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    session.server_capabilities = {"resources": {}}
    monkeypatch.setattr(
        session, "list_resources", AsyncMock(return_value=[ResourceDef(uri=SKILLS_LIST, name="skills/list")])
    )

    async def read(uri: str) -> ReadResourceResult:
        return result(uri, json.dumps({"skills": [], "nextCursor": "same"}))

    monkeypatch.setattr(session, "read_resource", read)
    with pytest.raises(ValueError, match="cursor"):
        await SkillCatalog.discover(session)


@pytest.mark.parametrize("change", ["digest", "size", "uri", "binary"])
async def test_changed_or_unexpected_content_is_rejected(change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    catalog = SkillCatalog(session, {})
    catalog.add([entry()])
    response = result(URI, TEXT)
    if change == "digest":
        response = result(URI, TEXT.replace("Keep", "Drop"))
    elif change == "size":
        response = result(URI, TEXT + "extra")
    elif change == "uri":
        response = result("skill://other/SKILL.md", TEXT)
    else:
        response = ReadResourceResult.model_validate({"contents": [{"uri": URI, "blob": "eA=="}]})
    monkeypatch.setattr(session, "read_resource", AsyncMock(return_value=response))
    with pytest.raises(ValueError):
        await catalog.read(URI)


async def test_unknown_files_and_skills_never_reach_server(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    catalog = SkillCatalog(session, {})
    catalog.add([entry()])
    read = AsyncMock()
    monkeypatch.setattr(session, "read_resource", read)
    with pytest.raises(ValueError, match="Unknown skill"):
        await catalog.read("skill://other/SKILL.md")
    for path in ("../other/SKILL.md", "/etc/passwd", "references/missing.md"):
        with pytest.raises(ValueError, match="not in the skill manifest"):
            await catalog.read(URI, path)
    read.assert_not_awaited()


@pytest.mark.parametrize("change", ["missing", "duplicate", "outside", "dynamic", "negative", "huge"])
async def test_invalid_manifests_are_rejected(change: str) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    catalog = SkillCatalog(session, {})
    item = entry()
    if change == "missing":
        item["resources"] = []
    elif change == "duplicate":
        item["resources"] *= 2
    elif change == "outside":
        item["resources"].append(entry("skill://outside/SKILL.md")["resources"][0])
    elif change == "dynamic":
        item["resources"] = "dynamic"
    elif change == "negative":
        item["resources"][0]["size"] = -1
    else:
        item["resources"][0]["size"] = 16 * 1024 * 1024 + 1
    with pytest.raises(ValueError):
        catalog.add([item])


@pytest.mark.parametrize("original", ["skill://plan/SKILL.md", URI, "custom://plan/SKILL.md"])
async def test_static_catalog_preserves_file_uri(monkeypatch: pytest.MonkeyPatch, original: str) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    session.server_capabilities = {"resources": {}}
    monkeypatch.setattr(
        session, "list_resources", AsyncMock(return_value=[ResourceDef(uri=SKILLS_CATALOG, name="skills")])
    )
    read = AsyncMock(side_effect=[result(SKILLS_CATALOG, json.dumps([entry(original)])), result(original, TEXT)])
    monkeypatch.setattr(session, "read_resource", read)
    catalog = await SkillCatalog.discover(session)
    assert json.loads(await catalog.read(original))["resource"] == original
    read.assert_awaited_with(original)
