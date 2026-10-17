from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp_tiny_mcp import Registry
from aiohttp_tiny_mcp.testing import serving
from axio.blocks import ToolUseBlock
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.permission_policy import Approval
from axio.transport import TransportCatalog, TransportEntry
from axio_tools_local.paths import LocalPaths as Workspace
from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.plugin import MCPPlugin
from axio_tools_mcp.provider import MCPToolProvider
from axio_tools_mcp.skills import SKILLS_CATALOG
from pydantic import BaseModel
from textual.widgets import Input, Markdown, OptionList, Static, Tree

from axio_tui.app import AgentApp, ModelRole
from axio_tui.args import Args
from axio_tui.mcp import MCPScreen
from axio_tui.options import StartupOptions
from axio_tui.skills import SkillsScreen


class Nothing(BaseModel):
    pass


class FakeTransport:
    name = "Fixture"
    api_key = ""
    model = ModelSpec(id="fixture", capabilities=frozenset({Capability.text, Capability.tool_use}))
    models = ModelRegistry([model])

    def __init__(self, **kwargs: object) -> None:
        self.session = kwargs.get("session")

    async def fetch_models(self) -> None:
        pass


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentApp:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr("axio_tui.app.DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr("axio_tui.app.discover_tools_plugins", lambda: {})
    monkeypatch.setattr("axio_tui.app.discover_guards", lambda: {})
    monkeypatch.setattr("axio_tui.transport_registry.discover_transport_settings", lambda: {})
    monkeypatch.setattr(
        "axio_tui.transport_registry.discover_transports",
        lambda: TransportCatalog({"fake": TransportEntry.from_target("fake", FakeTransport)}),
    )
    return AgentApp(
        paths=Workspace.discover(tmp_path),
        options=StartupOptions(skill_directories=(tmp_path / "skills",)),
    )


async def test_tui_loads_skills_scoped_instructions_and_reloads(isolated_app: AgentApp, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Root policy")
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\nSkill body")
    child = tmp_path / "child"
    child.mkdir()
    (child / "AGENTS.md").write_text("Scoped policy")
    (child / "file.py").touch()
    app = isolated_app
    async with app.run_test():
        await asyncio.wait_for(app.ready.wait(), 5)
        assert app._agent is not None
        await app._agent._prepare_instructions()
        assert "Root policy" in app._agent._effective_system()
        item = next(i for i in app._agent._catalog() if i.source == str(skill / "SKILL.md"))
        assert "Skill body" in (await app._agent.load_instruction(item.id)).text
        await app._agent._prepare_instructions()
        assert "read_local_skill" in {t.name for t in app._agent.get_tools()}
        binding = app._local_binding
        assert binding is not None
        await binding.before_tools([ToolUseBlock("r", "read_file", {"filename": "child/file.py"})], 0)
        await app._agent._prepare_instructions()
        assert "Scoped policy" in app._agent._effective_system()
        assert str(child / "AGENTS.md") in binding.instruction.snapshot.explain(binding.provider.paths, app.paths.cwd)
        sub, _ = await app._make_subagent()
        await sub._prepare_instructions()
        assert "Scoped policy" not in sub._effective_system()
        assert sub.before_tools_hooks != app._agent.before_tools_hooks
        (skill / "SKILL.md").unlink()
        await app._reload_workspace_work()
        await app._agent._prepare_instructions()
        assert "read_local_skill" not in {t.name for t in app._agent.get_tools()}
        assert "Root policy" in app._agent._effective_system()


async def test_tui_connects_mcp_resources_and_closes(isolated_app: AgentApp, tmp_path: Path) -> None:
    registry = Registry("fixture", "1")

    async def guide(args: Nothing) -> str:
        return "Resource contents"

    registry.resource("notes://guide")(guide)

    async with serving(registry) as url:
        (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"docs": {"url": url}}}))
        async with isolated_app.run_test():
            await asyncio.wait_for(isolated_app.ready.wait(), 5)
            agent = isolated_app._agent
            assert agent is not None
            tool = next(t for t in agent.get_tools() if t.name == "docs__read_resource")
            approval = AsyncMock(return_value=Approval("allow"))
            isolated_app._permissions.prompt = approval
            assert "Resource contents" in await tool(uri="notes://guide")
            approval.assert_awaited_once()
            loaded = await agent.load_instruction("mcp:docs:resources")
            assert "notes://guide" in loaded.text
        assert not isolated_app._providers["mcp"].all_tools


async def test_mcp_command_inspects_automatic_manual_disabled_and_failed_servers(
    isolated_app: AgentApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry("inspection", "1")
    reads: list[str] = []

    async def guide(args: Nothing) -> str:
        reads.append("guide")
        return "Guide contents"

    registry.resource("notes://guide", description="A reference guide")(guide)
    async with serving(registry) as url:
        (tmp_path / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "docs": {"url": url},
                        "disabled": {"enabled": False},
                        "broken": {"url": url, "headersFrom": {"Authorization": "AXIO_TEST_MISSING_SECRET"}},
                    }
                }
            )
        )
        monkeypatch.delenv("AXIO_TEST_MISSING_SECRET", raising=False)
        async with isolated_app.run_test(size=(120, 40)) as pilot:
            await asyncio.wait_for(isolated_app.ready.wait(), 5)
            plugin = MCPPlugin()
            await plugin._registry.add_server(MCPServerConfig(name="saved", url=url))
            isolated_app._tools_plugins["mcp"] = plugin
            meta = AsyncMock()
            monkeypatch.setattr(isolated_app, "_write_meta", meta)
            prompt = isolated_app.query_one("#input", Input)
            prompt.value = "/mcp"
            prompt.focus()
            await pilot.press("enter")
            assert isinstance(isolated_app.screen, MCPScreen)
            tree = isolated_app.screen.query_one("#mcp-tree", Tree)
            assert len(tree.root.children) == 4
            snapshots = {item.name: item for item in isolated_app._inspect_mcp()}
            assert snapshots["docs"].resources[0].uri == "notes://guide"
            assert snapshots["saved"].resources[0].description == "A reference guide"
            assert snapshots["disabled"].state == "disabled"
            assert snapshots["broken"].state == "error" and snapshots["broken"].errors
            assert dict(snapshots["docs"].details)["Negotiated protocol"] != "Not connected"
            await pilot.press("g", "u", "i", "d", "e", "enter")
            assert len(tree.root.children) == 2
            await pilot.click("#mcp-refresh")
            assert len(tree.root.children) == 2
            assert reads == []
            meta.assert_not_called()
            assert not isolated_app._is_running
            await pilot.press("escape")
            assert not isinstance(isolated_app.screen, MCPScreen)


async def test_skills_command_browses_local_and_both_mcp_sources(
    isolated_app: AgentApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "skills" / "demo" / "SKILL.md"
    local.parent.mkdir(parents=True)
    local.write_text("---\nname: demo\ndescription: Local demo\n---\nLocal instructions.")
    uri = "skill://demo/SKILL.md"
    body = "# Remote instructions\n\nKeep this source separate."
    reads: list[str] = []
    registry = Registry("skills", "1")

    async def catalog(args: Nothing) -> list[dict[str, Any]]:
        return [
            {
                "uri": uri,
                "frontmatter": {"name": "demo", "description": "Remote demo"},
                "resources": [
                    {
                        "uri": uri,
                        "size": len(body.encode()),
                        "digest": "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
                    }
                ],
            }
        ]

    async def instructions(args: Nothing) -> str:
        reads.append(uri)
        return body

    registry.resource(SKILLS_CATALOG)(catalog)
    registry.resource(uri)(instructions)
    async with serving(registry) as url:
        (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"docs": {"url": url}}}))
        async with isolated_app.run_test(size=(120, 40)) as pilot:
            await asyncio.wait_for(isolated_app.ready.wait(), 5)
            plugin = MCPPlugin()
            await plugin._registry.add_server(MCPServerConfig(name="saved", url=url))
            isolated_app._tools_plugins["mcp"] = plugin
            meta = AsyncMock()
            monkeypatch.setattr(isolated_app, "_write_meta", meta)
            prompt = isolated_app.query_one("#input", Input)
            prompt.value = "/skills"
            prompt.focus()
            await pilot.press("enter")
            screen = isolated_app.screen
            assert isinstance(screen, SkillsScreen)
            assert screen.query_one("#skills-list", OptionList).option_count == 2
            meta.assert_not_called()
            assert reads == []
            for query, expected in (("Local", "Local instructions."), ("manual", body), ("MCP: docs", body)):
                await pilot.press("ctrl+f")
                screen.query_one("#skills-filter", Input).value = query
                await pilot.pause()
                assert screen.query_one("#skills-list", OptionList).option_count == 1
                await pilot.press("enter")
                await screen.workers.wait_for_complete()
                assert expected in screen.query_one("#skill-content", Markdown).source
                assert "demo" in str(screen.query_one("#skill-name", Static).content)
            assert reads == [uri]
            meta.assert_not_called()
            assert not isolated_app._is_running
            await pilot.press("escape")
            local.unlink()
            await isolated_app._reload_workspace_work()
            await isolated_app._show_skills()
            await pilot.pause()
            assert isolated_app.screen.query_one("#skills-list", OptionList).option_count == 1


async def test_signed_out_start_can_activate_without_restart(
    isolated_app: AgentApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("axio_tui.transport_registry.discover_transports", lambda: TransportCatalog())
    app = isolated_app
    async with app.run_test():
        await asyncio.wait_for(app.ready.wait(), 5)
        assert app._agent is None
        source = FakeTransport(session=app._session)
        app._transports.register_dynamic("fake", source)
        await app._activate_transport("fake")
        assert app._agent is not None
        assert app._agent.before_tools_hooks
        assert app._role_bindings[ModelRole.CHAT].model == source.model


async def test_reload_during_run_does_not_replace_connections(
    isolated_app: AgentApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = isolated_app
    app.ready.set()
    app._is_running = True
    reload = AsyncMock()
    monkeypatch.setattr(app, "_load_workspace", reload)
    monkeypatch.setattr(app, "_write_meta", AsyncMock())
    await app._reload_workspace()
    reload.assert_not_called()


def test_workspace_arguments() -> None:
    args = Args().parse_args(["--isolated", "--skills-dir", "one", "--skills-dir", "two", "--mcp-config", "mcp.json"])
    assert args.isolated
    assert args.skills_dir == [Path("one"), Path("two")]
    assert args.mcp_config == [Path("mcp.json")]


@pytest.mark.parametrize("available", [False, True])
async def test_saved_provider_never_falls_back_to_another_provider(
    isolated_app: AgentApp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
) -> None:
    from axio_tui.sqlite_config import ProjectConfig, connect_config

    connection = await connect_config(tmp_path / "app.db")
    try:
        await ProjectConfig(connection).set("model.chat", "chatgpt:removed-model")
    finally:
        await connection.close()
    transports = {"fake": FakeTransport}
    if available:
        transports["chatgpt"] = FakeTransport
    monkeypatch.setattr(
        "axio_tui.transport_registry.discover_transports",
        lambda: TransportCatalog({name: TransportEntry.from_target(name, cls) for name, cls in transports.items()}),
    )
    async with isolated_app.run_test():
        await asyncio.wait_for(isolated_app.ready.wait(), 5)
        assert isolated_app._agent is None
        assert ModelRole.CHAT not in isolated_app._role_bindings
        assert await isolated_app._config.get("model.chat") == "chatgpt:removed-model"


async def test_mcp_reconnect_button_replaces_automatic_and_saved_sessions(
    isolated_app: AgentApp, tmp_path: Path
) -> None:
    registry = Registry("reconnect", "1")

    async def guide(args: Nothing) -> str:
        return "Guide contents"

    registry.resource("notes://guide")(guide)
    app = isolated_app
    async with serving(registry) as url:
        (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"automatic": {"url": url}}}))
        async with app.run_test(size=(120, 40)) as pilot:
            await asyncio.wait_for(app.ready.wait(), 5)
            plugin = MCPPlugin()
            await plugin._registry.add_server(MCPServerConfig(name="saved", url=url))
            app._tools_plugins["mcp"] = plugin
            app._on_tools_plugin_dismissed()
            mcp = app._providers["mcp"]
            assert isinstance(mcp, MCPToolProvider)
            automatic = mcp.runtime.servers[0].session
            saved = plugin._registry._sessions["saved"]
            assert automatic is not None
            old_automatic_http, old_saved_http = automatic.http, saved.http
            assert old_automatic_http is not None and old_saved_http is not None
            await app._show_mcp()
            await pilot.pause()
            await pilot.click("#mcp-reconnect")
            assert not isinstance(app.screen, MCPScreen)
            assert app._workspace_task is not None
            await app._workspace_task.wait()
            assert not app._workspace_loading
            assert old_automatic_http.closed and old_saved_http.closed
            assert mcp.runtime.servers[0].session is not automatic
            assert plugin._registry._sessions["saved"] is not saved
            assert app._agent is not None
            names = {tool.name for tool in app._agent.get_tools()}
            assert {"automatic__read_resource", "saved__read_resource"} <= names
            assert all(server.state == "connected" for server in app._inspect_mcp())


async def test_mcp_reconnect_refuses_active_run(isolated_app: AgentApp) -> None:
    app = isolated_app
    async with app.run_test():
        await asyncio.wait_for(app.ready.wait(), 5)
        app._is_running = True
        try:
            await app._reconnect_mcp()
            assert app._workspace_task is None
            assert not app._workspace_loading
        finally:
            app._is_running = False


async def test_unavailable_saved_model_is_preserved_without_default_substitution(isolated_app: AgentApp) -> None:
    import axio_tui.app as ui
    from axio_tui.sqlite_config import ProjectConfig

    settings = ProjectConfig(ui.DB_PATH)
    await settings.set("model.chat", "fake:anthropic/claude-sonnet-4-6")
    await settings.close()
    async with isolated_app.run_test() as pilot:
        await asyncio.wait_for(isolated_app.ready.wait(), 5)
        assert isolated_app._agent is None
        assert ModelRole.CHAT not in isolated_app._role_bindings
        assert await isolated_app._config.get("model.chat") == "fake:anthropic/claude-sonnet-4-6"
        await isolated_app._on_model_selected(ModelRole.CHAT, ("fake", FakeTransport.model))
        await pilot.pause()
        assert isolated_app._agent is not None
        assert isolated_app._chat_model == FakeTransport.model
        assert await isolated_app._config.get("model.chat") == "fake:fixture"


async def test_mcp_manager_reloads_saved_file_into_agent(isolated_app: AgentApp) -> None:
    from textual.widgets import Button

    from axio_tui.mcp_manage import MCPFileEditScreen, MCPManagementScreen

    registry = Registry("managed", "1")

    @registry.tool
    async def echo(args: Nothing) -> str:
        return "managed"

    async with serving(registry) as url:
        async with isolated_app.run_test(size=(120, 40)) as pilot:
            await asyncio.wait_for(isolated_app.ready.wait(), 5)
            await isolated_app._show_mcp()
            await pilot.pause()
            await pilot.click("#mcp-manage")
            assert isinstance(isolated_app.screen, MCPManagementScreen)
            await pilot.click("#mcp-add")
            editor = isolated_app.screen_stack[-1]
            assert isinstance(editor, MCPFileEditScreen)
            editor.query_one("#mcp-edit-name", Input).value = "managed"
            editor.query_one("#mcp-edit-url", Input).value = url
            editor.query_one("#mcp-file-save", Button).press()
            await pilot.pause()
            isolated_app.screen.query_one("#mcp-manager-close", Button).press()
            await pilot.pause()
            assert isolated_app._workspace_task is not None
            await isolated_app._workspace_task.wait()
            await pilot.pause()
            assert isinstance(isolated_app.screen_stack[-1], MCPScreen)
            assert isolated_app._agent is not None
            assert "managed__echo" in {tool.name for tool in isolated_app._agent.get_tools()}
            assert any(
                server.name == "managed" and server.state == "connected" for server in isolated_app._inspect_mcp()
            )
