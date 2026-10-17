"""Tests for MCPRegistry."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp_tiny_mcp import CallToolResult, TextContent
from aiohttp_tiny_mcp.protocol.models import ToolDef
from pydantic import BaseModel

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.registry import MCPRegistry


async def test_inspection_masks_connection_values_and_tracks_discovery_failures() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(
        name="docs",
        url="https://user:password@example.com:8443/private-token?key=query-token",
        headers={"Authorization": "Bearer header-token"},
    )
    session = _mock_session()
    session.process = None
    session.protocol_version = "2025-06-18"
    session.server_capabilities["resources"] = {}
    session.list_resources = AsyncMock(side_effect=ValueError("secret discovery payload"))
    with patch("axio_tools_mcp.registry.MCPSession", side_effect=session_factory(session)):
        with patch(
            "axio_tools_mcp.registry.SkillCatalog.discover", new=AsyncMock(side_effect=ValueError("skill secret"))
        ):
            await registry.add_server(config)
    info = registry.inspect_servers()[0]
    assert info.state == "connected"
    assert dict(info.details)["Endpoint origin"] == "https://example.com:8443"
    assert dict(info.details)["Negotiated protocol"] == "2025-06-18"
    assert info.errors == ("Resource discovery failed: ValueError", "Skill discovery failed: ValueError")
    text = str(info.details) + str(info.errors)
    for secret in ("password", "private-token", "query-token", "header-token", "secret discovery", "skill secret"):
        assert secret not in text
    registry.inspect_servers()
    session.list_resources.assert_awaited_once()
    session.is_connected = False
    offline = registry.inspect_servers()[0]
    assert offline.state == "error" and not offline.tools
    await registry.remove_server("docs")
    assert registry.inspect_servers() == []


async def test_inspection_does_not_expose_connection_exception_values() -> None:
    registry = MCPRegistry()
    session = _mock_session()
    session.connect.side_effect = RuntimeError("connection failed at https://host/token-secret")
    with patch("axio_tools_mcp.registry.MCPSession", side_effect=session_factory(session)):
        await registry.add_server(MCPServerConfig(name="broken", command="command-secret", args=["argument-secret"]))
    info = registry.inspect_servers()[0]
    assert info.state == "error" and not info.tools
    assert info.errors == ("Connection failed: RuntimeError",)
    text = str(info.details) + str(info.errors)
    assert all(secret not in text for secret in ("token-secret", "command-secret", "argument-secret"))


class Nothing(BaseModel):
    pass


def _mock_session(tools: list[ToolDef] | None = None) -> MagicMock:
    if tools is None:
        tools = [ToolDef(name="test", description="Test tool", input_schema={"type": "object", "properties": {}})]
    session = MagicMock()
    session.server_capabilities = {"tools": {}}
    session.connect = AsyncMock()
    session.list_tools = AsyncMock(return_value=tools)
    session.call_tool = AsyncMock(
        return_value=CallToolResult(content=[TextContent(text="ok")], is_error=False),
    )
    session.close = AsyncMock()
    session.is_connected = True
    return session


def session_factory(session: MagicMock) -> Callable[[MCPServerConfig], MagicMock]:
    """Hand the registry one mock session, carrying the config it connected with."""

    def make(config: MCPServerConfig) -> MagicMock:
        session.config = config
        return session

    return make


async def test_add_server() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="myserver", command="python")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        tools = await registry.add_server(config)

    assert len(tools) == 1
    assert tools[0].name == "myserver__test"
    assert "myserver" in registry.server_names
    assert registry.server_status("myserver") == "connected"
    assert registry.server_tool_count("myserver") == 1


async def test_remove_server() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="myserver", command="python")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        await registry.add_server(config)
        await registry.remove_server("myserver")

    assert "myserver" not in registry.server_names
    assert registry.all_tools == []


async def test_update_server() -> None:
    registry = MCPRegistry()
    config1 = MCPServerConfig(name="myserver", command="python")
    config2 = MCPServerConfig(name="myserver", command="node")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        await registry.add_server(config1)

        new_tools = [
            ToolDef(name="new_tool", description="New", input_schema={"type": "object", "properties": {}}),
        ]
        mock_cls.side_effect = session_factory(_mock_session(new_tools))
        tools = await registry.update_server("myserver", config2)

    assert len(tools) == 1
    assert tools[0].name == "myserver__new_tool"


async def test_all_tools_aggregation() -> None:
    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        tools_a = [ToolDef(name="a", description="A", input_schema={"type": "object", "properties": {}})]
        mock_cls.side_effect = session_factory(_mock_session(tools_a))
        await registry.add_server(MCPServerConfig(name="server_a", command="a"))

        tools_b = [ToolDef(name="b", description="B", input_schema={"type": "object", "properties": {}})]
        mock_cls.side_effect = session_factory(_mock_session(tools_b))
        await registry.add_server(MCPServerConfig(name="server_b", command="b"))

    all_tools = registry.all_tools
    assert len(all_tools) == 2
    names = {t.name for t in all_tools}
    assert "server_a__a" in names
    assert "server_b__b" in names


async def test_server_status_error() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="bad", command="fail")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        session = _mock_session()
        session.connect = AsyncMock(side_effect=ConnectionError("refused"))
        mock_cls.side_effect = session_factory(session)
        await registry.add_server(config)

    assert registry.server_status("bad") == "error"
    assert registry.server_tool_count("bad") == 0


async def test_duplicate_server_raises() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="myserver", command="python")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        await registry.add_server(config)

    import pytest

    with pytest.raises(ValueError, match="already exists"):
        await registry.add_server(config)


async def test_close_all() -> None:
    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        session = _mock_session()
        mock_cls.side_effect = session_factory(session)
        await registry.add_server(MCPServerConfig(name="s1", command="a"))
        await registry.close()

    session.close.assert_awaited()


async def test_init_from_config_db() -> None:
    """Registry loads saved configs from config DB on init."""
    mock_db = MagicMock()
    mock_db.get_prefix = AsyncMock(
        return_value={
            "mcp.myserver.command": "python",
            "mcp.myserver.args": '["-m", "server"]',
        }
    )
    mock_db.delete_prefix = AsyncMock()
    mock_db.set = AsyncMock()

    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        await registry.init(mock_db)

    assert "myserver" in registry.server_names
    mock_db.get_prefix.assert_awaited_once_with("mcp.")


async def test_dual_config_init() -> None:
    """Registry loads servers from both global and project config DBs."""
    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(
        return_value={
            "mcp.global_server.command": "python",
            "mcp.global_server.args": '["-m", "global_mod"]',
        }
    )
    global_db.delete_prefix = AsyncMock()
    global_db.set = AsyncMock()

    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(
        return_value={
            "mcp.project_server.command": "node",
        }
    )
    project_db.delete_prefix = AsyncMock()
    project_db.set = AsyncMock()

    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        await registry.init(config=project_db, global_config=global_db)

    assert "global_server" in registry.server_names
    assert "project_server" in registry.server_names

    # Verify scope tracking
    assert registry.get_server_scope("global_server") is global_db
    assert registry.get_server_scope("project_server") is project_db


async def test_dual_config_persist_to_correct_scope() -> None:
    """Adding a server with explicit scope persists to the correct DB."""
    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(return_value={})
    global_db.delete_prefix = AsyncMock()
    global_db.set = AsyncMock()

    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(return_value={})
    project_db.delete_prefix = AsyncMock()
    project_db.set = AsyncMock()

    registry = MCPRegistry()
    await registry.init(config=project_db, global_config=global_db)

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        await registry.add_server(MCPServerConfig(name="proj_srv", command="python"), scope=project_db)

    # Should persist to project DB, not global
    proj_set_calls = {call.args[0] for call in project_db.set.call_args_list}
    assert "mcp.proj_srv.command" in proj_set_calls
    assert not any("proj_srv" in str(call) for call in global_db.set.call_args_list)
    assert registry.get_server_scope("proj_srv") is project_db


async def test_scope_only_change_skips_reconnect() -> None:
    """Changing scope without modifying config should not reconnect."""
    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(
        return_value={
            "mcp.srv.command": "python",
        }
    )
    global_db.delete_prefix = AsyncMock()
    global_db.set = AsyncMock()

    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(return_value={})
    project_db.delete_prefix = AsyncMock()
    project_db.set = AsyncMock()

    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_session = _mock_session()
        mock_cls.side_effect = session_factory(mock_session)
        await registry.init(config=project_db, global_config=global_db)

        # Reset to track calls during update
        mock_cls.reset_mock()
        mock_session.close.reset_mock()

        # Change scope only (same config)
        same_config = MCPServerConfig(name="srv", command="python")
        await registry.update_server("srv", same_config, scope=project_db)

    # Should NOT have created a new session or closed the old one
    mock_cls.assert_not_called()
    mock_session.close.assert_not_awaited()

    # Should be persisted to project DB now
    assert registry.get_server_scope("srv") is project_db
    proj_set_calls = {call.args[0] for call in project_db.set.call_args_list}
    assert "mcp.srv.command" in proj_set_calls

    # Old scope (global) should have been cleaned up
    global_db.delete_prefix.assert_awaited_with("mcp.srv.")


async def test_persistence_roundtrip() -> None:
    """Add server persists config, remove deletes it."""
    mock_db = MagicMock()
    mock_db.get_prefix = AsyncMock(return_value={})
    mock_db.delete_prefix = AsyncMock()
    mock_db.set = AsyncMock()

    registry = MCPRegistry()
    await registry.init(mock_db)

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.side_effect = session_factory(_mock_session())
        await registry.add_server(MCPServerConfig(name="persisted", command="python"))

    # Check that set was called with mcp.persisted.command
    set_calls = {call.args[0]: call.args[1] for call in mock_db.set.call_args_list}
    assert "mcp.persisted.command" in set_calls
    assert set_calls["mcp.persisted.command"] == "python"

    # Remove should delete prefix
    await registry.remove_server("persisted")
    mock_db.delete_prefix.assert_awaited_with("mcp.persisted.")


async def test_resources_without_tools_are_exposed_and_discarded_on_remove() -> None:
    from aiohttp_tiny_mcp import Registry
    from aiohttp_tiny_mcp.testing import serving

    registry = Registry("resources", "1")

    async def guide(args: Nothing) -> str:
        return "Resource body"

    registry.resource("notes://guide")(guide)
    async with serving(registry) as url:
        client = MCPRegistry()
        await client.add_server(MCPServerConfig(name="notes", url=url))
        try:
            tools = {t.name: t for t in client.all_tools}
            assert "Resource body" in await tools["notes__read_resource"](uri="notes://guide")
            assert "notes://guide" in client.instructions()
            await client.remove_server("notes")
            assert not client.all_tools
            assert not client.instructions()
        finally:
            await client.close()


async def test_cancelled_connection_closes_session() -> None:
    import asyncio

    session = _mock_session()
    session.connect.side_effect = asyncio.CancelledError()
    registry = MCPRegistry()
    with patch("axio_tools_mcp.registry.MCPSession", return_value=session):
        try:
            await registry.add_server(MCPServerConfig(name="cancel", command="python"))
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation must propagate")
    session.close.assert_awaited_once()


async def test_initialization_deadline_discards_partial_catalog_and_reconnects() -> None:
    import asyncio

    config = MCPServerConfig(name="slow", command="python", timeout=0.05)
    session = _mock_session()
    session.server_capabilities["resources"] = {}
    session.list_resources = AsyncMock(side_effect=lambda: None)

    async def hang() -> None:
        await asyncio.Event().wait()

    session.list_resources.side_effect = hang
    updates: list[str] = []
    registry = MCPRegistry(on_status=updates.append)
    db = MagicMock()
    db.delete_prefix = AsyncMock()
    db.set = AsyncMock()
    with patch("axio_tools_mcp.registry.MCPSession", side_effect=session_factory(session)):
        async with asyncio.timeout(2):
            tools = await registry.add_server(config, scope=db)
    assert not tools and not registry.all_tools and not registry.instructions()
    assert registry.server_status("slow") == "error"
    assert "timed out after 0.05s" in registry.inspect_servers()[0].errors[0]
    assert "timeout 0.05s" in updates[0]
    session.close.assert_awaited_once()
    db.reset_mock()
    healthy = _mock_session()
    healthy.process = None
    with patch("axio_tools_mcp.registry.MCPSession", side_effect=session_factory(healthy)):
        await registry.reconnect()
    assert registry.server_status("slow") == "connected"
    assert len(registry.all_tools) == 1
    assert registry.get_server_scope("slow") is db
    assert registry.server_config("slow") == config
    db.set.assert_not_called()
    assert not registry.inspect_servers()[0].errors
    await registry.close()


async def test_tools_catalog_has_initialization_deadline() -> None:
    import asyncio

    async def hang() -> None:
        await asyncio.Event().wait()

    session = _mock_session()
    session.list_tools.side_effect = hang
    registry = MCPRegistry()
    with patch("axio_tools_mcp.registry.MCPSession", side_effect=session_factory(session)):
        async with asyncio.timeout(2):
            await registry.add_server(MCPServerConfig(name="slow", command="python", timeout=0.05))
    assert registry.server_status("slow") == "error"
    assert "timed out after 0.05s" in registry.inspect_servers()[0].errors[0]
    session.close.assert_awaited_once()
