from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from axio.exceptions import GuardError
from axio.permission import PermissionGuard
from axio.permission_policy import Approval, PermissionRule, PolicyGuard
from axio.tool import Tool
from axio.tool_access import file_access
from axio_tools_local.paths import LocalPaths as Workspace
from textual.app import App
from textual.widgets import OptionList, Select, Static

from axio_tui.app import AgentApp
from axio_tui.permissions import ConfigRuleStore, PermissionDialog, PermissionsScreen, prompt_permission
from axio_tui.sqlite_config import ProjectConfig


async def write(path: str, content: str = "") -> str:
    return content


def file_tool() -> Tool[Any]:
    return Tool("write", write, access=file_access("write", "path"))


async def test_saved_rules_survive_restart_and_stay_in_project(tmp_path: Path) -> None:
    config = ProjectConfig(tmp_path / "rules.db", project=str(tmp_path))
    store = ConfigRuleStore(config)
    prompt = AsyncMock(return_value=Approval("allow", "project"))
    policy = PolicyGuard(tmp_path, prompt, store=store)
    tool = replace(file_tool(), guards=(policy,))
    try:
        await tool(path=str(tmp_path / "out"))
        rule = (await store.load())[0]
        await config.close()
        restarted = PolicyGuard(tmp_path, AsyncMock(return_value=Approval()), store=store)
        await replace(tool, guards=(restarted,))(path=str(tmp_path / "out"))
        restarted.prompt.assert_not_awaited()  # type: ignore[attr-defined]
        other = ProjectConfig(tmp_path / "rules.db", project=str(tmp_path / "other"))
        try:
            assert await ConfigRuleStore(other).load() == []
        finally:
            await other.close()
        await restarted.revoke(rule.id)
        with pytest.raises(GuardError):
            await replace(tool, guards=(restarted,))(path=str(tmp_path / "out"))
    finally:
        await config.close()


async def test_invalid_rule_blocks_calls_and_can_be_removed(tmp_path: Path) -> None:
    config = ProjectConfig(tmp_path / "rules.db")
    store = ConfigRuleStore(config)
    policy = PolicyGuard(tmp_path, AsyncMock(), store=store)
    try:
        await config.set("permissions.v1.broken", "invalid json")
        with pytest.raises(GuardError, match="Cannot load"):
            await policy.check(file_tool(), path=str(tmp_path / "out"))
        app: App[None] = App()
        async with app.run_test() as pilot:
            screen = PermissionsScreen(policy, store)
            await app.push_screen(screen)
            await pilot.pause()
            assert screen.entries[0][0] == "broken"
            screen.query_one(OptionList).highlighted = 0
            await pilot.click("#rule-revoke")
            assert await store.load() == []
    finally:
        await config.close()


async def test_dialog_shows_scope_and_returns_explicit_project_approval(tmp_path: Path) -> None:
    policy = PolicyGuard(tmp_path, AsyncMock())
    request = policy.describe(file_tool(), {"path": str(tmp_path / "src/a.py"), "content": "[red]literal[/]"})
    app: App[None] = App()
    answers: list[Approval | None] = []
    async with app.run_test(size=(110, 45)) as pilot:
        await app.push_screen(PermissionDialog(request), answers.append)
        await pilot.pause()
        assert app.focused is app.screen.query_one("#permission-deny")
        app.screen.query_one("#permission-lifetime", Select).value = "project"
        app.screen.query_one("#permission-target", Select).value = "directories"
        await pilot.pause()
        assert "including descendants" in str(app.screen.query_one("#permission-scope", Static).render())
        await pilot.click("#permission-allow")
        assert answers == [Approval("allow", "project", directories=True)]


async def test_escape_always_denies_once(tmp_path: Path) -> None:
    policy = PolicyGuard(tmp_path, AsyncMock())
    request = policy.describe(file_tool(), {"path": str(tmp_path / "file")})
    app: App[None] = App()
    answers: list[Approval | None] = []
    async with app.run_test(size=(110, 45)) as pilot:
        await app.push_screen(PermissionDialog(request), answers.append)
        app.screen.query_one("#permission-lifetime", Select).value = "project"
        await pilot.press("escape")
        assert answers == [Approval()]


async def test_app_preserves_existing_guards_and_protects_new_tools(tmp_path: Path) -> None:
    class Redirect(PermissionGuard):
        async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
            return {**kwargs, "path": str(tmp_path.parent / "outside")}

    original = Redirect()
    app = AgentApp(paths=Workspace.discover(tmp_path))
    app._all_tools = [replace(file_tool(), guards=(original,)), Tool("new_mcp_tool", write)]
    app._permissions.prompt = AsyncMock(return_value=Approval())
    app._rebuild_agent_tools()
    assert app._active_tools[0].guards == (original, app._tool_guard)
    for tool in app._active_tools:
        with pytest.raises(GuardError):
            await tool(path=str(tmp_path / "file"))
    assert len(app._permissions.prompt.await_args_list) == 2
    assert app._permissions.prompt.await_args_list[0].args[0].accesses[0].path == str(tmp_path.parent / "outside")


async def test_cancelled_worker_removes_dialog(tmp_path: Path) -> None:
    app: App[None] = App()
    policy = PolicyGuard(tmp_path, partial(prompt_permission, app))
    tool = replace(file_tool(), guards=(policy,))
    async with app.run_test(size=(110, 45)) as pilot:
        worker = app.run_worker(tool(path=str(tmp_path / "file")), exit_on_error=False)
        for _ in range(30):
            await pilot.pause()
            if isinstance(app.screen, PermissionDialog):
                break
        assert isinstance(app.screen, PermissionDialog)
        worker.cancel()
        await pilot.pause()
        assert not isinstance(app.screen, PermissionDialog)
        assert await policy.rules() == []
        worker = app.run_worker(tool(path=str(tmp_path / "file")), exit_on_error=False)
        for _ in range(30):
            await pilot.pause()
            if isinstance(app.screen, PermissionDialog):
                break
        await pilot.click("#permission-allow")
        await asyncio.wait_for(worker.wait(), 2)


async def test_rules_screen_revokes_chat_rule(tmp_path: Path) -> None:
    config = ProjectConfig(tmp_path / "rules.db")
    store = ConfigRuleStore(config)
    policy = PolicyGuard(tmp_path, AsyncMock(return_value=Approval("allow", "chat")), store=store)
    await policy.check(file_tool(), path=str(tmp_path / "out"))
    try:
        app: App[None] = App()
        async with app.run_test(size=(110, 45)) as pilot:
            screen = PermissionsScreen(policy, store)
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one(OptionList).highlighted = 0
            await pilot.click("#rule-revoke")
            assert await policy.rules() == []
    finally:
        await config.close()


async def test_separate_connections_do_not_overwrite_rules(tmp_path: Path) -> None:
    configs = [ProjectConfig(tmp_path / "rules.db") for _ in range(2)]
    stores = [ConfigRuleStore(c) for c in configs]
    policy = PolicyGuard(tmp_path, AsyncMock())
    rules = [
        PermissionRule.from_request(
            policy.describe(file_tool(), {"path": str(tmp_path / str(i))}), Approval("allow", "project")
        )
        for i in range(2)
    ]
    try:
        await stores[0].load()  # Create the database before concurrent connections.
        await asyncio.gather(*(s.add(r) for s, r in zip(stores, rules)))
        assert {r.id for r in await stores[0].load()} == {r.id for r in rules}
    finally:
        for config in configs:
            await config.close()
