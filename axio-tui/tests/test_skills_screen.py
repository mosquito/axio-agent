from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from axio_tools_local.skills import LocalSkills
from axio_tools_mcp.runtime import MCPRuntime
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Markdown, OptionList, Static

from axio_tui.skills import SkillEntry, SkillsScreen, group_entries, provider_entries


@pytest.mark.parametrize("size", [(120, 40), (60, 24)])
async def test_filter_browse_duplicate_names_and_close(size: tuple[int, int]) -> None:
    read = AsyncMock(return_value=json.dumps({"text": "# Instructions\n\nUse this skill."}))
    description = "A full description with [literal] brackets. " * 12
    entries = [
        SkillEntry("demo", description, "Local", "/first/demo/SKILL.md", read),
        SkillEntry("demo", "Second source", "Local", "/second/demo/SKILL.md", read),
    ]
    app: App[None] = App()
    screen = SkillsScreen(entries)
    async with app.run_test(size=size) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.has_class("narrow") == (size[0] < 90)
        assert screen.query_one("#skills-list", OptionList).option_count == 2
        assert str(screen.query_one("#skill-description", Static).content) == description
        read.assert_not_awaited()
        await pilot.press("s", "e", "c", "o", "n", "d")
        assert screen.query_one("#skills-list", OptionList).option_count == 1
        assert "/second/" in str(screen.query_one("#skill-source", Static).content)
        await pilot.press("enter")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        read.assert_awaited_once()
        assert "Use this skill." in screen.query_one("#skill-content", Markdown).source
        await pilot.press("tab", "pagedown")
        assert screen.query_one("#skills-details", VerticalScroll).has_focus
        if size[0] < 90:
            assert screen.query_one("#skills-details", VerticalScroll).scroll_y > 0
        for selector in ("#skill-open", "#skills-close", "#skills-list", "#skills-details"):
            region = screen.query_one(selector).region
            assert region.width > 0 and region.height > 0
            assert 0 <= region.x and region.right <= size[0]
            assert 0 <= region.y and region.bottom <= size[1]
        await pilot.press("ctrl+f")
        assert screen.query_one("#skills-filter", Input).has_focus
        await pilot.press("escape")
        assert app.screen is not screen


async def test_empty_filter_and_discovery_notes() -> None:
    app: App[None] = App()
    screen = SkillsScreen([], ["Discovery failed [literal]."])
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert "No skills available" in str(screen.query_one("#skill-name", Static).content)
        assert screen.query_one("#skill-open", Button).disabled
        await pilot.press("x", "enter")
        assert screen.query_one("#skills-list", OptionList).option_count == 0
        assert "Discovery failed [literal]." == str(screen.query_one("#skills-notes Static", Static).content)
        await pilot.click("#skills-close")
        assert app.screen is not screen


async def test_read_failure_can_retry_and_filter_has_no_matches() -> None:
    read = AsyncMock(side_effect=[ValueError("Catalog changed [reload]"), json.dumps({"text": "Recovered"})])
    app: App[None] = App()
    screen = SkillsScreen([SkillEntry("demo", "Demo", "MCP: docs", "skill://demo/SKILL.md", read)])
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.press("enter")
        await screen.workers.wait_for_complete()
        assert "Catalog changed [reload]" in str(screen.query_one("#skill-status", Static).content)
        assert not screen.query_one("#skill-open", Button).disabled
        await pilot.click("#skill-open")
        await screen.workers.wait_for_complete()
        assert screen.query_one("#skill-content", Markdown).source == "Recovered"
        await pilot.press("ctrl+f", "z")
        assert "No skills match" in str(screen.query_one("#skill-name", Static).content)
        assert screen.query_one("#skill-content", Markdown).source == ""


async def test_slow_read_does_not_replace_another_selection() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def slow() -> str:
        started.set()
        await release.wait()
        return json.dumps({"text": "Slow content"})

    fast = AsyncMock(return_value=json.dumps({"text": "Fast content"}))
    app: App[None] = App()
    screen = SkillsScreen(
        [
            SkillEntry("a", "Slow", "Local", "/a", slow),
            SkillEntry("b", "Fast", "Local", "/b", fast),
        ]
    )
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 2)
        await pilot.press("down", "enter")
        await pilot.pause()
        assert screen.query_one("#skill-content", Markdown).source == "Fast content"
        release.set()
        await screen.workers.wait_for_complete()
        assert screen.query_one("#skill-content", Markdown).source == "Fast content"
        await pilot.press("up")
        assert screen.query_one("#skill-content", Markdown).source == "Slow content"


async def test_close_cancels_pending_read() -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def read() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ""

    app: App[None] = App()
    screen = SkillsScreen([SkillEntry("demo", "Demo", "Local", "/demo", read)])
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 2)
        await pilot.press("escape")
        await asyncio.wait_for(cancelled.wait(), 2)
        assert app.screen is not screen


async def test_identical_local_contents_share_paths_and_filter(tmp_path: Path) -> None:
    text = "---\nname: demo\ndescription: Demo\n---\nSame instructions."
    paths = [tmp_path / folder / "demo" / "SKILL.md" for folder in ("first", "second", "different")]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text(text if path != paths[2] else text + "\nDifferent instructions.")
    skills = LocalSkills(tmp_path, isolated=True, directories=[path.parent for path in paths])
    await skills.reload()
    entries, notes = provider_entries(skills, MCPRuntime())
    app: App[None] = App()
    screen = SkillsScreen(entries, notes)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#skills-list", OptionList).option_count == 2
        screen.query_one("#skills-filter", Input).value = "second"
        await pilot.pause()
        assert screen.query_one("#skills-list", OptionList).option_count == 1
        assert f"{paths[0]}, {paths[1]}" in str(screen.query_one("#skill-source", Static).content)
        await pilot.press("enter")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#skill-content", Markdown).source == text
        screen.query_one("#skills-filter", Input).value = "different"
        await pilot.pause()
        assert screen.query_one("#skills-list", OptionList).option_count == 1
        assert str(paths[0]) not in str(screen.query_one("#skill-source", Static).content)


async def test_grouped_read_rejects_changed_content_and_tries_other_copy() -> None:
    text = "Same skill"
    digest = hashlib.sha256(text.encode()).hexdigest()
    first = AsyncMock(return_value=json.dumps({"text": "Changed skill"}))
    second = AsyncMock(return_value=json.dumps({"text": text}))
    entries = [
        SkillEntry("demo", "Demo", "Local", "/first", first, digest),
        SkillEntry("demo", "Demo", "MCP: docs", "skill://second/SKILL.md", second, digest),
        SkillEntry("demo", "Demo", "Local", "/unknown", first),
    ]
    grouped = group_entries(entries)
    assert len(grouped) == 2
    group = next(entry for entry in grouped if entry.digest is not None)
    assert group.source == "/first, skill://second/SKILL.md"
    assert group.origin == "Local, MCP: docs"
    assert json.loads(await group.read())["text"] == text
    first.assert_awaited_once()
    second.assert_awaited_once()
    second.side_effect = ValueError("unavailable")
    with pytest.raises(ValueError, match="Skill content changed"):
        await group.read()
