"""Tests for how the answer being written reaches the transcript."""

from __future__ import annotations

import asyncio

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from axio_tui.app import AgentApp


async def test_the_answer_streams_into_one_markdown_widget(isolated_app: AgentApp) -> None:
    # `Markdown.update` reparses and remounts the whole document, so a periodic update of the
    # accumulated answer cost more the longer the answer got. The stream appends instead.
    app = isolated_app
    async with app.run_test() as pilot:
        await asyncio.wait_for(app.ready.wait(), 5)
        log = app.query_one("#log", VerticalScroll)
        before = len(log.children)

        for fragment in ("# Title\n\n", "first ", "paragraph.\n\n", "second paragraph."):
            await app._write_text(fragment)
        await app._close_answer()
        await pilot.pause()

        answers = [child for child in log.children if isinstance(child, Markdown)]
        assert len(log.children) == before + 1
        assert answers[-1].source == "# Title\n\nfirst paragraph.\n\nsecond paragraph."


async def test_a_mounted_line_goes_below_the_whole_answer(isolated_app: AgentApp) -> None:
    # The stream holds fragments it has not applied yet. Mounted without draining them, the line
    # appeared above the end of the answer it was about.
    app = isolated_app
    async with app.run_test() as pilot:
        await asyncio.wait_for(app.ready.wait(), 5)
        log = app.query_one("#log", VerticalScroll)

        await app._write_text("the answer")
        await app._write_meta("[dim]--- done ---[/]")
        await pilot.pause()

        answer, line = log.children[-2], log.children[-1]
        assert isinstance(answer, Markdown) and answer.source == "the answer"
        assert isinstance(line, Static)


async def test_the_next_answer_opens_a_new_widget(isolated_app: AgentApp) -> None:
    app = isolated_app
    async with app.run_test() as pilot:
        await asyncio.wait_for(app.ready.wait(), 5)
        log = app.query_one("#log", VerticalScroll)

        await app._write_text("first turn")
        await app._close_answer()
        await app._write_text("second turn")
        await app._close_answer()
        await pilot.pause()

        answers = [child for child in log.children if isinstance(child, Markdown)]
        assert [answer.source for answer in answers[-2:]] == ["first turn", "second turn"]


async def test_the_log_follows_the_answer(isolated_app: AgentApp) -> None:
    # Scrolling to the end on every change took the view back from a user who scrolled up.
    # An anchor follows new content and is released as soon as the user scrolls.
    app = isolated_app
    async with app.run_test() as pilot:
        await asyncio.wait_for(app.ready.wait(), 5)
        log = app.query_one("#log", VerticalScroll)

        await app._write_text("an answer")
        await pilot.pause()

        assert log.is_anchored
