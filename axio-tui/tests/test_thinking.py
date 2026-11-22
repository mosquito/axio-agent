"""Tests for the reasoning pane: live streaming, folding and reading it again."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Markdown

from axio_tui.app import _ThinkingWidget


class _ThinkingApp(App[None]):
    """One reasoning pane in a log, which is where the application puts it."""

    def __init__(self) -> None:
        super().__init__()
        self.widget = _ThinkingWidget()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="log"):
            yield self.widget


class TestTheReasoningPane:
    """Reasoning streams into an expanded pane and folds to one line on the answer."""

    async def test_it_streams_the_reasoning_in_full(self) -> None:
        # The pane kept the last 6 lines and dropped the rest, so long reasoning could not be
        # read back at all.
        app = _ThinkingApp()
        async with app.run_test() as pilot:
            for at in range(200):
                await app.widget.add(f"line {at}\n")
            await pilot.pause()

            body = app.widget.query_one(Markdown)
            assert "line 0" in body.source
            assert "line 199" in body.source

    async def test_it_is_expanded_while_the_model_reasons(self) -> None:
        app = _ThinkingApp()
        async with app.run_test() as pilot:
            await app.widget.add("reasoning")
            await pilot.pause()

            assert not app.widget.collapsed
            assert "thinking…" in app.widget.title

    async def test_the_pane_follows_its_own_tail(self) -> None:
        app = _ThinkingApp()
        async with app.run_test() as pilot:
            await app.widget.add("reasoning\n" * 40)
            await pilot.pause()

            assert app.widget.query_one("#thinking-body", VerticalScroll).is_anchored

    async def test_the_answer_folds_it_and_keeps_the_reasoning(self) -> None:
        app = _ThinkingApp()
        async with app.run_test() as pilot:
            await app.widget.add("a lot of reasoning")

            await app.widget.collapse(938)
            await pilot.pause()

            assert app.widget.collapsed
            assert "938 tokens" in app.widget.title
            assert "expand" in app.widget.title
            assert app.widget.query_one(Markdown).source == "a lot of reasoning"

    async def test_the_folded_line_reports_how_long_it_took(self) -> None:
        app = _ThinkingApp()
        async with app.run_test():
            await app.widget.add("reasoning")

            await app.widget.collapse()

            assert "thought for" in app.widget.title

    async def test_the_count_arrives_after_the_fold(self) -> None:
        # reasoning_tokens comes with IterationEnd, which is after the answer has started and the
        # pane has already folded. `collapse` is the one way in: it keeps the figure it has when
        # called without one, and redraws when a later call brings it.
        app = _ThinkingApp()
        async with app.run_test():
            await app.widget.add("x")
            await app.widget.collapse()

            await app.widget.collapse(42)

            assert "42 tokens" in app.widget.title
            assert app.widget.collapsed

    async def test_it_can_be_read_again_after_it_folded(self) -> None:
        app = _ThinkingApp()
        async with app.run_test() as pilot:
            await app.widget.add("the reasoning")
            await app.widget.collapse(10)
            await pilot.pause()

            app.widget.collapsed = False
            await pilot.pause()

            assert app.widget.query_one(Markdown).display

    async def test_it_opens_at_the_first_line_once_it_folded(self) -> None:
        # Watched, the pane follows its own tail. Read afterwards, it starts where the reasoning
        # starts and shows more of it at once.
        app = _ThinkingApp()
        async with app.run_test() as pilot:
            await app.widget.add("reasoning\n\n" * 40)
            await app.widget.collapse(10)
            await pilot.pause()

            body = app.widget.query_one("#thinking-body", VerticalScroll)
            assert not body.is_anchored
            assert body.scroll_y == 0
            assert body.has_class("done")

    async def test_it_says_when_the_provider_billed_for_thinking_and_sent_none(self) -> None:
        # OpenAI withholds reasoning summaries from an unverified organisation. Silence there is
        # indistinguishable from a model that did not reason, and the tokens are billed either way.
        app = _ThinkingApp()
        async with app.run_test():
            await app.widget.collapse(938)

            assert app.widget.withheld
            assert "not sent by the provider" in app.widget.title

    async def test_it_says_nothing_of_the_sort_when_the_reasoning_arrived(self) -> None:
        app = _ThinkingApp()
        async with app.run_test():
            await app.widget.add("some thinking")

            await app.widget.collapse(938)

            assert not app.widget.withheld
            assert "not sent" not in app.widget.title

    async def test_an_iteration_with_no_reasoning_at_all_is_empty(self) -> None:
        # A tool round that neither reasoned nor was billed for it. Shown, a bare line says
        # nothing and reads as a second turn.
        app = _ThinkingApp()
        async with app.run_test():
            await app.widget.collapse(0)

            assert app.widget.empty
            assert not app.widget.withheld

    async def test_a_pane_with_something_to_report_is_not_empty(self) -> None:
        billed, spoken = _ThinkingApp(), _ThinkingApp()
        async with billed.run_test():
            await billed.widget.collapse(120)
        async with spoken.run_test():
            await spoken.widget.add("brief")
            await spoken.widget.collapse(0)

        assert not billed.widget.empty and not spoken.widget.empty
