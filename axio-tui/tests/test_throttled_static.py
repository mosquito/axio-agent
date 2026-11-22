"""Tests for the repaint rate of the widgets written from the event stream."""

from __future__ import annotations

from textual.app import App, ComposeResult

from axio_tui.app import _ThrottledStatic, _ToolStatusWidget


class _CountingStatic(_ThrottledStatic):
    """Counts its repaints, which is what the throttle is there to limit."""

    def __init__(self) -> None:
        super().__init__("")
        self.paints = 0

    def update(self, *args: object, **kwargs: object) -> None:
        self.paints += 1
        super().update(*args, **kwargs)  # type: ignore[arg-type]


class _ThrottleApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.widget = _CountingStatic()

    def compose(self) -> ComposeResult:
        yield self.widget


class TestTheRepaintThrottle:
    """A delta changes what the widget holds. A timer puts it on screen."""

    async def test_many_deltas_paint_once(self) -> None:
        # Repainting per delta reparsed the markup and refreshed the widget on every token,
        # which starved the message loop and delayed every keypress.
        app = _ThrottleApp()
        async with app.run_test() as pilot:
            app.widget.paints = 0

            for index in range(500):
                app.widget.draw(f"delta {index}")
            await pilot.pause(1 / _ThrottledStatic.REPAINT_HZ * 2)

            assert app.widget.paints == 1
            assert app.widget.shown == "delta 499"

    async def test_an_unchanged_draw_paints_nothing(self) -> None:
        app = _ThrottleApp()
        async with app.run_test() as pilot:
            app.widget.draw("same")
            await pilot.pause(1 / _ThrottledStatic.REPAINT_HZ * 2)
            app.widget.paints = 0

            app.widget.draw("same")
            await pilot.pause(1 / _ThrottledStatic.REPAINT_HZ * 2)

            assert app.widget.paints == 0

    def test_the_content_is_readable_before_the_paint(self) -> None:
        # The tool line is written to before it is mounted, so `shown` must hold the content at
        # once rather than wait for a tick that an unmounted widget never gets.
        tools = _ToolStatusWidget()

        tools.track("t1", "shell")

        assert "shell" in tools.shown
