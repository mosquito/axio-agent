"""Settings screens for Google GenAI / Vertex AI transports."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class _SharedFields:
    """Mixin: shared generation parameter fields and save logic."""

    _settings: dict[str, str]

    def _compose_shared(self) -> ComposeResult:
        yield Static("Temperature:", classes="field-label")
        yield Input(value=self._settings.get("temperature", ""), id="temperature", placeholder="0.0 - 2.0")

        yield Static("Thinking budget:", classes="field-label")
        yield Input(
            value=self._settings.get("thinking_budget", ""),
            id="thinking-budget",
            placeholder="token count (0=off, -1=auto)",
        )

        yield Static("Service tier:", classes="field-label")
        yield Input(
            value=self._settings.get("service_tier", ""),
            id="service-tier",
            placeholder="flex / standard / priority",
        )

    def _collect_shared(self, result: dict[str, str]) -> None:
        for field_id in ("temperature", "service-tier"):
            val = self.query_one(f"#{field_id}", Input).value.strip()  # type: ignore[attr-defined]
            if val:
                result[field_id.replace("-", "_")] = val
        tb = self.query_one("#thinking-budget", Input).value.strip()  # type: ignore[attr-defined]
        if tb:
            result["thinking_budget"] = tb


_CSS = """
    #google-settings {
        width: 70; height: auto; border: heavy $accent;
        background: $panel; padding: 1 2;
    }
    .field-label { margin-top: 1; }
    .settings-buttons { height: auto; margin-top: 1; }
    .settings-buttons Button { margin: 0 1; }
"""


class GoogleSettingsScreen(_SharedFields, ModalScreen[dict[str, str] | None]):
    """Settings for Google GenAI (Gemini API key) transport."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = f"GoogleSettingsScreen {{ align: center middle; }}\n{_CSS}"

    def __init__(self, settings: dict[str, str]) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        with Container(id="google-settings"):
            yield Static("[bold]Google GenAI Settings[/]")

            yield Static("API Key (GEMINI_API_KEY):", classes="field-label")
            yield Input(
                value=self._settings.get("api_key", ""),
                id="api-key",
                password=True,
            )

            yield from self._compose_shared()

            with Horizontal(classes="settings-buttons"):
                yield Button("Save", id="btn-save", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#api-key", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            result: dict[str, str] = {}
            api_key = self.query_one("#api-key", Input).value.strip()
            if api_key:
                result["api_key"] = api_key
            self._collect_shared(result)
            self.dismiss(result)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class VertexSettingsScreen(_SharedFields, ModalScreen[dict[str, str] | None]):
    """Settings for Google Vertex AI transport."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = f"VertexSettingsScreen {{ align: center middle; }}\n{_CSS}"

    def __init__(self, settings: dict[str, str]) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        with Container(id="google-settings"):
            yield Static("[bold]Google Vertex AI Settings[/]")

            yield Static("Project:", classes="field-label")
            yield Input(value=self._settings.get("project", ""), id="project", placeholder="GCP project ID")

            yield Static("Location:", classes="field-label")
            yield Input(value=self._settings.get("location", ""), id="location", placeholder="e.g. us-central1")

            yield from self._compose_shared()

            with Horizontal(classes="settings-buttons"):
                yield Button("Save", id="btn-save", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#project", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            result: dict[str, str] = {"vertexai": "true"}
            for field_id in ("project", "location"):
                val = self.query_one(f"#{field_id}", Input).value.strip()
                if val:
                    result[field_id] = val
            self._collect_shared(result)
            self.dismiss(result)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
