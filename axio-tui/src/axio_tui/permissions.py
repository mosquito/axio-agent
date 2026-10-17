"""TUI permission persistence and screens. Policy lives in axio.permission_policy."""

from __future__ import annotations

from typing import Any

from axio.permission_policy import Approval, PermissionRequest, PermissionRule, PolicyGuard
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList, Select, Static

from .sqlite_config import ProjectConfig

RULE_PREFIX = "permissions.v1."


async def prompt_permission(app: App[Any], request: PermissionRequest) -> Approval:
    """Await a worker-owned prompt and remove it when that worker is cancelled."""
    dialog = PermissionDialog(request)
    try:
        return await app.push_screen_wait(dialog)
    finally:
        if dialog in app.screen_stack:
            while app.screen is not dialog:
                app.pop_screen()
            app.pop_screen()


class ConfigRuleStore:
    """One SQLite row per rule avoids overwriting unrelated concurrent changes."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    async def load(self) -> list[PermissionRule]:
        rules = []
        for key, value in (await self.config.get_prefix(RULE_PREFIX)).items():
            rule = PermissionRule.from_json(value)
            if key != RULE_PREFIX + rule.id:
                raise ValueError("Permission rule ID does not match its storage key")
            rules.append(rule)
        return rules

    async def add(self, rule: PermissionRule) -> None:
        # Validate before writing. Only project rules belong in durable storage.
        value = rule.to_json()
        PermissionRule.from_json(value)
        await self.config.set(RULE_PREFIX + rule.id, value)

    async def remove(self, rule_id: str) -> None:
        await self.config.delete(RULE_PREFIX + rule_id)


class PermissionDialog(ModalScreen[Approval]):
    BINDINGS = [Binding("escape", "deny", "Deny")]
    CSS = """
    PermissionDialog { align: center middle; }
    #permission-dialog { width: 90%; max-width: 110; height: 90%; border: heavy $warning; padding: 1 2; }
    #permission-details { height: 1fr; }
    #permission-dialog Select { height: 3; }
    #permission-buttons { height: 3; }
    #permission-buttons Button { margin-right: 2; }
    #permission-scope { margin-top: 1; }
    """

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Container(id="permission-dialog"):
            yield Label(f"Permission: {self.request.tool}", markup=False)
            with VerticalScroll(id="permission-details"):
                yield Static(self.request.reason, markup=False)
                yield Static(f"Working directory: {self.request.cwd}", markup=False)
                yield Static(f"Arguments:\n{self.request.arguments}", markup=False)
                yield Static(self.request.describe_scope(), markup=False, id="permission-scope")
            yield Select(
                [("This call only", "once"), ("Until this chat ends", "chat"), ("Save for this project", "project")],
                value="once",
                allow_blank=False,
                id="permission-lifetime",
            )
            options = [("Exact paths and operations", "exact"), ("Parent directories and descendants", "directories")]
            if not self.request.files_only:
                options = [("Exact tool, arguments, and working directory", "exact")]
            yield Select(options, value="exact", allow_blank=False, id="permission-target", disabled=True)
            with Horizontal(id="permission-buttons"):
                yield Button("Allow once", id="permission-allow", variant="success")
                yield Button("Deny once", id="permission-deny", variant="error")

    def on_mount(self) -> None:
        self.query_one("#permission-deny", Button).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        lifetime = self.query_one("#permission-lifetime", Select).value
        target = self.query_one("#permission-target", Select)
        target.disabled = lifetime == "once" or not self.request.files_only
        directories = lifetime != "once" and target.value == "directories" and self.request.files_only
        self.query_one("#permission-scope", Static).update(self.request.describe_scope(directories=directories))
        suffix = {"once": "once", "chat": "for this chat", "project": "for this project"}.get(str(lifetime), "once")
        self.query_one("#permission-allow", Button).label = f"Allow {suffix}"
        self.query_one("#permission-deny", Button).label = f"Deny {suffix}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id not in {"permission-allow", "permission-deny"}:
            return
        lifetime = self.query_one("#permission-lifetime", Select).value
        if lifetime not in {"once", "chat", "project"}:
            return
        self.dismiss(
            Approval(
                "allow" if event.button.id == "permission-allow" else "deny",
                lifetime,
                lifetime != "once"
                and self.request.files_only
                and self.query_one("#permission-target", Select).value == "directories",
            )
        )

    def action_deny(self) -> None:
        self.dismiss(Approval())


class PermissionsScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close")]
    CSS = """
    PermissionsScreen { align: center middle; }
    #permissions { width: 90%; height: 90%; border: heavy $accent; padding: 1 2; }
    #permission-rules { height: 1fr; }
    #rule-details-scroll { height: 1fr; }
    #rules-buttons { height: 3; }
    #rules-buttons Button { margin-right: 2; }
    """

    def __init__(self, policy: PolicyGuard, store: ConfigRuleStore) -> None:
        super().__init__()
        self.policy = policy
        self.store = store
        self.entries: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        with Container(id="permissions"):
            yield Label("Permissions")
            yield Static(
                f"Project: {self.policy.project_root}\n"
                "Read-only access inside this project is automatic. Other actions require a rule or confirmation.\n"
                "Denials take priority. Revocation affects subsequent calls; it does not stop running tools.",
                markup=False,
            )
            yield OptionList(id="permission-rules")
            with VerticalScroll(id="rule-details-scroll"):
                yield Static("", markup=False, id="rule-details")
            with Horizontal(id="rules-buttons"):
                yield Button("Revoke selected", id="rule-revoke", variant="error")
                yield Button("Refresh", id="rules-refresh")
                yield Button("Close", id="rules-close")

    async def on_mount(self) -> None:
        await self.refresh_rules()

    async def refresh_rules(self) -> None:
        self.entries = []
        try:
            for key, value in (await self.store.config.get_prefix(RULE_PREFIX)).items():
                try:
                    rule = PermissionRule.from_json(value)
                    if key != RULE_PREFIX + rule.id:
                        raise ValueError("ID mismatch")
                    detail = rule.describe()
                except Exception:
                    detail = "Invalid saved rule. Permission checks remain blocked until this entry is removed."
                self.entries.append((key.removeprefix(RULE_PREFIX), detail))
            # The policy returns chat rules as well as saved rules.
            try:
                self.entries.extend((r.id, r.describe()) for r in await self.policy.rules() if r.lifetime == "chat")
            except Exception:
                pass  # Invalid rows above remain individually removable.
        except Exception as error:
            self.query_one("#rule-details", Static).update(f"Cannot load rules: {error}")
            return
        options = self.query_one("#permission-rules", OptionList)
        options.clear_options()
        for _, detail in self.entries:
            options.add_option(Text(detail.splitlines()[0]))
        self.query_one("#rule-revoke", Button).disabled = not self.entries
        self.query_one("#rule-details", Static).update(self.entries[0][1] if self.entries else "No saved rules.")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index < len(self.entries):
            self.query_one("#rule-details", Static).update(self.entries[event.option_index][1])

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rules-close":
            self.dismiss(None)
        elif event.button.id == "rules-refresh":
            await self.refresh_rules()
        elif event.button.id == "rule-revoke":
            index = self.query_one("#permission-rules", OptionList).highlighted
            if index is not None and index < len(self.entries):
                try:
                    await self.policy.revoke(self.entries[index][0])
                except Exception as error:
                    self.query_one("#rule-details", Static).update(f"Cannot revoke rule: {error}")
                    return
                await self.refresh_rules()

    def action_close(self) -> None:
        self.dismiss(None)
