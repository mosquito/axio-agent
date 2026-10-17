"""Browse skill metadata and read instructions without adding chat messages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

from axio.instructions import Instruction

if TYPE_CHECKING:
    from axio_tools_local.skills import LocalSkills
    from axio_tools_mcp.runtime import MCPRuntime
    from axio_tools_mcp.skills import SkillCatalog
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.events import Key, Resize
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Markdown, OptionList, Static


@dataclass(frozen=True, slots=True)
class SkillEntry:
    name: str
    description: str
    origin: str
    source: str
    read: Callable[[], Awaitable[str]]
    digest: str | None = None


def instruction_entry(origin: str, instruction: Instruction) -> SkillEntry:
    async def read() -> str:
        loaded = await instruction.load()
        return json.dumps({"text": loaded.text})

    return SkillEntry(instruction.id, instruction.summary, origin, instruction.source, read)


def catalog_entries(origin: str, catalog: SkillCatalog) -> list[SkillEntry]:
    return [
        SkillEntry(
            skill.name,
            skill.description,
            origin,
            skill.uri,
            partial(catalog.read, skill.uri),
            next((item.digest.removeprefix("sha256:") for item in skill.resources if item.uri == skill.uri), None),
        )
        for skill in catalog.skills.values()
    ]


def provider_entries(
    skills: LocalSkills | None = None, mcp: MCPRuntime | None = None
) -> tuple[list[SkillEntry], list[str]]:
    entries: list[SkillEntry] = []
    notes: list[str] = []
    if skills is not None:
        from axio_tools_local.skills import content_digest

        entries = [
            SkillEntry(
                skill.name,
                skill.description,
                "Local",
                str(skill.source),
                partial(skills.read, location),
                content_digest(skill.source),
            )
            for location, skill in skills.catalog.skills.items()
        ]
        notes = list(skills.catalog.diagnostics)
    for server in mcp.servers if mcp is not None else ():
        if server.skill_error:
            notes.append(f"{server.alias}: skill discovery failed ({server.skill_error}); use /mcp reload")
        if server.state == "connected" and server.skills is not None:
            entries.extend(catalog_entries(f"MCP: {server.alias}", server.skills))
    return entries, notes


async def read_group(entries: tuple[SkillEntry, ...]) -> str:
    """Read one available copy, checking that its full content still matches the group."""
    errors = []
    for entry in entries:
        try:
            payload = await entry.read()
            text = json.loads(payload)["text"]
            if not isinstance(text, str):
                raise ValueError("Skill content must be text")
            if entry.digest != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError("Skill content changed; reload the skills catalog")
            return payload
        except Exception as error:
            errors.append(f"{entry.source}: {error}")
    raise ValueError("; ".join(errors))


def group_entries(entries: list[SkillEntry]) -> list[SkillEntry]:
    """Group only known full-content digests; unknown or failed hashes stay separate."""
    groups: dict[str, list[SkillEntry]] = {}
    result = []
    for entry in entries:
        if entry.digest is None:
            result.append(entry)
        else:
            groups.setdefault(entry.digest, []).append(entry)
    for copies in groups.values():
        result.append(
            replace(
                copies[0],
                name=", ".join(dict.fromkeys(entry.name for entry in copies)),
                description="\n\n".join(dict.fromkeys(entry.description for entry in copies)),
                origin=", ".join(dict.fromkeys(entry.origin for entry in copies)),
                source=", ".join(dict.fromkeys(entry.source for entry in copies)),
                read=partial(read_group, tuple(copies)),
            )
        )
    return result


class SkillsScreen(ModalScreen[None]):
    """Filter skills by metadata and load only the instructions the user opens."""

    BINDINGS = [Binding("escape", "close", "Close"), Binding("ctrl+f", "filter", "Filter")]
    CSS = """
    SkillsScreen { align: center middle; }
    SkillsScreen #skills-dialog {
        width: 95%; max-width: 150; height: 90%;
        border: heavy $accent; background: $panel; padding: 0 1;
    }
    SkillsScreen #skills-title { text-style: bold; height: 1; }
    SkillsScreen #skills-count { height: 1; color: $text-muted; }
    SkillsScreen #skills-body { height: 1fr; }
    SkillsScreen #skills-list { width: 2fr; height: 1fr; }
    SkillsScreen #skills-details { width: 3fr; height: 1fr; padding: 0 1; }
    SkillsScreen #skill-name { text-style: bold; margin-bottom: 1; }
    SkillsScreen #skill-source { color: $text-muted; margin-bottom: 1; }
    SkillsScreen #skill-description { margin-bottom: 1; }
    SkillsScreen #skill-status { color: $text-muted; }
    SkillsScreen #skills-buttons { height: 3; align-horizontal: right; }
    SkillsScreen #skills-buttons Button { margin-left: 1; }
    SkillsScreen #skills-notes { height: auto; max-height: 4; }
    SkillsScreen.narrow #skills-dialog { width: 100%; height: 100%; }
    SkillsScreen.narrow #skills-body { layout: vertical; }
    SkillsScreen.narrow #skills-list { width: 1fr; height: 5; }
    SkillsScreen.narrow #skills-details { width: 1fr; height: 1fr; padding: 0; }
    """

    def __init__(self, entries: list[SkillEntry], notes: list[str] | None = None) -> None:
        super().__init__()
        self._entries = sorted(
            group_entries(entries), key=lambda entry: (entry.origin.casefold(), entry.name.casefold(), entry.source)
        )
        self._filtered = list(self._entries)
        self._notes = notes or []
        self._selected: SkillEntry | None = None
        self._contents: dict[SkillEntry, str] = {}
        self._loading: set[SkillEntry] = set()

    def compose(self) -> ComposeResult:
        with Container(id="skills-dialog"):
            yield Static("Skills", id="skills-title")
            yield Input(placeholder="Filter by name, description, or source", id="skills-filter")
            yield Static(id="skills-count")
            with Horizontal(id="skills-body"):
                yield OptionList(id="skills-list")
                with VerticalScroll(id="skills-details"):
                    yield Static(id="skill-name", markup=False)
                    yield Static(id="skill-source", markup=False)
                    yield Static(id="skill-description", markup=False)
                    yield Static(id="skill-status", markup=False)
                    yield Markdown(id="skill-content", open_links=False)
            if self._notes:
                with Collapsible(title="Discovery notes", collapsed=True):
                    with VerticalScroll(id="skills-notes"):
                        yield Static("\n\n".join(self._notes), markup=False)
            with Horizontal(id="skills-buttons"):
                yield Button("Open skill", id="skill-open", variant="primary", disabled=True)
                yield Button("Close", id="skills-close")

    def on_mount(self) -> None:
        self.set_class(self.size.width < 90, "narrow")
        self._filter("")
        self.action_filter()

    def on_resize(self, event: Resize) -> None:
        self.set_class(event.size.width < 90, "narrow")

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._filter(event.value)

    def _filter(self, query: str) -> None:
        query = query.casefold().strip()
        self._filtered = [
            entry
            for entry in self._entries
            if query in f"{entry.name}\n{entry.description}\n{entry.origin}\n{entry.source}".casefold()
        ]
        options = self.query_one("#skills-list", OptionList)
        options.clear_options()
        options.add_options([Text(f"{entry.name}\n{entry.origin}\n{entry.source}") for entry in self._filtered])
        self.query_one("#skills-count", Static).update(
            f"{len(self._filtered)}/{len(self._entries)} skills · ↑↓ browse · Tab details · Esc close"
        )
        options.highlighted = 0 if self._filtered else None
        self._show(self._filtered[0] if self._filtered else None)

    def _show(self, entry: SkillEntry | None) -> None:
        self._selected = entry
        self.query_one("#skill-open", Button).disabled = entry is None or entry in self._loading
        empty = "No skills match this filter." if self._entries else "No skills available."
        self.query_one("#skill-name", Static).update(entry.name if entry else empty)
        self.query_one("#skill-source", Static).update(f"{entry.origin}\n{entry.source}" if entry else "")
        self.query_one("#skill-description", Static).update(entry.description if entry else "")
        status = ""
        if entry is not None and entry not in self._contents:
            status = (
                "Loading instructions…"
                if entry in self._loading
                else "Press Enter or Open skill to read instructions."
            )
        self.query_one("#skill-status", Static).update(status)
        self.query_one("#skill-content", Markdown).update(self._contents.get(entry, "") if entry else "")
        self.query_one("#skills-details", VerticalScroll).scroll_home(animate=False)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        if event.option_index < len(self._filtered):
            self._show(self._filtered[event.option_index])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option_index < len(self._filtered):
            self._show(self._filtered[event.option_index])
            self._open()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._filtered:
            self.query_one("#skills-list", OptionList).focus()
            self._open()

    def on_key(self, event: Key) -> None:
        if event.key == "down" and self.focused is self.query_one("#skills-filter", Input):
            self.query_one("#skills-list", OptionList).focus()
            event.prevent_default()
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "skills-close":
            self.action_close()
        elif event.button.id == "skill-open":
            self._open()

    def _open(self) -> None:
        entry = self._selected
        if entry is None or entry in self._loading:
            return
        if entry in self._contents:
            self.query_one("#skills-details", VerticalScroll).focus()
            return
        self._loading.add(entry)
        self._show(entry)
        self.run_worker(self._read(entry), group="skill-read")

    async def _read(self, entry: SkillEntry) -> None:
        error = ""
        try:
            payload = json.loads(await entry.read())
            text = payload["text"]
            if not isinstance(text, str):
                raise ValueError("Skill content must be text")
            self._contents[entry] = text
        except Exception as exc:
            error = f"Cannot read skill: {exc}"
        finally:
            self._loading.discard(entry)
        if self._selected is entry:
            self._show(entry)
            if error:
                self.query_one("#skill-status", Static).update(error)

    def action_filter(self) -> None:
        self.query_one("#skills-filter", Input).focus()

    def action_close(self) -> None:
        self.dismiss(None)
