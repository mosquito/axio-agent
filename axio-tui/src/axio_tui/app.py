"""AgentApp: Textual TUI for the Axio coding-assist agent."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import aiosqlite
from axio.agent import Agent
from axio.compaction import compact_context
from axio.context import ContextStore, MemoryContextStore, SessionInfo
from axio.events import (
    Citation,
    Error,
    IterationEnd,
    ReasoningDelta,
    Refusal,
    SessionEndEvent,
    TextDelta,
    ToolFieldDelta,
    ToolFieldEnd,
    ToolFieldStart,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.exceptions import GuardError
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.permission import PermissionGuard
from axio.permission_policy import Approval, PermissionRequest, PolicyGuard
from axio.plugins import (
    ContextBackend,
    RefreshableToolProvider,
    ToolProvider,
    collect_tools,
    discover_context_backends,
    discover_tool_providers,
    standalone_tools,
)
from axio.plugins import discover_tools as discover_tool_catalog
from axio.selector import ToolSelector
from axio.tool import Tool
from axio.tool_args import ToolArgStream
from axio.types import INCOMPLETE
from pygments.token import Token  # type: ignore[import-untyped]
from rich.markup import escape
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.content import Content
from textual.events import Key
from textual.highlight import HighlightTheme
from textual.highlight import highlight as hl_highlight
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Markdown, OptionList, RichLog, Static
from textual.widgets._markdown import MarkdownFence

from axio_tui.plugin import (
    ToolsPlugin,
    discover_guards,
    discover_selectors,
    discover_tools_by_package,
    discover_tools_plugins,
)
from axio_tui.transport_registry import RoleBinding, TransportRegistry

from . import tools as _tools
from .mcp import MCPScreen
from .options import StartupOptions
from .paths import ConfigPaths, ConfigurationPaths
from .permissions import ConfigRuleStore, PermissionsScreen, prompt_permission
from .prompt import LAST_ITERATION_MESSAGE, SYSTEM_PROMPT
from .screens import ModelSelectScreen, PluginHubScreen, QuitDialog, SessionSelectScreen, ToolDetailScreen
from .skills import SkillEntry, SkillsScreen, catalog_entries, instruction_entry, provider_entries
from .sqlite_config import GLOBAL_PROJECT, ProjectConfig, connect_config
from .tools import SUBAGENT_SYSTEM_PROMPT, _short, confirm

if TYPE_CHECKING:
    from axio_tools_local.provider import LocalBinding
    from axio_tools_mcp.inspection import ServerInspection

    from .configuration import MCPConfiguration

logger = logging.getLogger(__name__)


class _MonokaiTheme(HighlightTheme):
    """Monokai-inspired syntax highlighting for code blocks."""

    STYLES: dict[tuple[str, ...], str] = {
        Token.Comment: "#75715e italic",
        Token.Error: "#f92672 on #49483e",
        Token.Generic.Strong: "bold",
        Token.Generic.Emph: "italic",
        Token.Generic.Error: "#f92672",
        Token.Generic.Heading: "#e6db74 bold",
        Token.Generic.Subheading: "#e6db74",
        Token.Keyword: "#f92672",
        Token.Keyword.Constant: "#66d9ef italic",
        Token.Keyword.Namespace: "#f92672",
        Token.Keyword.Type: "#66d9ef italic",
        Token.Literal.Number: "#ae81ff",
        Token.Literal.String: "#e6db74",
        Token.Literal.String.Backtick: "#e6db74",
        Token.Literal.String.Doc: "#e6db74 italic",
        Token.Literal.String.Double: "#e6db74",
        Token.Name: "#f8f8f2",
        Token.Name.Attribute: "#a6e22e",
        Token.Name.Builtin: "#66d9ef",
        Token.Name.Builtin.Pseudo: "#66d9ef italic",
        Token.Name.Class: "#a6e22e bold",
        Token.Name.Constant: "#66d9ef",
        Token.Name.Decorator: "#a6e22e bold",
        Token.Name.Function: "#a6e22e",
        Token.Name.Function.Magic: "#a6e22e",
        Token.Name.Tag: "#f92672",
        Token.Name.Variable: "#f8f8f2",
        Token.Number: "#ae81ff",
        Token.Operator: "#f92672",
        Token.Operator.Word: "#f92672",
        Token.String: "#e6db74",
        Token.Whitespace: "",
    }


class _MonokaiFence(MarkdownFence):
    """MarkdownFence with Monokai highlighting."""

    @classmethod
    def highlight(cls, code: str, language: str, ansi: bool = False, dark: bool = False) -> Content:
        return hl_highlight(code, language=language or None, theme=_MonokaiTheme)


class _Markdown(Markdown):
    """Markdown widget with Monokai code-block highlighting."""

    BLOCKS = {**Markdown.BLOCKS, "fence": _MonokaiFence, "code_block": _MonokaiFence}


def _citation_markdown(title: str | None, cited: str, url: str | None) -> str:
    """Render one citation as a Markdown link.

    A `]` in a provider's title or a `)` in its query string closes the link early and spills the
    rest into the transcript, so the label is escaped and the target is encoded.
    """
    # The backslash goes first, or a title ending in one escapes the bracket that closes the label.
    label = (title or cited or url or "source").replace("\\", r"\\").replace("[", r"\[").replace("]", r"\]")
    target = (url or "").replace("(", "%28").replace(")", "%29").replace(" ", "%20")
    return f" [{label}]({target})" if url else f" _[{label}]_"


@dataclass
class _ToolCallInfo:
    name: str = ""
    status: bool | None = None  # None=pending, True=ok, False=error
    input: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    live_args: dict[str, str] = field(default_factory=dict)  # live streaming args


class _ThinkingWidget(Static):
    """The model's reasoning while it lasts, and one line about it afterwards.

    Kept out of the transcript. Left in, a model that reasons at length buries the answer it was
    reasoning towards, and scrolling back through the conversation stops being useful.
    """

    #: How much of the reasoning stays on screen while it streams.
    LINES = 6

    DEFAULT_CSS = """
    _ThinkingWidget {
        height: auto;
        margin: 0;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__("", classes="meta")
        #: What was last drawn. Kept because Static gives no way to read it back.
        self.shown = ""
        #: Only the lines this widget shows. The whole reasoning was kept and split again on every
        #: delta, so a model that reasons at length paid for its own output once per chunk.
        self._tail: list[str] = [""]
        #: How much reasoning arrived, which is all the two properties below need of it.
        self._chars = 0
        self._done = False
        self._tokens = 0

    def add(self, delta: str) -> None:
        self._chars += len(delta)
        lines = delta.split("\n")
        self._tail[-1] += lines[0]
        self._tail.extend(lines[1:])
        del self._tail[: -self.LINES]
        self._draw()

    def collapse(self, tokens: int = 0) -> None:
        """Fold to one line, which is what the answer arriving means."""
        self._done, self._tokens = True, tokens or self._tokens
        self._draw()

    @property
    def empty(self) -> bool:
        """Nothing to show and nothing to report, so the marker would say nothing either."""
        return not self._chars and not self._tokens

    @property
    def withheld(self) -> bool:
        """The provider charged for reasoning and sent none of it.

        Its own choice, not a fault here: OpenAI withholds reasoning summaries from an
        unverified organisation. Worth saying, because the tokens are billed either way.
        """
        return bool(self._tokens) and not self._chars

    def _draw(self) -> None:
        if self._done:
            cost = f" · {self._tokens:,} tokens" if self._tokens else ""
            note = ", not sent by the provider" if self.withheld else ""
            self.shown = f"[dim]✻ thinking{cost}{note}[/]"
        else:
            body = "\n".join(f"[dim]{escape(line)}[/]" for line in self._tail)
            self.shown = f"[dim italic]✻ thinking…[/]\n{body}"
        self.update(self.shown)


class _ToolStatusWidget(Static):
    """Single-line live-updating tool status tracker with clickable completed tools."""

    DEFAULT_CSS = """
    _ToolStatusWidget {
        height: auto;
        margin: 0;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__("", classes="meta")
        self._tools: dict[str, _ToolCallInfo] = {}  # keyed by tool_use_id

    def track(self, tool_use_id: str, name: str) -> None:
        self._tools[tool_use_id] = _ToolCallInfo(name=name)
        self._refresh_content()

    def update_arg(self, tool_use_id: str, event: ToolFieldStart | ToolFieldDelta | ToolFieldEnd) -> None:
        """Update a field being streamed from ToolArgStream events."""
        from axio.events import ToolFieldDelta, ToolFieldEnd, ToolFieldStart

        if tool_use_id not in self._tools:
            return
        info = self._tools[tool_use_id]

        if isinstance(event, ToolFieldStart):
            # Start a new field
            info.live_args[event.key] = ""
        elif isinstance(event, ToolFieldDelta):
            # Append to existing field
            current = info.live_args.get(event.key, "")
            info.live_args[event.key] = current + event.text
        elif isinstance(event, ToolFieldEnd):
            # Finalize field - convert to final value
            pass  # Already updated via delta

        self._refresh_content()

    def complete(
        self, tool_use_id: str, *, is_error: bool, content: str = "", tool_input: dict[str, Any] | None = None
    ) -> None:
        if tool_use_id in self._tools:
            info = self._tools[tool_use_id]
            info.status = not is_error
            info.content = content
            info.input = tool_input or {}
        self._refresh_content()

    def _refresh_content(self) -> None:
        parts: list[str] = []
        for tid, info in self._tools.items():
            if info.status is None:
                # Show live args while pending
                args_parts = [f"{k}={_short(v, 30)}" for k, v in info.live_args.items()]
                args_str = f" [dim]({', '.join(args_parts)})[/]" if args_parts else ""
                parts.append(f"[yellow]{info.name} ⏳{args_str}[/]")
            elif info.status:
                parts.append(f"[@click=show_detail('{tid}')][green]{info.name} ✓[/][/]")
            else:
                parts.append(f"[@click=show_detail('{tid}')][red]{info.name} ✗[/][/]")
        self.update("[dim]tools:[/] " + "  ".join(parts))

    def action_show_detail(self, tool_use_id: str) -> None:
        info = self._tools.get(tool_use_id)
        if info is None:
            return
        is_error = info.status is not None and not info.status
        self.app.push_screen(ToolDetailScreen(info.name, info.input, info.content, is_error))


class ModelRole(StrEnum):
    CHAT = "chat"
    COMPACT = "compact"
    SUBAGENT = "subagent"
    GUARD = "guard"
    VISION = "vision"
    REASONING = "reasoning"


class ModelRoleSelectScreen(ModalScreen[ModelRole | None]):
    """Modal screen for selecting which model role to configure."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    ModelRoleSelectScreen { align: center middle; }
    #role-select {
        width: 70;
        height: auto;
        border: heavy $accent;
        background: $panel;
        padding: 1 2;
    }
    #role-list { height: auto; }
    """

    def __init__(self, role_bindings: dict[ModelRole, RoleBinding]) -> None:
        super().__init__()
        self._role_bindings = role_bindings

    def compose(self) -> ComposeResult:
        with Container(id="role-select"):
            yield Static("[bold]Select Model Role[/]")
            options: list[str] = []
            for role in ModelRole:
                label = _ROLE_LABELS[role]
                if role in self._role_bindings:
                    b = self._role_bindings[role]
                    options.append(f"{label:<12} [{b.transport}] {b.model.id}")
                else:
                    options.append(f"{label:<12} (not configured)")
            yield OptionList(*options, id="role-list")

    def on_mount(self) -> None:
        self.query_one("#role-list", OptionList).focus()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        roles = list(ModelRole)
        idx = message.option_index
        if 0 <= idx < len(roles):
            self.dismiss(roles[idx])

    def action_cancel(self) -> None:
        self.dismiss(None)


_ROLE_LABELS: dict[ModelRole, str] = {
    ModelRole.CHAT: "Chat",
    ModelRole.COMPACT: "Compaction",
    ModelRole.SUBAGENT: "Subagent",
    ModelRole.GUARD: "Guard",
    ModelRole.VISION: "Vision",
    ModelRole.REASONING: "Reasoning",
}

DB_PATH = Path.home() / ".local" / "share" / "axio-tui" / "axio.db"


class _AxioLogFilter(logging.Filter):
    """Only pass log records from axio-related loggers.

    Third-party loggers (aiosqlite, markdown_it, asyncio, …) produce
    thousands of DEBUG messages that flood the RichLog widget and
    starve the event loop when routed through _RichLogHandler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        return name.startswith("axio") or name == "root"


class _RichLogHandler(logging.Handler):
    """Routes log records to a Textual RichLog widget."""

    def __init__(self, widget: RichLog) -> None:
        super().__init__()
        self._widget = widget
        self.addFilter(_AxioLogFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._widget.write(self.format(record))
        except Exception:
            self.handleError(record)


class _ApplicationGuard(PermissionGuard):
    def __init__(self, app: AgentApp) -> None:
        self.app = app

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        if tool.name in self.app._disabled_plugins:
            raise GuardError("Tool is disabled")
        for name, names in self.app._guard_tool_map.items():
            if name in self.app._disabled_guards or name in {"path", "llm"}:
                continue
            guard = self.app._guard_instances.get(name)
            if tool.name in names and guard is not None:
                kwargs = await guard(tool, **kwargs)
        return await self.app._permissions(tool, **kwargs)


class _ApplicationSelector:
    def __init__(self, app: AgentApp, excluded: frozenset[str] = frozenset()) -> None:
        self.app = app
        self.excluded = excluded

    async def select(self, messages: Iterable[Message], tools: Iterable[Tool[Any]]) -> Iterable[Tool[Any]]:
        available = [tool for tool in tools if tool.name not in self.app._disabled_plugins | self.excluded]
        selector = self.app._get_active_selector()
        return await selector.select(messages, available) if selector is not None else available


class AgentApp(App[None]):
    """Axio Agent TUI."""

    TITLE = "Axio Agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("escape", "stop_agent", "Stop", show=False),
        Binding("f12", "toggle_dev_console", "DevConsole", show=False),
        Binding("alt+up", "nav_up", "Prev Msg"),
        Binding("alt+down", "nav_down", "Next Msg"),
    ]
    CSS = """
    #log {
        height: 1fr;
    }
    #log > .nav-selected {
        background: $accent 10%;
        border-left: thick $accent;
    }
    #log > Markdown {
        margin: 0;
        padding: 0 1;
    }
    #log > .meta {
        height: auto;
        margin: 0;
        padding: 0 1;
    }
    #log > .user-msg {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
        background: $surface;
        border-left: thick $accent;
    }
    #dev-console {
        height: 1fr;
        border-top: solid $accent;
        display: none;
    }
    #dev-console.visible {
        display: block;
    }
    #tui-status {
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
        display: none;
    }
    #tui-status.visible {
        display: block;
    }
    #status {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, *, paths: ConfigurationPaths | None = None, options: StartupOptions = StartupOptions()) -> None:
        super().__init__()
        self.paths = paths or ConfigPaths.discover(Path.cwd())
        self.options = options
        self._providers: dict[str, ToolProvider] = {}
        self._provider_catalog = discover_tool_providers()
        self._base_tools: list[Tool[Any]] = []
        self._projects: tuple[MCPConfiguration, ...] = ()
        self._local_binding: LocalBinding | None = None
        self._tool_guard = _ApplicationGuard(self)
        self._tool_selector = _ApplicationSelector(self)
        self._workspace_loading = False
        self._workspace_task: Any = None
        self._init_task: Any = None
        self._session: aiohttp.ClientSession | None = None
        self._agent: Agent | None = None
        self._transports = TransportRegistry()
        self._role_bindings: dict[ModelRole, RoleBinding] = {}
        self._context_backend: ContextBackend | None = None
        self._config_conn: aiosqlite.Connection | None = None
        self._chat_context_: ContextStore | None = None
        self._config_: ProjectConfig | None = None
        self._global_config_: ProjectConfig | None = None
        self._is_running = False
        self.ready: asyncio.Event = asyncio.Event()
        self._worker: Any = None
        self._pending_input: asyncio.Queue[tuple[str, Static]] = asyncio.Queue()
        self._guard_lock = asyncio.Lock()
        self._permission_store: ConfigRuleStore | None = None
        self._permissions = PolicyGuard(
            self.paths.project_root, self._permission_prompt, advisor=self._permission_advice
        )
        self._current_md: Markdown | None = None
        self._response_text = ""
        self._declined = False
        self._thinking: _ThinkingWidget | None = None
        self._text_dirty = False
        self._last_input_tokens = 0
        self._chat_model: ModelSpec | None = None
        self._model_name = ""
        self._all_tools: list[Tool[Any]] = []
        self._disabled_plugins: set[str] = set()
        self._disabled_transports: set[str] = set()
        self._disabled_guards: set[str] = set()
        self._guard_tool_map: dict[str, set[str]] = {}
        self._guard_instances: dict[str, PermissionGuard] = {}
        self._guard_classes: dict[str, type[PermissionGuard]] = {}
        self._active_tools: list[Tool[Any]] = []
        self._chat_transport: Any = None
        self._selector_classes: dict[str, type] = {}
        self._selector_instances: dict[str, Any] = {}
        self._selector_labels: dict[str, str] = {}
        self._active_selector: str | None = None
        self._log_handler: _RichLogHandler | None = None
        self._tool_status: _ToolStatusWidget | None = None
        self._tool_arg_streams: dict[str, ToolArgStream] = {}  # keyed by tool_use_id
        self._agent_emoji = ""
        self._tools_plugins: dict[str, ToolsPlugin] = {}
        self._nav_index: int | None = None

    @property
    def _chat_context(self) -> ContextStore:
        assert self._chat_context_ is not None
        return self._chat_context_

    @_chat_context.setter
    def _chat_context(self, value: ContextStore) -> None:
        self._chat_context_ = value
        self._permissions.set_session(value.session_id)

    @property
    def _config(self) -> ProjectConfig:
        assert self._config_ is not None
        return self._config_

    @_config.setter
    def _config(self, value: ProjectConfig) -> None:
        self._config_ = value

    @property
    def _global_config(self) -> ProjectConfig:
        assert self._global_config_ is not None
        return self._global_config_

    @_global_config.setter
    def _global_config(self, value: ProjectConfig) -> None:
        self._global_config_ = value

    def compose(self) -> ComposeResult:
        yield Header()
        log = VerticalScroll(id="log")
        log.can_focus = False
        yield log
        yield RichLog(highlight=True, markup=True, id="dev-console")
        yield Static("", id="tui-status")
        yield Static("", id="status")
        yield Input(placeholder="Enter a coding task...", id="input")
        yield Footer()

    async def on_mount(self) -> None:
        console = self.query_one("#dev-console", RichLog)
        handler = _RichLogHandler(console)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s[%(name)s]: %(message)s"))
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

        _tools.status_line_callback = self._on_agent_status
        _tools.subagent_factory = self._make_subagent

        self._session = aiohttp.ClientSession()

        for name, backend in discover_context_backends().entries.items():
            try:
                await backend.init(DB_PATH)
            except BaseException as error:
                await backend.close()
                if not isinstance(error, Exception):
                    raise
                logger.warning("Context backend %s failed: %s", name, error)
                continue
            self._context_backend = backend
            break
        self._config_conn = await connect_config(DB_PATH)
        self._chat_context = self._create_context()
        if self._context_backend is None:
            await self._write_meta("[yellow]History is temporary. Install axio-tui[history] to save conversations.[/]")
        self._config = ProjectConfig(self._config_conn)
        self._global_config = ProjectConfig(self._config_conn, project=GLOBAL_PROJECT)
        self._permission_store = ConfigRuleStore(
            ProjectConfig(self._config_conn, project=str(self.paths.project_root))
        )
        self._permissions.store = self._permission_store

        # Show UI immediately; heavy init runs in background
        inp = self.query_one("#input", Input)
        inp.placeholder = "Initializing..."
        inp.focus()
        self.set_interval(0.1, self._flush_text)
        self._init_task = self.run_worker(self._init_worker(), group="initialization")

    async def _init_worker(self) -> None:
        try:
            await self._write_meta("[dim]Loading transports...[/]")

            # Discover selector entry points and load active selector from config
            self._selector_classes = discover_selectors()
            self._selector_labels = {n: getattr(cls, "label", n) for n, cls in self._selector_classes.items()}
            raw_sel = await self._config.get("selector.active")
            self._active_selector = raw_sel or None

            # Discover MCP/tool plugins synchronously (no I/O), then init concurrently with transports
            self._tools_plugins = discover_tools_plugins()
            if self.options.isolated:
                self._tools_plugins.pop("mcp", None)

            async def _init_plugins() -> None:
                for plugin in self._tools_plugins.values():
                    try:
                        set_status = getattr(plugin, "set_status_callback", None)
                        if callable(set_status):
                            set_status(self._mcp_status)
                        await plugin.init(config=self._config, global_config=self._global_config)
                    except Exception:
                        logger.warning("Tools plugin init failed", exc_info=True)

            assert self._session is not None
            await asyncio.gather(
                self._transports.init(self._session, config=self._config, global_config=self._global_config),
                _init_plugins(),
            )

            for name, reason in self._transports.errors.items():
                await self._write_meta(f"[yellow]{escape(name)} unavailable: {escape(reason)}[/]")

            await self._load_workspace()

            requested_chat: str | None = None
            # Resolve model for each role from persisted config
            for role in ModelRole:
                config_value = await self._config.get(f"model.{role}")
                if config_value:
                    if role is ModelRole.CHAT and ":" in config_value:
                        requested_chat = config_value.split(":", 1)[0]
                    binding = self._transports.resolve(config_value)
                    if binding is not None:
                        self._role_bindings[role] = binding
                    elif role is ModelRole.CHAT:
                        detail = f"Saved model {config_value!r} is unavailable. Select a model explicitly."
                        if requested_chat in self._transports.available:
                            source = self._transports.get_transport(requested_chat)
                            diagnostic = getattr(source, "model_selection_error", None)
                            if callable(diagnostic):
                                detail = diagnostic(config_value.split(":", 1)[1])
                        await self._write_meta("[yellow]" + escape(detail) + "[/]")

            # Choose defaults from installed providers without replacing saved roles.
            for role, capability in ((ModelRole.CHAT, Capability.text), (ModelRole.VISION, Capability.vision)):
                if role not in self._role_bindings:
                    default = self._transports.default_model(str(role), capability)
                    if default is not None and (role is not ModelRole.CHAT or requested_chat is None):
                        self._role_bindings[role] = default

            # Discover guards via entry points
            self._guard_classes = discover_guards()
            # The built-in policy replaces legacy path prompts. Keep other guard plugins.
            self._guard_classes.pop("path", None)

            # Load disabled plugins from config
            raw = await self._config.get("plugins.disabled")
            if raw:
                self._disabled_plugins = {n.strip() for n in raw.split(",") if n.strip()}

            # Load disabled transports from config
            raw = await self._config.get("transports.disabled")
            if raw:
                self._disabled_transports = {n.strip() for n in raw.split(",") if n.strip()}

            # Load disabled guards from config
            raw = await self._config.get("guards.disabled")
            if raw:
                self._disabled_guards = {n.strip() for n in raw.split(",") if n.strip()}

            # Load optional guard/reviewer assignments. Policy covers every tool.
            for guard_name in self._guard_classes:
                raw = await self._config.get(f"guards.{guard_name}.tools")
                if raw is not None:
                    self._guard_tool_map[guard_name] = {n.strip() for n in raw.split(",") if n.strip()}
                else:
                    self._guard_tool_map[guard_name] = set()

            # Instantiate enabled guards
            for guard_name, cls in self._guard_classes.items():
                if guard_name in self._disabled_guards:
                    continue
                instance = self._instantiate_guard(guard_name, cls)
                if instance is not None:
                    self._guard_instances[guard_name] = instance

            self._on_tools_plugin_dismissed()

            if not self._transports.available:
                await self._write_meta("[yellow]Configure a transport from the command palette to start chatting.[/]")
                return

            if requested_chat is not None and requested_chat not in self._transports.available:
                await self._write_meta(
                    f"[yellow]Saved transport {escape(requested_chat)} is unavailable. "
                    "Configure it or explicitly choose another model.[/]"
                )
                return

            if requested_chat is not None and ModelRole.CHAT not in self._role_bindings:
                return

            # Set up chat transport
            if ModelRole.CHAT in self._role_bindings:
                chat_binding = self._role_bindings[ModelRole.CHAT]
                chat_transport = self._transports.make_transport(chat_binding.transport, chat_binding.model)
                self._chat_model = chat_binding.model
            else:
                # Fallback: first available transport with its default model
                first = self._transports.available[0]
                t = self._transports.get_transport(first)
                chat_transport = self._transports.make_transport(first, t.model)
                self._chat_model = t.model
                self._role_bindings[ModelRole.CHAT] = RoleBinding(first, t.model)

            self._model_name = self._chat_model.id
            self.sub_title = self._model_name

            if ModelRole.VISION in self._role_bindings:
                vb = self._role_bindings[ModelRole.VISION]
                _tools.vision_transport = self._transports.make_transport(vb.transport, vb.model)

            # Connect optional provider tools through the transport contract
            self._transports.bind_tools()

            self._chat_transport = chat_transport

            self._agent = Agent(
                system=SYSTEM_PROMPT,
                tools=self._base_tools,
                providers=self._agent_providers(),
                tool_guards=(self._tool_guard,),
                selector=self._tool_selector,
                transport=chat_transport,
                last_iteration_message=LAST_ITERATION_MESSAGE,
            )

            self._bind_workspace()
            await self._update_status()
            await self._show_welcome()
        except Exception:
            logger.error("Initialization failed", exc_info=True)
            await self._write_meta("[red]Initialization failed - see logs[/]")
        finally:
            self.ready.set()
            inp = self.query_one("#input", Input)
            inp.placeholder = "Enter a coding task..."
            inp.focus()

    async def _show_welcome(self) -> None:
        lines: list[str] = [
            "# Axio Agent",
            "",
            "AI coding assistant. Type a task below to get started.",
            "Press `Ctrl+P` to open the command palette.",
            "",
        ]

        # Transports
        lines.append("**Transports**")
        lines.append("")
        for name in self._transports.discovered:
            if name in self._transports.available:
                t = self._transports.get_transport(name)
                lines.append(f"- {t.name} ({len(t.models)} models)")
            else:
                label = name.replace("-", " ").title()
                lines.append(f"- ~~{label}~~ - no API key")

        # Model roles
        lines.append("")
        lines.append("**Models**")
        lines.append("")
        lines.append("| Role | Model |")
        lines.append("|------|-------|")
        for role in ModelRole:
            label = _ROLE_LABELS[role]
            if role in self._role_bindings:
                b = self._role_bindings[role]
                lines.append(f"| {label} | `{b.model.id}` ({b.transport}) |")
            else:
                lines.append(f"| {label} | *not configured* |")

        # Tools
        lines.append("")
        enabled = [t for t in self._all_tools if t.name not in self._disabled_plugins]
        disabled_count = len(self._all_tools) - len(enabled)
        suffix = f" ({disabled_count} disabled)" if disabled_count else ""
        lines.append(f"**Tools ({len(enabled)})**{suffix}")
        lines.append("")
        lines.append(", ".join(f"`{t.name}`" for t in enabled))

        # Guards
        if self._guard_classes:
            lines.append("")
            enabled_guards = [g for g in self._guard_classes if g not in self._disabled_guards]
            disabled_guards = [g for g in self._guard_classes if g in self._disabled_guards]
            parts: list[str] = [f"`{g}`" for g in enabled_guards]
            parts.extend(f"~~`{g}`~~" for g in disabled_guards)
            lines.append(f"**Guards:** {', '.join(parts)}")

        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await scroll.mount(_Markdown("\n".join(lines)))
        scroll.scroll_end(animate=False)

    async def on_unmount(self) -> None:
        if self._init_task is not None and self._init_task.is_running:
            self._init_task.cancel()
            try:
                await self._init_task.wait()
            except Exception:
                pass
        # Cancel the active agent worker first so the SSE connection is
        # released before we close the aiohttp session.  Without this the
        # executor shutdown hangs waiting for orphan threads.
        if self._worker is not None and self._worker.is_running:
            self._worker.cancel()
            try:
                await self._worker.wait()
            except Exception:
                pass

        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        _tools.status_line_callback = None
        _tools.subagent_factory = None
        _tools.vision_transport = None
        self._transports.unbind_tools()
        for plugin in self._tools_plugins.values():
            try:
                await plugin.close()
            except Exception:
                pass
        if self._workspace_task is not None and self._workspace_task.is_running:
            self._workspace_task.cancel()
            try:
                await self._workspace_task.wait()
            except Exception:
                pass
        for provider in reversed(list(self._providers.values())):
            try:
                await provider.close()
            except Exception:
                logger.warning("Provider close failed", exc_info=True)
        await self._transports.close()
        if self._context_backend is not None:
            await self._context_backend.close()
        if self._config_conn is not None:
            await self._config_conn.close()
        if self._session is not None:
            await self._session.close()
            # Give aiohttp time to close underlying SSL transports
            await asyncio.sleep(0.25)

    async def _make_subagent(self) -> tuple[Agent, ContextStore]:
        context = await MemoryContextStore.from_context(self._chat_context)
        assert self._agent is not None
        # Tools are immutable. Reuse them to retain explicit MCP schemas, validation and concurrency limits.
        sub_tools = [tool for tool in self._agent.tools if tool.name != "subagent"]
        transport = self._chat_transport if self._chat_transport is not None else self._agent.transport
        sub_agent = self._agent.copy(
            system=SUBAGENT_SYSTEM_PROMPT,
            tools=sub_tools,
            max_iterations=25,
            selector=_ApplicationSelector(self, frozenset({"subagent"})),
            transport=transport,
        )
        if ModelRole.SUBAGENT in self._role_bindings:
            sb = self._role_bindings[ModelRole.SUBAGENT]
            sub_transport = self._transports.make_transport(sb.transport, sb.model)
            sub_agent = sub_agent.copy(transport=sub_transport)
        for provider in self._providers.values():
            bind = getattr(provider, "bind", None)
            if callable(bind):
                bind(sub_agent)
        return sub_agent, context

    def _on_agent_status(self, message: str) -> None:
        widget = self.query_one("#tui-status", Static)
        widget.update(message)
        widget.add_class("visible")

    def _hide_agent_status(self) -> None:
        self.query_one("#tui-status", Static).remove_class("visible")

    async def _update_status(self) -> None:
        total_in, total_out = await self._chat_context.get_context_tokens()
        parts: list[str] = []
        if self._agent_emoji:
            parts.append(self._agent_emoji)
        parts.append(f"Model: {self._model_name}")
        if self._chat_model is not None and self._last_input_tokens > 0:
            ctx_win = self._chat_model.context_window
            pct = self._last_input_tokens * 100 // ctx_win
            parts.append(f"Context: {self._last_input_tokens:,} / {ctx_win:,} ({pct}%)")
        parts.append(f"Tokens: {total_in:,} in / {total_out:,} out")
        self.query_one("#status", Static).update("  |  ".join(parts))

    async def _flush_text(self) -> None:
        if not self._text_dirty or self._current_md is None:
            return
        self._text_dirty = False
        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await self._current_md.update(self._response_text)
        scroll.scroll_end(animate=False)

    async def _ensure_thinking(self) -> _ThinkingWidget:
        """The widget this turn's reasoning goes to, mounted the first time there is any.

        Reasoning arrives before the answer, so the answer being written is closed first.
        """
        if self._thinking is None:
            await self._flush_text()
            self._current_md = None
            self._response_text = ""
            await self._mount_thinking()
        assert self._thinking is not None
        return self._thinking

    async def _mount_thinking(self, *, before_answer: bool = False) -> _ThinkingWidget:
        """A thinking widget in the log, in front of the answer where the answer came first.

        Reasoning happens before the answer. The marker for reasoning a provider withheld is
        mounted once the answer is rendered, so it goes in front of that widget, not below it.
        """
        self._thinking = _ThinkingWidget()
        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return self._thinking
        answer = self._current_md
        if before_answer and answer is not None and answer.is_attached:
            await scroll.mount(self._thinking, before=answer)
        else:
            await scroll.mount(self._thinking)
            scroll.scroll_end(animate=False)
        return self._thinking

    async def _close_thinking(self, tokens: int = 0) -> None:
        """Fold the reasoning pane and let go of it, which is the only way to let go of it.

        Cleared without folding, an expanded pane was stranded in the transcript.
        """
        if self._thinking is None:
            return
        self._thinking.collapse(tokens)
        if self._thinking.empty:
            # An iteration that neither reasoned nor was billed for it. A bare marker there says
            # nothing and reads as a second turn.
            await self._thinking.remove()
        self._thinking = None

    async def _ensure_md(self) -> Markdown:
        if self._current_md is None:
            scroll = self.query_one("#log", VerticalScroll)
            if not scroll.is_attached:
                self._current_md = _Markdown("")
                return self._current_md
            self._current_md = _Markdown("")
            await scroll.mount(self._current_md)
        return self._current_md

    async def _write_meta(self, content: str) -> None:
        """Flush pending text, then mount a styled Static line."""
        await self._flush_text()
        self._current_md = None
        self._response_text = ""
        self._declined = False
        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await scroll.mount(Static(content, classes="meta"))
        scroll.scroll_end(animate=False)

    async def _write_user(self, content: str) -> None:
        """Flush pending text, then mount a user message with distinct styling."""
        await self._flush_text()
        self._current_md = None
        self._response_text = ""
        self._declined = False
        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await scroll.mount(Static(content, classes="user-msg"))
        scroll.scroll_end(animate=False)

    async def _ensure_tool_status(self) -> _ToolStatusWidget:
        if self._tool_status is None:
            await self._flush_text()
            self._current_md = None
            self._response_text = ""
            self._declined = False
            self._tool_status = _ToolStatusWidget()
            scroll = self.query_one("#log", VerticalScroll)
            if scroll.is_attached:
                await scroll.mount(self._tool_status)
                scroll.scroll_end(animate=False)
        return self._tool_status

    async def _handle_tool_field_event(
        self, event: ToolFieldStart | ToolFieldDelta | ToolFieldEnd, tool_use_id: str
    ) -> None:
        """Handle streaming tool field events to show live args in UI."""
        w = await self._ensure_tool_status()
        w.update_arg(tool_use_id, event)

    async def _path_guard_prompt_fn(self, msg: str) -> str:
        async with self._guard_lock:
            try:
                from axio_tui_guards.dialogs import PathGuardDialog

                result: str = await self.push_screen_wait(PathGuardDialog(msg))
                return result
            except ImportError:
                raise GuardError("Path permission dialog is unavailable")

    async def _llm_guard_prompt_fn(self, msg: str) -> str:
        async with self._guard_lock:
            try:
                from axio_tui_guards.dialogs import LLMGuardDialog

                result = await self.push_screen_wait(LLMGuardDialog(msg))
                return str(result)
            except ImportError:
                raise GuardError("Review dialog is unavailable")

    async def _permission_prompt(self, request: PermissionRequest) -> Approval:
        return await prompt_permission(self, request)

    async def _permission_advice(self, tool: Tool[Any], arguments: Mapping[str, Any]) -> str:
        if "llm" in self._disabled_guards or tool.name not in self._guard_tool_map.get("llm", set()):
            return ""
        reviewer = self._guard_instances.get("llm")
        review = getattr(reviewer, "review", None)
        if callable(review):
            return str(await review(tool, **arguments))
        return "Review unavailable; explicit permission is required."

    def _show_permissions(self) -> None:
        if self._permission_store is not None:
            self.push_screen(PermissionsScreen(self._permissions, self._permission_store))

    def _instantiate_guard(self, name: str, cls: type[PermissionGuard]) -> PermissionGuard | None:
        match name:
            case "path":
                return cls(prompt_fn=self._path_guard_prompt_fn)  # type: ignore[call-arg]
            case "llm":
                if ModelRole.GUARD not in self._role_bindings:
                    return None
                gb = self._role_bindings[ModelRole.GUARD]
                guard_transport = self._transports.make_transport(gb.transport, gb.model)
                from axio.context import MemoryContextStore as _MemCtx

                guard_agent = Agent(
                    system="You review tool calls for safety. Use the confirm tool to classify.",
                    tools=[Tool(name="confirm", handler=confirm)],
                    transport=guard_transport,
                )
                guard_context = _MemCtx()
                return cls(  # type: ignore[call-arg]
                    agent=guard_agent,
                    context=guard_context,
                    prompt_fn=self._llm_guard_prompt_fn,
                )
            case _:
                try:
                    return cls()
                except TypeError:
                    return None

    def _rebuild_agent_tools(self) -> None:
        self._active_tools = [
            replace(tool, guards=(*tool.guards, self._tool_guard))
            for tool in self._all_tools
            if tool.name not in self._disabled_plugins
        ]
        if self._agent is not None:
            self._agent.tools = list(self._base_tools)
            self._agent.providers = self._agent_providers()
            self._agent.tool_guards = (self._tool_guard,)
            self._agent.selector = self._tool_selector

    def _agent_providers(self) -> list[ToolProvider]:
        return [*self._providers.values(), *(p for p in self._tools_plugins.values() if isinstance(p, ToolProvider))]

    def _show_model_select(self) -> None:
        if not self._transports.available:
            return
        self.push_screen(ModelRoleSelectScreen(self._role_bindings), self._on_role_selected)

    def _on_role_selected(self, role: ModelRole | None) -> None:
        if role is None or not self._transports.available:
            return
        match role:
            case ModelRole.GUARD:
                models = self._transports.all_models()
            case ModelRole.VISION:
                models = self._transports.all_models(Capability.vision)
            case ModelRole.REASONING:
                models = self._transports.all_models(Capability.reasoning)
            case _:
                models = self._transports.all_models(Capability.tool_use)
        models = [(n, m) for n, m in models if n not in self._disabled_transports]
        self.push_screen(
            ModelSelectScreen(models),
            lambda result: self._on_model_selected(role, result),
        )

    async def _on_model_selected(self, role: ModelRole, result: tuple[str, ModelSpec] | None) -> None:
        if result is None or self._is_running:
            return
        transport_name, spec = result
        binding = RoleBinding(transport=transport_name, model=spec)
        self._role_bindings[role] = binding
        await self._config.set(f"model.{role}", self._transports.encode(transport_name, spec.id))

        match role:
            case ModelRole.CHAT:
                if self._agent is None:
                    await self._activate_transport(transport_name)
                self._chat_transport = self._transports.make_transport(transport_name, spec)
                if self._agent is not None:
                    self._agent.transport = self._chat_transport
                self._chat_model = spec
                self._model_name = spec.id
                self.sub_title = spec.id
            case ModelRole.VISION:
                _tools.vision_transport = self._transports.make_transport(transport_name, spec)
            case ModelRole.GUARD:
                # Re-instantiate LLMGuard with the new transport so it uses the chosen model.
                if "llm" in self._guard_classes and "llm" not in self._disabled_guards:
                    instance = self._instantiate_guard("llm", self._guard_classes["llm"])
                    if instance is not None:
                        self._guard_instances["llm"] = instance
                        self._rebuild_agent_tools()
            case ModelRole.COMPACT | ModelRole.SUBAGENT | ModelRole.REASONING:
                pass  # stored in self._role_bindings, used on demand
        await self._update_status()
        self.push_screen(ModelRoleSelectScreen(self._role_bindings), self._on_role_selected)

    def _create_context(self, session_id: str | None = None) -> ContextStore:
        if self._context_backend is not None:
            return self._context_backend.create(session_id)
        return MemoryContextStore()

    async def _new_session(self) -> None:
        if self._is_running:
            self.notify("Stop the active run before starting a new chat.")
            return
        self._chat_context = self._create_context()
        self._last_input_tokens = 0
        await self.action_clear_log()
        await self._write_meta("[dim]--- New session ---[/]")

    async def _fork_conversation(self) -> None:
        if self._is_running:
            return
        self._chat_context = await self._chat_context.fork()
        await self.action_clear_log()
        await self._render_history()
        await self._write_meta("[dim]--- Forked conversation ---[/]")

    async def _show_session_select(self) -> None:
        if self._context_backend is None:
            await self._write_meta("[yellow]History is temporary. Previous sessions are not saved.[/]")
            return
        sessions = await self._chat_context.list_sessions()
        if not sessions:
            await self._write_meta("[dim]No previous sessions found[/]")
            return
        self.push_screen(SessionSelectScreen(sessions), self._on_session_selected)

    async def _on_session_selected(self, info: SessionInfo | None) -> None:
        if info is None or self._is_running:
            return
        if self._context_backend is None:
            return
        self._chat_context = self._create_context(info.session_id)
        self._last_input_tokens = 0
        await self._update_status()
        await self.action_clear_log()
        await self._write_meta(f"[dim]--- Restored session ({info.message_count} msgs): {info.preview} ---[/]")
        await self._render_history()

    async def _render_history(self) -> None:
        """Replay stored messages into the chat log."""
        self._nav_index = None
        from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock

        history = await self._chat_context.get_history()
        scroll = self.query_one("#log", VerticalScroll)
        # Build result map: tool_use_id -> ToolResultBlock
        result_map: dict[str, ToolResultBlock] = {}
        for msg in history:
            if msg.role == "user":
                for b in msg.content:
                    if isinstance(b, ToolResultBlock):
                        result_map[b.tool_use_id] = b

        for msg in history:
            if msg.role == "user":
                texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                if texts:
                    await scroll.mount(Static(f"[bold]You:[/] {texts[0]}", classes="user-msg"))
                # Tool results rendered as part of assistant tool summary - skip here
            else:
                texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                if texts:
                    md = _Markdown("\n\n".join(texts))
                    await scroll.mount(md)
                tool_blocks = [b for b in msg.content if isinstance(b, ToolUseBlock)]
                if tool_blocks:
                    w = _ToolStatusWidget()
                    for b in tool_blocks:
                        result_block = result_map.get(b.id)
                        content_str = ""
                        is_error = False
                        if result_block is not None:
                            is_error = result_block.is_error
                            content_str = (
                                result_block.content
                                if isinstance(result_block.content, str)
                                else "\n".join(tb.text for tb in result_block.content if isinstance(tb, TextBlock))
                            )
                        w.track(b.id, b.name)
                        w.complete(b.id, is_error=is_error, content=content_str, tool_input=b.input)
                    await scroll.mount(w)
        scroll.scroll_end(animate=False)

    async def _replace_context(self, messages: list[Message]) -> None:
        """Create a fresh context and retain the conversation token counts."""
        total_in, total_out = await self._chat_context.get_context_tokens()
        # Compaction changes storage, but does not end the permission lifetime of this chat.
        self._chat_context_ = self._create_context()
        for msg in messages:
            await self._chat_context.append(msg)
        if total_in or total_out:
            await self._chat_context.set_context_tokens(total_in, total_out)

    def _should_compact(self) -> bool:
        if self._chat_model is None or self._last_input_tokens == 0:
            return False
        return self._last_input_tokens / self._chat_model.context_window > 0.7

    async def _compact_with_progress(self) -> list[Message] | None:
        """Run compaction while showing a live elapsed-time counter in the chat."""
        if self._agent is None:
            return None
        elapsed = 0
        widget = Static("[dim]Compacting in progress... 0s[/]", classes="meta")
        scroll = self.query_one("#log", VerticalScroll)
        await scroll.mount(widget)
        scroll.scroll_end(animate=False)

        def _tick() -> None:
            nonlocal elapsed
            elapsed += 1
            widget.update(f"[dim]Compacting in progress... {elapsed}s[/]")

        timer = self.set_interval(1, _tick)
        try:
            if ModelRole.COMPACT in self._role_bindings:
                cb = self._role_bindings[ModelRole.COMPACT]
                transport = self._transports.make_transport(cb.transport, cb.model)
            else:
                transport = self._agent.transport
            return await compact_context(self._chat_context, transport)
        finally:
            timer.stop()
            await widget.remove()

    async def _compact_conversation(self) -> None:
        if self._agent is None or self._is_running:
            return
        self.run_worker(self._do_compact(), exclusive=True)

    async def _do_compact(self) -> None:
        messages = await self._compact_with_progress()
        if messages is not None:
            await self._replace_context(messages)
            await self._write_meta("[dim]--- Context compacted ---[/]")
        else:
            await self._write_meta("[dim]--- Nothing to compact ---[/]")

    def action_command_palette(self) -> None:
        self._nav_select(None)
        super().action_command_palette()

    def get_system_commands(self, screen: object) -> Generator[SystemCommand, None, None]:
        yield from super().get_system_commands(screen)  # type: ignore[arg-type]
        yield SystemCommand("Change Model", "Switch the active LLM model", self._show_model_select)
        yield SystemCommand("Clear Log", "Clear the chat log", self.action_clear_log)
        yield SystemCommand("New Session", "Reset conversation context", self._new_session)
        yield SystemCommand("Fork Conversation", "Branch from current conversation", self._fork_conversation)
        yield SystemCommand("Restore Session", "Load a previous conversation", self._show_session_select)
        yield SystemCommand(
            "Compact Conversation", "Summarize old messages to save context", self._compact_conversation
        )
        yield SystemCommand(
            "Toggle Dev Console", "Show or hide the developer log console", self.action_toggle_dev_console
        )
        yield SystemCommand("Workspace Configuration", "Show instruction and MCP sources", self._show_workspace)
        yield SystemCommand("Skills", "Show local and MCP skills", self._show_skills)
        yield SystemCommand("MCP Servers", "Inspect connections, tools, resources and skills", self._show_mcp)
        yield SystemCommand("Reconnect MCP", "Reconnect automatic and saved MCP servers", self._reconnect_mcp)
        yield SystemCommand("Reload Workspace", "Reload instructions, skills and MCP", self._reload_workspace)
        yield SystemCommand("Manage Plugins", "Configure tools and guards", self._show_plugin_hub)
        yield SystemCommand("Permissions", "Inspect and revoke chat or project rules", self._show_permissions)
        for pname, plugin in self._tools_plugins.items():
            yield SystemCommand(
                f"Manage {plugin.label}",
                f"Configure {plugin.label}",
                lambda p=plugin: self._show_tools_plugin(p),
            )
        screens = self._transports.settings_screens()
        for name in self._transports.discovered:
            if name not in screens:
                continue
            label = (
                self._transports.display_name(name)
                if name in self._transports.available
                else name.replace("-", " ").title()
            )
            screen_cls = screens[name]
            yield SystemCommand(
                f"Configure {label}",
                f"Edit {label} transport settings",
                lambda n=name, s=screen_cls: self._show_transport_settings(n, s),
            )

    async def _load_workspace(self) -> None:
        projects: tuple[MCPConfiguration, ...] = ()
        config: dict[str, Any] = {"application": "tui"}
        if importlib.util.find_spec("axio_tools_local") is not None:
            from axio_tools_local.provider import LocalOptions

            config["local"] = LocalOptions(
                self.paths.cwd,
                self.paths.roots,
                self.options.isolated,
                self.options.instructions_files,
                self.options.skill_directories,
                self.options.instructions_max_bytes,
            )
        elif self.options.instructions_files or self.options.skill_directories:
            await self._write_meta("[yellow]Local instructions and skills require axio-tui[local].[/]")
        if importlib.util.find_spec("axio_tools_mcp") is not None:
            from .configuration import connection_configs, load_mcp_config

            projects = await asyncio.to_thread(
                load_mcp_config, self.paths, files=self.options.mcp_files, isolated=self.options.isolated
            )
            config["mcp"] = connection_configs(projects)
        elif self.options.mcp_files:
            await self._write_meta("[yellow]MCP configuration requires axio-tui[mcp].[/]")
        for name, error in self._provider_catalog.errors.items():
            await self._write_meta(escape(f"Provider {name}: {error}"))
        for name, provider in self._provider_catalog.entries.items():
            try:
                if name in self._providers:
                    if isinstance(provider, RefreshableToolProvider):
                        await provider.refresh(config=config)
                else:
                    set_status = getattr(provider, "set_status_callback", None)
                    if callable(set_status):
                        set_status(self._mcp_status)
                    try:
                        await provider.init(config=config, global_config=self._global_config)
                    except BaseException:
                        await provider.close()
                        raise
                    self._providers[name] = provider
                    self._provider_catalog.errors.pop(name, None)
            except Exception as error:
                if name not in self._providers:
                    self._provider_catalog.errors[name] = str(error)
                await self._write_meta(escape(f"Provider {name} failed: {error}"))
        self._projects = projects
        self._on_tools_plugin_dismissed()

    def _mcp_status(self, message: str) -> None:
        self.call_later(self._write_meta, escape(message))

    def _plugin_instructions(self) -> str:
        parts = []
        for plugin in self._tools_plugins.values():
            if isinstance(plugin, ToolProvider):
                continue
            describe = getattr(plugin, "instructions", None)
            if callable(describe):
                parts.append(str(describe()))
        return "\n\n".join(["", *parts])

    def _bind_workspace(self) -> None:
        if self._agent is None:
            return
        self._agent.system = SYSTEM_PROMPT + self._plugin_instructions()
        for provider in self._providers.values():
            bind = getattr(provider, "bind", None)
            if callable(bind):
                binding = bind(self._agent)
                if getattr(binding, "instruction", None) is not None:
                    self._local_binding = binding

    async def _reload_workspace(self) -> None:
        if self._is_running or self._workspace_loading or not self.ready.is_set():
            await self._write_meta("[yellow]Stop the current operation before reloading the workspace.[/]")
            return
        self._workspace_loading = True
        self._workspace_task = self.run_worker(self._reload_workspace_work(), group="workspace")

    async def _reconnect_mcp(self) -> None:
        if self._is_running or self._workspace_loading or not self.ready.is_set():
            self.notify("Stop the current operation before reconnecting MCP.", severity="warning")
            return
        self._workspace_loading = True
        self._workspace_task = self.run_worker(self._reload_workspace_work(reconnect_mcp=True), group="workspace")

    async def _reload_workspace_work(self, *, reconnect_mcp: bool = False) -> None:
        self._workspace_loading = True
        try:
            if reconnect_mcp:
                await self._write_meta("[dim]Reconnecting automatic and saved MCP servers...[/]")
                for plugin in self._tools_plugins.values():
                    reconnect = getattr(plugin, "reconnect", None)
                    if callable(reconnect):
                        await reconnect()
            await self._load_workspace()
            if reconnect_mcp:
                servers = self._inspect_mcp()
                connected = sum(server.state == "connected" for server in servers)
                failed = sum(server.state == "error" for server in servers)
                await self._write_meta(
                    f"MCP reconnect finished: {connected} connected, {failed} failed. Open /mcp for details."
                )
            else:
                await self._write_meta("[dim]Workspace instructions, skills and MCP reloaded.[/]")
        except Exception as error:
            await self._write_meta(f"[red]Workspace reload failed: {escape(str(error))}[/]")
        finally:
            self._on_tools_plugin_dismissed()
            self._workspace_loading = False

    async def _show_workspace(self) -> None:
        text = self.paths.describe()
        if self._local_binding is not None:
            snapshot = await self._local_binding.instruction.load()
            text += "\n\n" + snapshot.text
        for project in self._projects:
            text += "\nMCP project: " + project.origin
            for source in project.files:
                text += f"\n  {source.layer}: {source.source}"
        await self._write_meta(escape(text))

    async def _show_skills(self) -> None:
        entries: list[SkillEntry] = []
        notes: list[str] = []
        for name, provider in self._providers.items():
            if importlib.util.find_spec("axio_tools_local") is not None:
                from axio_tools_local.provider import LocalToolProvider

                if isinstance(provider, LocalToolProvider):
                    found, diagnostics = await asyncio.to_thread(provider_entries, skills=provider.skills)
                    entries.extend(found)
                    notes.extend(diagnostics)
                    continue
            if importlib.util.find_spec("axio_tools_mcp") is not None:
                from axio_tools_mcp.provider import MCPToolProvider

                if isinstance(provider, MCPToolProvider):
                    found, diagnostics = await asyncio.to_thread(provider_entries, mcp=provider.runtime)
                    entries.extend(found)
                    notes.extend(diagnostics)
                    continue
            for instruction in provider.get_instructions():
                entries.append(instruction_entry(name, instruction))
        for plugin in self._tools_plugins.values():
            catalogs = getattr(plugin, "skill_catalogs", None)
            if callable(catalogs):
                for name, catalog in catalogs().items():
                    entries.extend(catalog_entries(f"MCP (manual): {name}", catalog))
            else:
                describe = getattr(plugin, "describe_skills", None)
                if callable(describe):
                    notes.append(str(describe()))
        self.push_screen(SkillsScreen(entries, notes))

    def _inspect_mcp(self) -> list[ServerInspection]:
        servers: list[ServerInspection] = []
        for provider in self._providers.values():
            inspect_servers = getattr(provider, "inspect_servers", None)
            if callable(inspect_servers):
                from .configuration import inspect_configurations

                servers.extend(inspect_configurations(self._projects, inspect_servers()))
        for plugin in self._tools_plugins.values():
            inspect_servers = getattr(plugin, "inspect_servers", None)
            if callable(inspect_servers):
                servers.extend(inspect_servers())
        return servers

    async def _show_mcp(self) -> None:
        available = importlib.util.find_spec("axio_tools_mcp") is not None
        self.push_screen(
            MCPScreen(
                self._inspect_mcp,
                reconnect=self._reconnect_mcp,
                manage=self._manage_mcp if available else None,
            )
        )

    async def _manage_mcp(self) -> None:
        if self._is_running or self._workspace_loading or not self.ready.is_set():
            self.notify("Stop the current operation before editing MCP servers.", severity="warning")
            return
        from .mcp_manage import MCPManagementScreen, destinations

        plugin = self._tools_plugins.get("mcp")

        async def legacy() -> None:
            if plugin is not None:
                self._show_tools_plugin(plugin)

        self.push_screen(
            MCPManagementScreen(
                destinations(self.paths, self.options.mcp_files),
                isolated=self.options.isolated,
                legacy=legacy if plugin is not None else None,
            ),
            self._mcp_management_closed,
        )

    async def _mcp_management_closed(self, changed: bool | None) -> None:
        if changed:
            self._workspace_loading = True
            self._workspace_task = self.run_worker(self._apply_mcp_management(), group="workspace")
        elif isinstance(self.screen, MCPScreen):
            self.screen.refresh_servers()

    async def _apply_mcp_management(self) -> None:
        await self._reload_workspace_work(reconnect_mcp=True)
        if isinstance(self.screen, MCPScreen):
            self.screen.refresh_servers()

    async def _show_transport_settings(self, name: str, screen_cls: type) -> None:
        if self._is_running or not self.ready.is_set():
            await self._write_meta("[yellow]Stop the active run before changing accounts or transport settings.[/]")
            return
        settings = await self._transports.get_settings(name)
        if name in self._transports.errors:
            settings["_error"] = self._transports.errors[name]
        self.push_screen(
            screen_cls(settings),
            lambda result: self._on_transport_configured(name, result),
        )

    async def _on_transport_configured(self, name: str, result: dict[str, str] | None) -> None:
        if result is None:
            return
        if self._is_running:
            await self._write_meta("[yellow]Stop the active run before applying transport settings.[/]")
            return
        was_available = name in self._transports.available
        await self._transports.save_settings(name, result)

        if name not in self._transports.available:
            label = name.replace("-", " ").title()
            reason = self._transports.errors.get(name, "Check the transport credentials.")
            await self._write_meta(f"[yellow]{escape(label)}: {escape(reason)}[/]")
            if (
                self._role_bindings.get(ModelRole.CHAT, None) is not None
                and self._role_bindings[ModelRole.CHAT].transport == name
            ):
                self._agent = None
                self._chat_transport = None
            return

        source = self._transports.get_transport(name)
        for role, binding in list(self._role_bindings.items()):
            if binding.transport == name:
                model = source.models.get(binding.model.id) or source.model
                self._role_bindings[role] = RoleBinding(name, model)
                await self._config.set(f"model.{role}", self._transports.encode(name, model.id))
        if not was_available or self._agent is None:
            await self._activate_transport(name)
        else:
            await self._reconfigure_transport(name)
        await self._update_status()

    async def _activate_transport(self, name: str) -> None:
        """Set up agent for a newly activated transport."""
        if ModelRole.CHAT not in self._role_bindings:
            transport = self._transports.get_transport(name)
            self._role_bindings[ModelRole.CHAT] = RoleBinding(name, transport.model)
        # Set up chat if this transport claimed the chat role
        if ModelRole.CHAT in self._role_bindings and self._role_bindings[ModelRole.CHAT].transport == name:
            cb = self._role_bindings[ModelRole.CHAT]
            self._chat_transport = self._transports.make_transport(name, cb.model)
            self._chat_model = cb.model
            self._model_name = cb.model.id
            self.sub_title = self._model_name
            if self._agent is None:
                self._rebuild_agent_tools()
                self._agent = Agent(
                    system=SYSTEM_PROMPT,
                    tools=self._base_tools,
                    providers=self._agent_providers(),
                    tool_guards=(self._tool_guard,),
                    selector=self._tool_selector,
                    transport=self._chat_transport,
                    last_iteration_message=LAST_ITERATION_MESSAGE,
                )
            else:
                self._agent.transport = self._chat_transport

        if ModelRole.VISION in self._role_bindings and self._role_bindings[ModelRole.VISION].transport == name:
            vb = self._role_bindings[ModelRole.VISION]
            _tools.vision_transport = self._transports.make_transport(name, vb.model)

        self._transports.bind_tools()

        self._bind_workspace()
        label = self._transports.display_name(name)
        await self._write_meta(f"[green]{label} activated[/]")

    async def _reconfigure_transport(self, name: str) -> None:
        """Rebuild cached transport instances after settings change."""
        for role, binding in self._role_bindings.items():
            if binding.transport != name:
                continue
            match role:
                case ModelRole.CHAT:
                    self._chat_model = binding.model
                    self._model_name = binding.model.id
                    self.sub_title = self._model_name
                    self._chat_transport = self._transports.make_transport(name, binding.model)
                    if self._agent is not None:
                        self._agent.transport = self._chat_transport
                case ModelRole.VISION:
                    _tools.vision_transport = self._transports.make_transport(name, binding.model)
                case ModelRole.GUARD:
                    if "llm" in self._guard_classes and "llm" not in self._disabled_guards:
                        instance = self._instantiate_guard("llm", self._guard_classes["llm"])
                        if instance is not None:
                            self._guard_instances["llm"] = instance
                            self._rebuild_agent_tools()
        label = self._transports.display_name(name)
        await self._write_meta(f"[dim]{label} settings updated[/]")

    def _guard_descriptions(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, cls in self._guard_classes.items():
            doc = cls.__doc__
            result[name] = doc.strip().split("\n")[0] if doc else name
        return result

    def _get_active_selector(self) -> ToolSelector | None:
        if self._active_selector is not None:
            return self._selector_instances.get(self._active_selector)
        return None

    def _show_plugin_hub(self) -> None:
        tool_groups = discover_tools_by_package()
        tool_groups.update({name: provider.all_tools for name, provider in self._providers.items()})
        for ep_name, plugin in self._tools_plugins.items():
            tool_groups[ep_name] = list(plugin.all_tools)
        self.push_screen(
            PluginHubScreen(
                tool_groups=tool_groups,
                transport_available=list(self._transports.available),
                transport_discovered=list(self._transports.discovered),
                transport_model_counts=self._transports.model_counts(),
                disabled_plugins=self._disabled_plugins,
                disabled_transports=self._disabled_transports,
                guard_names=self._guard_descriptions(),
                disabled_guards=self._disabled_guards,
                guard_tool_map=self._guard_tool_map,
                on_plugins_changed=self._on_plugins_changed,
                on_transports_changed=self._on_transports_changed,
                on_guards_changed=self._on_guards_changed,
                reload_transport_models=self._reload_transport_models,
                selector_classes=self._selector_classes,
                active_selector=self._active_selector,
                on_selector_changed=self._on_selector_changed,
            ),
        )

    async def _on_selector_changed(self, active: str | None) -> None:
        self._active_selector = active
        await self._config.set("selector.active", active or "")
        if self._agent is not None:
            self._agent.selector = self._tool_selector

    async def _reload_transport_models(self) -> dict[str, int]:
        if self._is_running:
            self.notify("Stop the active run before reloading models.")
            return self._transports.model_counts()
        self.ready.clear()
        try:
            await self._transports.reload_models()
        finally:
            self.ready.set()
        return self._transports.model_counts()

    def _show_tools_plugin(self, plugin: ToolsPlugin) -> None:
        if self._is_running or self._workspace_loading:
            self.notify("Stop the current operation before changing MCP connections.")
            return
        screen = plugin.settings_screen()
        self.push_screen(screen, lambda _result: self._on_tools_plugin_dismissed())

    def _on_tools_plugin_dismissed(self) -> None:
        self._base_tools = standalone_tools(
            discover_tool_catalog(), replace(self._provider_catalog, entries=self._providers)
        )
        for plugin in self._tools_plugins.values():
            if not isinstance(plugin, ToolProvider):
                self._base_tools.extend(plugin.all_tools)
        self._all_tools = collect_tools(self._agent_providers(), tools=self._base_tools)
        self._rebuild_agent_tools()
        self._bind_workspace()

    async def _on_transports_changed(self, disabled: set[str]) -> None:
        self._disabled_transports = disabled
        if disabled:
            await self._config.set("transports.disabled", ",".join(sorted(disabled)))
        else:
            await self._config.delete("transports.disabled")

    async def _on_plugins_changed(self, disabled: set[str]) -> None:
        if self._agent is None:
            return
        self._disabled_plugins = disabled
        if disabled:
            await self._config.set("plugins.disabled", ",".join(sorted(disabled)))
        else:
            await self._config.delete("plugins.disabled")
        self._rebuild_agent_tools()

    async def _on_guards_changed(self, disabled_guards: set[str], guard_tool_map: dict[str, set[str]]) -> None:
        if self._agent is None:
            return
        # Persist disabled guards
        self._disabled_guards = disabled_guards
        if disabled_guards:
            await self._config.set("guards.disabled", ",".join(sorted(disabled_guards)))
        else:
            await self._config.delete("guards.disabled")

        # Persist guard-tool assignments
        self._guard_tool_map = guard_tool_map
        for guard_name, tool_names in guard_tool_map.items():
            if tool_names:
                await self._config.set(f"guards.{guard_name}.tools", ",".join(sorted(tool_names)))
            else:
                await self._config.delete(f"guards.{guard_name}.tools")

        # Re-instantiate guards that were re-enabled
        for guard_name, cls in self._guard_classes.items():
            if guard_name in disabled_guards:
                self._guard_instances.pop(guard_name, None)
            elif guard_name not in self._guard_instances:
                instance = self._instantiate_guard(guard_name, cls)
                if instance is not None:
                    self._guard_instances[guard_name] = instance

        self._rebuild_agent_tools()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        self._nav_select(None)
        value = message.value.strip()
        if not value:
            return
        if not self.ready.is_set():
            message.input.value = ""
            await self._write_meta("[dim]Still initializing, please wait...[/]")
            return
        message.input.value = ""

        if value == "/permissions":
            self._show_permissions()
            return

        if value.split()[0] in {"/skills", "/mcp", "/instructions", "/config"}:
            if value in {"/mcp reconnect", "/mcp reload"}:
                await self._reconnect_mcp()
            elif value.endswith(" reload"):
                await self._reload_workspace()
            elif value == "/skills":
                await self._show_skills()
            elif value == "/mcp":
                await self._show_mcp()
            elif value in {"/instructions", "/config"}:
                await self._show_workspace()
            else:
                await self._write_meta("Usage: /skills, /mcp [reconnect|reload], /instructions, /config [reload]")
            return

        if self._workspace_loading:
            await self._write_meta("[dim]Workspace is loading; please wait.[/]")
            return

        if value.lower() == "exit":
            self.exit()
            return

        if self._is_running:
            widget = Static(f"[dim]You (queued):[/] {value}", classes="user-msg")
            scroll = self.query_one("#log", VerticalScroll)
            await scroll.mount(widget)
            scroll.scroll_end(animate=False)
            self._pending_input.put_nowait((value, widget))
            return

        await self._write_user(f"[bold]You:[/] {value}")
        self._worker = self.run_worker(self._run_agent_work(value), exclusive=True)

    def action_stop_agent(self) -> None:
        if self._worker is not None and self._worker.is_running:
            self._worker.cancel()

    async def _run_agent_work(self, task: str) -> None:
        if self._agent is None:
            await self._write_meta("[yellow]Configure a transport from the command palette first.[/]")
            return

        self._is_running = True
        if self._should_compact():
            messages = await self._compact_with_progress()
            if messages is not None:
                await self._replace_context(messages)
                self._last_input_tokens = 0
                await self._write_meta("[dim]--- Context compacted ---[/]")

        cancelled = False
        try:
            while True:
                self._agent_emoji = "💬"
                await self._update_status()
                stream = self._agent.run_stream(task, self._chat_context)
                try:
                    async for event in stream:
                        match event:
                            case ReasoningDelta(delta=delta):
                                self._tool_status = None
                                self._agent_emoji = "🤔"
                                (await self._ensure_thinking()).add(delta)
                            case TextDelta(delta=delta):
                                self._tool_status = None
                                self._agent_emoji = "💬"
                                if self._thinking is not None:
                                    self._thinking.collapse()
                                await self._ensure_md()
                                self._response_text += delta
                                self._text_dirty = True
                            case Refusal(text=text, category=category, blocked_input=blocked):
                                # Rendered, and not as ordinary text: a declined turn otherwise looked like the model
                                # answering with nothing. The banner goes once, because a refusal arrives in fragments.
                                self._tool_status = None
                                self._agent_emoji = "🚫"
                                await self._ensure_md()
                                if not self._declined:
                                    self._declined = True
                                    what = "Prompt blocked" if blocked else "Declined"
                                    tail = f" ({category})" if category else ""
                                    self._response_text += f"\n\n**{what}{tail}.** "
                                self._response_text += text
                                self._text_dirty = True
                            case Citation(cited_text=cited, title=title, url=url):
                                await self._ensure_md()
                                self._response_text += _citation_markdown(title, cited, url)
                                self._text_dirty = True
                            case ToolUseStart(tool_use_id=tid, name=name) if name != "status_line":
                                self._agent_emoji = "🔧"
                                await self._update_status()
                                w = await self._ensure_tool_status()
                                w.track(tid, name)
                                # Start a new arg stream for this tool
                                self._tool_arg_streams[tid] = ToolArgStream(tid)
                            case ToolInputDelta(tool_use_id=tid, partial_json=json_chunk):
                                arg_stream = self._tool_arg_streams.get(tid)
                                if arg_stream is not None:
                                    for field_event in arg_stream.feed(json_chunk):
                                        await self._handle_tool_field_event(field_event, tid)
                            case ToolResult(
                                tool_use_id=tid,
                                name=name,
                                is_error=is_error,
                                content=content,
                                input=inp,
                            ) if name != "status_line":
                                w = await self._ensure_tool_status()
                                w.complete(tid, is_error=is_error, content=content, tool_input=inp)
                            case IterationEnd(usage=usage):
                                # One arm, or a turn that reasoned skips the status update below.
                                if self._thinking is None and usage.reasoning_tokens:
                                    # Billed for reasoning and none streamed: the provider
                                    # withheld it. The answer is already on screen.
                                    await self._mount_thinking(before_answer=True)
                                await self._close_thinking(usage.reasoning_tokens)
                                self._agent_emoji = "✅"
                                self._last_input_tokens = usage.input_tokens
                                await self._update_status()
                            case SessionEndEvent(stop_reason=reason):
                                await self._close_thinking()
                                if reason in INCOMPLETE:
                                    # Nothing else says so, and the answer reads as a whole one.
                                    await self._write_meta(f"[red]Incomplete: {reason}[/]")
                                else:
                                    await self._flush_text()
                                    self._current_md = None
                                    self._response_text = ""
                                    self._declined = False
                                self._tool_status = None
                            case Error(exception=exc):
                                describe = getattr(self._agent.transport, "format_error", str)
                                detail = describe(exc) if isinstance(exc, Exception) else type(exc).__name__
                                await self._write_meta(f"[red]Error: {escape(detail)}[/]")
                finally:
                    await stream.aclose()

                if not self._pending_input.empty():
                    task, widget = self._pending_input.get_nowait()
                    widget.update(f"[bold]You:[/] {task}")
                    continue
                break
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            await self._write_meta(f"[red]Error: {exc}[/]")
        finally:
            self._is_running = False
            self._agent_emoji = ""
            self._hide_agent_status()
            # Drain stale queued messages on cancellation or error
            while not self._pending_input.empty():
                self._pending_input.get_nowait()
            self._tool_status = None
            await self._close_thinking()
            await self._update_status()
            if cancelled:
                await self._flush_text()
                self._current_md = None
                self._response_text = ""
                self._declined = False
                await self._write_meta("[yellow]--- Cancelled ---[/]")
            self.query_one("#input", Input).focus()

    def action_toggle_dev_console(self) -> None:
        self.query_one("#dev-console", RichLog).toggle_class("visible")

    # -- Chat message navigation (Alt+Up / Alt+Down) --

    def _nav_children(self) -> list[Widget]:
        return list(self.query_one("#log").children)

    def _nav_select(self, index: int | None) -> None:
        children = self._nav_children()
        if self._nav_index is not None and self._nav_index < len(children):
            children[self._nav_index].remove_class("nav-selected")
        self._nav_index = index
        if index is not None and index < len(children):
            children[index].add_class("nav-selected")
            children[index].scroll_visible()

    def _nav_exit(self) -> None:
        self._nav_select(None)
        self.query_one("#input", Input).focus()

    def _nav_activate(self) -> None:
        if self._nav_index is None:
            return
        children = self._nav_children()
        if not children or self._nav_index >= len(children):
            return
        widget = children[self._nav_index]
        if isinstance(widget, _ToolStatusWidget):
            for tid, info in widget._tools.items():
                if info.status is not None:
                    widget.action_show_detail(tid)
                    return

    def action_nav_up(self) -> None:
        children = self._nav_children()
        if not children:
            return
        if self._nav_index is None:
            self._nav_select(len(children) - 1)
        elif self._nav_index > 0:
            self._nav_select(self._nav_index - 1)

    def action_nav_down(self) -> None:
        children = self._nav_children()
        if not children:
            return
        if self._nav_index is None:
            self._nav_select(0)
        elif self._nav_index < len(children) - 1:
            self._nav_select(self._nav_index + 1)
        else:
            self._nav_exit()

    def on_key(self, event: Key) -> None:
        if self._nav_index is None:
            return
        if isinstance(self.screen, ModalScreen):
            return
        if isinstance(self.focused, Input):
            return
        if event.key == "escape":
            self._nav_exit()
            event.prevent_default()
            event.stop()
        elif event.key == "enter":
            self._nav_activate()
            event.prevent_default()
            event.stop()

    async def action_clear_log(self) -> None:
        await self.query_one("#log", VerticalScroll).remove_children()
        self._current_md = None
        self._response_text = ""
        self._declined = False
        self._tool_status = None
        # Removed with every other child, so what is left here is a detached widget. Dropped
        # rather than folded, because the pane it pointed at is gone with the rest of the log.
        self._thinking = None
        self._nav_index = None

    def action_quit(self) -> None:  # type: ignore[override]
        self.push_screen(QuitDialog(), callback=self._on_quit_confirmed)

    def _on_quit_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()
