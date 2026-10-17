"""Common entry-point discovery and tool-provider lifecycle, without UI imports."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .context import ContextStore
from .instructions import Instruction
from .permission import PermissionGuard
from .selector import ToolSelector
from .tool import Tool

logger = logging.getLogger(__name__)

TOOLS_GROUP = "axio.tools"
TOOL_PROVIDERS_GROUP = "axio.tools.providers"
GUARDS_GROUP = "axio.guards"
SELECTOR_GROUP = "axio.selector"


class PluginError(RuntimeError):
    """An extension could not be loaded or conflicts with another extension."""


@dataclass(frozen=True, slots=True)
class PluginSource:
    """The installed declaration that owns an entry-point name."""

    group: str
    distribution: str
    value: str


@dataclass
class PluginCatalog[T]:
    """Successfully loaded extensions, failures, and their installed sources."""

    entries: dict[str, T] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    sources: dict[str, PluginSource] = field(default_factory=dict)

    def require(self, name: str) -> T:
        if name in self.errors:
            raise PluginError(f"Plugin {name!r} failed to load: {self.errors[name]}")
        if name not in self.entries:
            raise PluginError(f"Unknown plugin {name!r}. Available: {', '.join(sorted(self.entries))}")
        return self.entries[name]


def load_entry_points[T](
    group: str,
    adapt: Callable[[str, Any], T],
    *,
    points: Iterable[EntryPoint] | None = None,
) -> PluginCatalog[T]:
    """Load each name once, retaining failures and reserving the first declaration.

    Discovery does not initialize providers. An adapter validates the contract
    for its own group. A broken declaration cannot be replaced by a duplicate.
    """
    from importlib.metadata import entry_points

    catalog: PluginCatalog[T] = PluginCatalog()
    for ep in entry_points(group=group) if points is None else points:
        if ep.name in catalog.sources:
            logger.warning("Duplicate %s entry point %r ignored (first declaration wins)", group, ep.name)
            continue
        dist = getattr(ep, "dist", None)
        catalog.sources[ep.name] = PluginSource(group, getattr(dist, "name", "unknown"), getattr(ep, "value", ""))
        try:
            catalog.entries[ep.name] = adapt(ep.name, ep.load())
        except Exception as error:
            catalog.errors[ep.name] = str(error)
            logger.warning("Failed to load %s entry point %r: %s", group, ep.name, error)
    return catalog


@runtime_checkable
class ToolLifecycle(Protocol):
    """The working interface shared by current and legacy tool providers."""

    async def init(self, config: Any = None, global_config: Any = None) -> None: ...

    @property
    def all_tools(self) -> list[Tool[Any]]: ...

    async def close(self) -> None: ...


@runtime_checkable
class ToolProvider(ToolLifecycle, Protocol):
    """A dynamic tool set and instruction catalog owned by an installed provider.

    Register a zero-argument factory in ``axio.tools.providers``. The factory
    must not acquire resources. ``init`` receives application configuration;
    only the provider interprets its contents. ``close`` must also work after
    a failed or cancelled initialization.
    """

    def get_instructions(self) -> Sequence[Instruction]: ...


@runtime_checkable
class RefreshableToolProvider(Protocol):
    """An optional capability for refreshing tools and instructions in place."""

    async def refresh(self, *, config: Any = None) -> None: ...


def tool_from_handler(name: str, handler: Any) -> Tool[Any]:
    """Validate a plain async handler and preserve its concurrency setting."""
    if not callable(handler) or not inspect.iscoroutinefunction(handler):
        raise TypeError("Expected an async tool handler")
    return Tool(name=name, handler=handler, concurrency=getattr(handler, "_tool_concurrency", None))


def discover_tools(*, points: Iterable[EntryPoint] | None = None) -> PluginCatalog[Tool[Any]]:
    return load_entry_points(TOOLS_GROUP, tool_from_handler, points=points)


def guard_from_target(name: str, target: Any) -> type[PermissionGuard]:
    if not isinstance(target, type) or not issubclass(target, PermissionGuard):
        raise TypeError("Expected a PermissionGuard subclass")
    return target


def discover_guards(*, points: Iterable[EntryPoint] | None = None) -> PluginCatalog[type[PermissionGuard]]:
    return load_entry_points(GUARDS_GROUP, guard_from_target, points=points)


def selector_from_target(name: str, target: Any) -> type[ToolSelector]:
    if not isinstance(target, type) or not issubclass(target, ToolSelector):
        raise TypeError("Expected a ToolSelector class")
    return target


def discover_selectors(*, points: Iterable[EntryPoint] | None = None) -> PluginCatalog[type[ToolSelector]]:
    return load_entry_points(SELECTOR_GROUP, selector_from_target, points=points)


def provider_from_target(name: str, target: Any) -> ToolProvider:
    if not callable(target):
        raise TypeError("Expected a tool-provider factory")
    provider = target()
    if not isinstance(provider, ToolProvider):
        raise TypeError("Expected a ToolProvider with init, close, all_tools and get_instructions")
    return provider


def discover_tool_providers(*, points: Iterable[EntryPoint] | None = None) -> PluginCatalog[ToolProvider]:
    return load_entry_points(TOOL_PROVIDERS_GROUP, provider_from_target, points=points)


def collect_tools(providers: Iterable[ToolLifecycle], *, tools: Iterable[Tool[Any]] = ()) -> list[Tool[Any]]:
    """Collect current tools, rejecting different tools with the same name."""
    result: dict[str, Tool[Any]] = {}
    for tool in (*tools, *(tool for provider in providers for tool in provider.all_tools)):
        if tool.name in result and result[tool.name] is not tool:
            raise PluginError(f"Conflicting tool name: {tool.name!r}")
        result[tool.name] = tool
    return list(result.values())


@asynccontextmanager
async def open_tool_providers[T: ToolLifecycle](
    providers: Iterable[T], *, config: Any = None, global_config: Any = None
) -> AsyncIterator[tuple[T, ...]]:
    """Initialize providers and close them in reverse order on every exit path.

    Cleanup is registered before initialization so partial startup is covered.
    A startup error propagates after closing the failing and earlier providers.
    The caller may use separate contexts when startup failures must be isolated.
    """
    async with AsyncExitStack() as stack:
        opened: list[T] = []
        for provider in providers:
            stack.push_async_callback(provider.close)
            await provider.init(config=config, global_config=global_config)
            opened.append(provider)
        collect_tools(opened)
        yield tuple(opened)


def standalone_tools(tools: PluginCatalog[Tool[Any]], providers: PluginCatalog[ToolProvider]) -> list[Tool[Any]]:
    """Prefer a distribution's initialized provider over its duplicate plain handlers.

    Different distributions still conflict. Plain handlers without a matching
    provider tool remain available.
    """
    supplied = {
        (tool.name, providers.sources[name].distribution)
        for name, provider in providers.entries.items()
        if name in providers.sources
        for tool in provider.all_tools
    }
    failed = {providers.sources[name].distribution for name in providers.errors if name in providers.sources}
    return [
        tool
        for name, tool in tools.entries.items()
        if name not in tools.sources
        or (
            tools.sources[name].distribution not in failed and (name, tools.sources[name].distribution) not in supplied
        )
    ]


@runtime_checkable
class ContextBackend(Protocol):
    """An installed store factory that owns resources shared by its sessions."""

    async def init(self, path: Path) -> None: ...

    def create(self, session_id: str | None = None) -> ContextStore: ...

    async def close(self) -> None: ...


def context_backend_from_target(name: str, target: Any) -> ContextBackend:
    backend = target()
    if not isinstance(backend, ContextBackend):
        raise TypeError("Expected a ContextBackend with init, create and close")
    return backend


def discover_context_backends() -> PluginCatalog[ContextBackend]:
    return load_entry_points("axio.context", context_backend_from_target)
