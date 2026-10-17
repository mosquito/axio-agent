"""Plugin discovery via entry points."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from axio.permission import PermissionGuard
from axio.plugins import (
    ToolLifecycle,
    load_entry_points,
)
from axio.plugins import (
    discover_guards as core_discover_guards,
)
from axio.plugins import (
    discover_selectors as core_discover_selectors,
)
from axio.plugins import (
    discover_tools as core_discover_tools,
)
from axio.tool import Tool
from axio.transport import discover_transports as discover_transports


@runtime_checkable
class ToolsPlugin(ToolLifecycle, Protocol):
    """Protocol for dynamic tool provider plugins.

    Plugins register via the ``axio.tools.settings`` entry point group.
    The TUI discovers them, calls ``init()``, collects tools, and shows
    settings screens - without knowing anything about the plugin internals.
    """

    @property
    def label(self) -> str: ...

    def settings_screen(self) -> Any: ...


TOOLS_GROUP = "axio.tools"
TRANSPORT_GROUP = "axio.transport"
TRANSPORT_SETTINGS_GROUP = "axio.transport.settings"
GUARDS_GROUP = "axio.guards"
TOOLS_SETTINGS_GROUP = "axio.tools.settings"
SELECTOR_GROUP = "axio.selector"


def discover_tools() -> list[Tool[Any]]:
    """Return handlers from the shared core catalog."""
    return list(core_discover_tools(points=entry_points(group=TOOLS_GROUP)).entries.values())


def discover_tools_by_package() -> dict[str, list[Tool[Any]]]:
    """Group the shared tool catalog by its declaring distribution."""
    catalog = core_discover_tools(points=entry_points(group=TOOLS_GROUP))
    groups: dict[str, list[Tool[Any]]] = {}
    for name, tool in catalog.entries.items():
        groups.setdefault(catalog.sources[name].distribution, []).append(tool)
    return groups


def _screen_from_target(name: str, target: Any) -> type:
    if not isinstance(target, type):
        raise TypeError("Expected a settings screen class")
    return target


def discover_transport_settings() -> dict[str, type]:
    """Load UI screens with the same conflict rules as the core catalog."""
    return load_entry_points(
        TRANSPORT_SETTINGS_GROUP,
        _screen_from_target,
        points=entry_points(group=TRANSPORT_SETTINGS_GROUP),
    ).entries


def _tools_plugin_from_target(name: str, target: Any) -> ToolsPlugin:
    plugin = target()
    if not isinstance(plugin, ToolsPlugin):
        raise TypeError("Expected a tool provider with a settings screen")
    return plugin


def discover_tools_plugins() -> dict[str, ToolsPlugin]:
    """Keep the legacy UI group separate from axio.tools.providers."""
    return load_entry_points(
        TOOLS_SETTINGS_GROUP,
        _tools_plugin_from_target,
        points=entry_points(group=TOOLS_SETTINGS_GROUP),
    ).entries


def discover_selectors() -> dict[str, type]:
    return dict(core_discover_selectors(points=entry_points(group=SELECTOR_GROUP)).entries)


def discover_guards() -> dict[str, type[PermissionGuard]]:
    return core_discover_guards(points=entry_points(group=GUARDS_GROUP)).entries
