"""TransportRegistry - discovers, initialises and manages transport plugins."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import aiohttp
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.transport import TransportEntry

from .plugin import discover_transport_settings, discover_transports
from .sqlite_config import ProjectConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """Associates a model role with a specific transport and model."""

    transport: str
    model: ModelSpec


@dataclass(slots=True)
class TransportRegistry:
    """Discovers transports, creates instances, and manages them."""

    _lifetimes: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)
    errors: dict[str, str] = field(default_factory=dict)
    _transports: dict[str, Any] = field(default_factory=dict, repr=False)
    _entries: dict[str, TransportEntry] = field(default_factory=dict, repr=False)
    _screens: dict[str, type] = field(default_factory=dict)
    _saved: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)
    _config: ProjectConfig | None = field(default=None, repr=False)
    _session: aiohttp.ClientSession | None = field(default=None, repr=False)

    async def init(
        self,
        session: aiohttp.ClientSession,
        config: ProjectConfig | None = None,
        global_config: ProjectConfig | None = None,
    ) -> None:
        """Discover transports, create instances for those that initialise successfully."""
        self._session = session
        self._config = global_config or config
        catalog = discover_transports()
        self._entries = catalog.entries
        self.errors = dict(catalog.errors)
        self._screens = discover_transport_settings()

        async def open_one(name: str, entry: TransportEntry) -> None:
            saved = await self._load_settings(self._config, name) if self._config is not None else {}
            self._saved[name] = saved
            await self._open_entry(name, entry, saved)

        await asyncio.gather(*(open_one(name, entry) for name, entry in self._entries.items()))
        self._transports = {name: self._transports[name] for name in self._entries if name in self._transports}

        # Bulk-load providers from hub screens that expose load_config()
        # (e.g. CustomHubScreen for openai-custom.* providers)
        for screen_name, screen_cls in self._screens.items():
            load = getattr(screen_cls, "load_config", None)
            if not callable(load):
                continue
            try:
                providers = await asyncio.to_thread(load, session)
                for t in providers:
                    self.register_dynamic(f"{screen_name}.{t.name}", t)
            except Exception:
                logger.warning("Failed to bulk-load providers from %r", screen_name, exc_info=True)

    async def _open_entry(self, name: str, entry: TransportEntry, settings: dict[str, str]) -> None:
        try:
            transport = await self._lifetimes.enter_async_context(
                entry.open(
                    session=self._session,
                    on_auth_refresh=partial(self._persist_auth, name),
                    **{key: value for key, value in settings.items() if value != ""},
                )
            )
        except Exception as error:
            self._transports.pop(name, None)
            self.errors[name] = entry.format_error(error)
            logger.warning("Transport %r is unavailable (%s)", name, type(error).__name__)
        else:
            self._transports[name] = transport
            self.errors.pop(name, None)

    async def close(self) -> None:
        """Close account sessions after all agent runs have stopped."""
        await self._lifetimes.aclose()

    @staticmethod
    async def _load_settings(config: ProjectConfig, name: str) -> dict[str, str]:
        """Load saved transport settings from config, stripping the key prefix."""
        prefix = f"transport.{name}."
        raw = await config.get_prefix(prefix)
        return {k[len(prefix) :]: v for k, v in raw.items()}

    async def get_settings(self, name: str) -> dict[str, str]:
        """Return saved settings for a transport from the config DB."""
        if self._config is None:
            return dict(self._saved.get(name, {}))
        return await self._load_settings(self._config, name)

    async def save_settings(self, name: str, settings: dict[str, str]) -> None:
        """Persist settings to config and re-create the transport instance."""
        if self._config is not None:
            prefix = f"transport.{name}."
            await self._config.delete_prefix(prefix)
            for key, value in settings.items():
                if value:
                    await self._config.set(f"{prefix}{key}", value)

        self._saved[name] = settings

        if name in self._entries and self._session is not None:
            await self._open_entry(name, self._entries[name], settings)

    @property
    def available(self) -> list[str]:
        """Names of transports that were successfully initialised."""
        return list(self._transports)

    @property
    def discovered(self) -> list[str]:
        """Names of all discovered transports (including those that failed to initialise)."""
        return list(dict.fromkeys([*self._entries, *self.errors]))

    def get_transport(self, name: str) -> Any:
        """Return the initialised transport instance by name."""
        return self._transports[name]

    def display_name(self, name: str) -> str:
        entry = self._entries.get(name)
        default = entry.display_name if entry is not None else name
        return str(getattr(self._transports.get(name), "name", default))

    def all_models(self, *caps: Capability) -> list[tuple[str, ModelSpec]]:
        """Return (transport_name, ModelSpec) across all transports, optionally filtered."""
        result: list[tuple[str, ModelSpec]] = []
        required = frozenset(caps)
        for name, transport in self._transports.items():
            registry: ModelRegistry = transport.models
            for spec in registry.values():
                if required and not (required <= spec.capabilities):
                    continue
                result.append((name, spec))
        return result

    def default_model(self, role: str, capability: Capability) -> RoleBinding | None:
        for name, entry in self._entries.items():
            transport = self._transports.get(name)
            model_id = getattr(entry.cls, "default_models", {}).get(role)
            if transport is not None and model_id in transport.models:
                return RoleBinding(name, transport.models[model_id])
        choices = self.all_models(capability, Capability.tool_use)
        return RoleBinding(*choices[0]) if choices else None

    def make_transport(self, name: str, model: ModelSpec) -> Any:
        """Create a new transport instance for the given name and model."""
        return self._entries[name].select_model(self._transports[name], model)

    def bind_tools(self) -> None:
        self.unbind_tools()
        for name, entry in self._entries.items():
            if name in self._transports:
                entry.bind_tools(self._transports[name])

    def unbind_tools(self) -> None:
        for entry in self._entries.values():
            entry.bind_tools(None)

    async def _persist_auth(self, name: str, tokens: dict[str, str]) -> None:
        if self._config is None:
            return
        prefix = f"transport.{name}."
        for key, value in tokens.items():
            if value:
                await self._config.set(f"{prefix}{key}", value)

    def register_dynamic(self, name: str, instance: Any) -> None:
        """Register a dynamically-created transport instance (e.g. custom providers)."""
        self._transports[name] = instance
        self._entries[name] = TransportEntry.from_target(name, type(instance))

    def unregister_by_prefix(self, prefix: str) -> None:
        """Remove all transports whose names start with *prefix*."""
        names = [n for n in list(self._transports) if n.startswith(prefix)]
        for n in names:
            del self._transports[n]
            self._entries.pop(n, None)
            self._saved.pop(n, None)

    def resolve(self, config_value: str) -> RoleBinding | None:
        """Parse a config value like ``"nebius:model_id"`` into a RoleBinding.

        Falls back to searching all transports if no prefix is present (migration).
        Returns None if the model cannot be found in any transport.
        """
        if ":" in config_value:
            transport_name, model_id = config_value.split(":", 1)
            if transport_name in self._transports:
                transport = self._transports[transport_name]
                if model_id in transport.models:
                    return RoleBinding(transport=transport_name, model=transport.models[model_id])
            return None

        # Migration: bare model ID - search all transports
        for name, transport in self._transports.items():
            if config_value in transport.models:
                return RoleBinding(transport=name, model=transport.models[config_value])
        return None

    def encode(self, name: str, model_id: str) -> str:
        """Encode a transport name and model ID for config persistence."""
        return f"{name}:{model_id}"

    def model_counts(self) -> dict[str, int]:
        """Return number of models per available transport."""
        return {name: len(t.models) for name, t in self._transports.items()}

    async def reload_models(self, name: str | None = None) -> None:
        """Re-fetch model catalogues for one transport (or all available if name is None)."""
        names = [name] if (name and name in self._transports) else list(self._transports)
        await asyncio.gather(*[self._reload_one(n) for n in names])

    async def _reload_one(self, name: str) -> None:
        transport = self._transports[name]
        transport.models.clear()
        try:
            await transport.fetch_models()
        except Exception:
            logger.warning("Transport %r: fetch_models failed on reload", name, exc_info=True)

    def settings_screens(self) -> dict[str, type]:
        """Return discovered settings screen classes keyed by transport name."""
        return dict(self._screens)
