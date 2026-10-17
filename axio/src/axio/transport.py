"""Transport protocols and application plugin discovery."""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .blocks import ContentBlock, ImageBlock
from .events import StreamEvent
from .messages import Message
from .models import ModelSpec
from .plugins import PluginSource, load_entry_points
from .tool import Tool
from .types import ToolCallID, ToolName

logger = logging.getLogger(__name__)


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]: ...


class TransportAuth(Protocol):
    """Account operations owned by a transport package, without frontend imports."""

    async def has_accounts(self, credentials: Path | None = None) -> bool: ...
    async def describe_accounts(self, credentials: Path | None, transport: Any) -> str: ...
    async def prompt_login_if_needed(self, credentials: Path | None, account: str | None) -> None: ...
    async def sign_in(self, credentials: Path | None, account: str | None) -> str: ...
    async def select_account(self, credentials: Path | None, account: str) -> None: ...
    async def logout(self, transport: Any) -> None: ...
    def account_id(self, transport: Any) -> str: ...


@runtime_checkable
class TransportPlugin(Protocol):
    """An entry point may expose a plugin object or a transport class.

    Plugin objects implement this contract. Legacy classes use TransportEntry's
    defaults and can supply the same metadata as class attributes. Factories
    yield a ready transport with its model catalog loaded.
    """

    display_name: str
    env_vars: tuple[str, ...]
    supports_temperature: bool
    supports_max_tokens: bool
    supports_debug: bool
    supports_thinking: bool
    auth: TransportAuth | None

    def has_credentials(self, config: Mapping[str, str] | None = None) -> bool: ...
    def open(self, *, session: Any, **settings: Any) -> AbstractAsyncContextManager[Any]: ...


@runtime_checkable
class ImageGenTransport(Protocol):
    """Generate ``n`` image samples for a text prompt.  Returns raw image bytes
    (PNG / JPEG / WebP — provider-defined)."""

    async def generate_images(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]: ...


@runtime_checkable
class ImageEditTransport(Protocol):
    """Edit supplied images according to a prompt, returning complete image bytes.

    Sources are required. Implementations must reject unsupported masks instead
    of ignoring them; supported formats and counts are provider-specific.
    """

    async def edit_images(
        self,
        prompt: str,
        *,
        images: list[ImageBlock],
        mask: ImageBlock | None = None,
        model: str | None = None,
        n: int = 1,
    ) -> list[bytes]: ...


@runtime_checkable
class VideoGenTransport(Protocol):
    """Generate ``n`` video samples for a text prompt.  Returns raw video bytes
    (MP4 / WebM — provider-defined).  Provider-specific knobs (duration,
    aspect ratio, seed image, etc.) live as extra kwargs on the implementation."""

    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        image: bytes | None = None,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]: ...


@runtime_checkable
class AudioGenTransport(Protocol):
    """Generate ``n`` non-speech audio samples for a text prompt — music,
    sound effects, ambient.  Returns raw audio bytes (MP3 / WAV / OGG —
    provider-defined).  Distinct from :class:`TTSTransport`, which is
    text-to-speech."""

    async def generate_audios(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]: ...


@runtime_checkable
class RealtimeSession(Protocol):
    """Active duplex realtime session — bidirectional audio / text / tools.

    Returned by :meth:`RealtimeTransport.connect`.  Events from the provider
    arrive on :meth:`events`.  User input is pushed via :meth:`send`.
    """

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        """Append user content (audio chunk, text, image) to the input buffer."""

    async def commit(self) -> None:
        """Signal end-of-utterance for manual VAD; no-op with server VAD."""

    async def interrupt(self) -> None:
        """Abort in-flight assistant generation (e.g. user interrupted)."""

    async def send_tool_result(
        self, tool_use_id: ToolCallID, name: ToolName, content: str | list[ContentBlock]
    ) -> None:
        """Deliver a tool's result to the provider so generation can resume.

        ``name`` is included because some providers (e.g. Gemini Live) require
        the tool name alongside the call id.  OpenAI realtime can ignore it.
        """

    def events(self) -> AsyncIterator[StreamEvent]:
        """Async iterator over server events for the lifetime of this session."""

    async def close(self) -> None:
        """Tear down the session and release resources."""


@runtime_checkable
class RealtimeTransport(Protocol):
    """Provider that supports duplex realtime sessions (e.g. OpenAI Realtime,
    Gemini Live).  Distinct from :class:`CompletionTransport` because the
    interaction is bidirectional, not request/response."""

    async def connect(
        self,
        *,
        system: str,
        tools: list[Tool[Any]],
        voice: str | None = None,
        input_audio_format: str = "audio/pcm;rate=16000",
        output_audio_format: str = "audio/pcm;rate=24000",
    ) -> RealtimeSession: ...


@runtime_checkable
class TTSTransport(Protocol):
    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]: ...


@runtime_checkable
class STTTransport(Protocol):
    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str: ...


@runtime_checkable
class EmbeddingTransport(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class DummyTransport:
    """Placeholder transport that fails loudly if actually used.

    Assign this as the default transport when constructing agent prototypes
    that will be configured later via ``agent.copy(transport=real_transport)``.

    Example::

        from axio.agent import Agent
        from axio.transport import DummyCompletionTransport

        researcher = Agent(
            system="You are a research assistant...",
            transport=DummyCompletionTransport(),
        )

        # At runtime, swap in the real transport:
        active = researcher.copy(transport=OpenAITransport())
        result = await active.run(task, context)
    """

    @staticmethod
    def _do_fail() -> None:
        logger.warning(
            "DummyCompletionTransport.stream() called - this agent has no real transport. "
            "Use agent.copy(transport=<real_transport>) before running."
        )
        raise RuntimeError(
            "DummyCompletionTransport is a placeholder. Configure a real transport with agent.copy(transport=...)."
        )


class DummyCompletionTransport(DummyTransport, CompletionTransport):
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyImageGenTransport(DummyTransport, ImageGenTransport):
    async def generate_images(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyImageEditTransport(DummyTransport, ImageEditTransport):
    async def edit_images(
        self,
        prompt: str,
        *,
        images: list[ImageBlock],
        mask: ImageBlock | None = None,
        model: str | None = None,
        n: int = 1,
    ) -> list[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyVideoGenTransport(DummyTransport, VideoGenTransport):
    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        image: bytes | None = None,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyAudioGenTransport(DummyTransport, AudioGenTransport):
    async def generate_audios(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyTTSTransport(DummyTransport, TTSTransport):
    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummySTTTransport(DummyTransport, STTTransport):
    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyEmbeddingTransport(DummyTransport, EmbeddingTransport):
    def embed(self, texts: list[str]) -> Any:
        self._do_fail()
        raise AssertionError("unreachable")


class TransportError(RuntimeError):
    """An installed transport could not be loaded, configured, or opened."""


def select_transport_model(transport: Any, model: ModelSpec) -> Any:
    """Clone a transport while preserving its session and provider settings."""
    select = getattr(transport, "select_model", None)
    if callable(select):
        return select(transport, model)
    clone = getattr(transport, "with_model", None)
    if callable(clone):
        return clone(model)
    selected = copy.copy(transport)
    selected.model = model
    return selected


@dataclass(frozen=True, slots=True)
class TransportEntry:
    """Adapt a plugin object or a legacy transport class to one application API."""

    name: str
    cls: Any
    display_name: str
    env_vars: tuple[str, ...] = ()
    supports_temperature: bool = True
    supports_max_tokens: bool = True
    supports_debug: bool = True
    supports_thinking: bool = True
    auth: TransportAuth | None = None
    auto_select: bool = True

    @classmethod
    def from_target(cls, name: str, target: Any) -> TransportEntry:
        if not callable(target) and not callable(getattr(target, "open", None)):
            raise TypeError("Expected a transport class or a plugin with open().")
        label = getattr(target, "display_name", name)
        env_vars = getattr(target, "env_vars", ())
        if (
            not isinstance(label, str)
            or not isinstance(env_vars, tuple)
            or any(not isinstance(value, str) for value in env_vars)
        ):
            raise TypeError("display_name must be a string and env_vars must be a tuple of strings.")
        return cls(
            name,
            target,
            label,
            env_vars,
            supports_temperature=getattr(target, "supports_temperature", True),
            supports_max_tokens=getattr(target, "supports_max_tokens", True),
            supports_debug=getattr(target, "supports_debug", True),
            supports_thinking=getattr(target, "supports_thinking", True),
            auth=getattr(target, "auth", None),
            auto_select=getattr(target, "auto_select", True),
        )

    def has_credentials(self, config: Mapping[str, str] | None = None) -> bool:
        check = getattr(self.cls, "has_credentials", None)
        if callable(check):
            return bool(check(config))
        return any(os.environ.get(key) or (config or {}).get(key) for key in self.env_vars)

    def format_error(self, error: BaseException) -> str:
        describe = getattr(self.cls, "format_error", str)
        return str(describe(error))

    @asynccontextmanager
    async def open(
        self,
        *,
        session: Any,
        account: str | None = None,
        credentials: Path | str | None = None,
        on_auth_refresh: Callable[..., Any] | None = None,
        **settings: Any,
    ) -> AsyncIterator[Any]:
        """Yield a ready transport; the caller owns the shared HTTP session.

        Managed factories own catalog loading and cleanup. Legacy constructors
        receive session and saved settings; this adapter loads their catalog.
        """
        kwargs = {key: value for key, value in settings.items() if value is not None}
        factory = getattr(self.cls, "open_for_app", None) or getattr(self.cls, "open", None)
        if callable(factory):
            if account is not None:
                kwargs["account"] = account
            if credentials is not None:
                kwargs["credentials"] = str(credentials)
            async with factory(session=session, **kwargs) as transport:
                yield transport
            return
        if account is not None or credentials is not None:
            raise TransportError(f"Transport {self.name!r} does not support account settings.")
        transport = self.cls(session=session, **kwargs)
        if on_auth_refresh is not None and hasattr(transport, "on_auth_refresh"):
            transport.on_auth_refresh = on_auth_refresh
        fetch = getattr(transport, "fetch_models", None)
        if callable(fetch):
            await fetch()
        yield transport

    def select_model(self, transport: Any, model: ModelSpec) -> Any:
        select = getattr(self.cls, "select_model", None)
        return select(transport, model) if callable(select) else select_transport_model(transport, model)

    def bind_tools(self, transport: Any | None) -> None:
        """Let the provider connect its optional tool handlers to a transport."""
        bind = getattr(self.cls, "bind_tools", None)
        if callable(bind):
            bind(transport)


@dataclass
class TransportCatalog:
    """Loaded entries and failures, including failures explicitly requested by name."""

    entries: dict[str, TransportEntry] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    sources: dict[str, PluginSource] = field(default_factory=dict)

    def require(self, name: str) -> TransportEntry:
        if name in self.errors:
            raise TransportError(f"Transport {name!r} failed to load: {self.errors[name]}. Check installation.")
        if name not in self.entries:
            raise TransportError(f"Unknown transport {name!r}. Available: {', '.join(sorted(self.entries))}")
        return self.entries[name]


def discover_transports() -> TransportCatalog:
    """Load axio.transport entries. The first declaration of each name wins.

    A failed first declaration reserves its name too. This prevents an unrelated
    package from silently replacing the requested provider after a load failure.
    """
    catalog = load_entry_points("axio.transport", TransportEntry.from_target)
    return TransportCatalog(catalog.entries, catalog.errors, catalog.sources)
