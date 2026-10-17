"""ChatGPT subscription completion and image generation with shared OAuth sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
from contextlib import AbstractAsyncContextManager, aclosing
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import aiohttp
from axio.blocks import (
    ContentBlock,
    ImageBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from axio.events import IterationEnd, IterationStart, StreamEvent
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.types import Usage
from axio_responses import ResponsesAPIError, convert_messages, convert_tools

from .app_auth import ChatGPTAuth
from .chatgpt_images import ChatGPTImageTransport
from .chatgpt_models import (
    CHATGPT_IMAGE_MODELS,
    ChatGPTImageModel,
    ChatGPTModel,
    parse_models,
    supported_reasoning_efforts,
)
from .session import ChatGPTSession
from .subscription import CODEX_BASE_URL, ChatGPTResponses


def validate_block(block: ContentBlock, role: str, model: ModelSpec) -> None:
    """Reject inputs that the shared Responses converter would drop or replace."""
    if isinstance(block, TextBlock):
        return
    if isinstance(block, ImageBlock) and role in ("user", "tool"):
        if Capability.vision not in model.capabilities:
            raise ValueError(f"Image input is not enabled for {model.id}.")
        return
    if isinstance(block, (ReasoningBlock, ToolUseBlock)) and role == "assistant":
        return
    if isinstance(block, ToolResultBlock) and role == "user":
        if isinstance(block.content, list):
            for part in block.content:
                validate_block(part, "tool", model)
        return
    if isinstance(block, ProviderBlock) and role == "assistant":
        # Web search history is supported by this route. Other hosted tool output
        # and injected tool declarations must not bypass current tool selection.
        if block.provider == "openai" and block.kind == block.data.get("type") == "web_search_call":
            return
    raise ValueError(f"ChatGPT subscription does not support {type(block).__name__} in {role} history.")


def build_chatgpt_payload(
    messages: list[Message],
    tools: list[Tool[Any]],
    system: str,
    model: ModelSpec,
    *,
    reasoning_effort: str | None = None,
    reasoning_summary: str = "auto",
) -> dict[str, Any]:
    """Declare current tools at input position zero, then replay complete history.

    Keeping this position stable makes unchanged tool sets reusable across turns.
    Selector or schema changes replace this declaration in place; old calls remain
    in history, but removed tools are no longer declared. No tool-loading state is
    held in the transport, so forks, restarts, and concurrent contexts are isolated.
    """
    if tools and Capability.tool_use not in model.capabilities:
        raise ValueError(f"Tool use is not enabled for {model.id}.")
    if reasoning_effort is not None and Capability.reasoning not in model.capabilities:
        raise ValueError(f"Reasoning is not enabled for {model.id}.")
    if reasoning_effort is not None and reasoning_effort not in supported_reasoning_efforts(model):
        available = ", ".join(supported_reasoning_efforts(model)) or "none"
        raise ValueError(f"Unsupported reasoning effort {reasoning_effort!r} for {model.id}. Available: {available}.")
    if len({tool.name for tool in tools}) != len(tools):
        raise ValueError("ChatGPT function names must be unique.")
    for message in messages:
        if message.role not in ("user", "assistant", "system"):
            raise ValueError(f"Unsupported message role: {message.role}")
        for block in message.content:
            validate_block(block, message.role, model)
    instructions, items = convert_messages(messages, system)
    for item in items:
        if item.get("role") == "system":
            item["role"] = "developer"
    if tools:
        items.insert(0, {"type": "additional_tools", "role": "developer", "tools": convert_tools(tools)})
    payload: dict[str, Any] = {
        "model": model.id,
        "input": items,
        "instructions": instructions,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if Capability.reasoning in model.capabilities:
        payload["reasoning"] = {"summary": reasoning_summary}
        if reasoning_effort is not None:
            payload["reasoning"]["effort"] = reasoning_effort
    return payload


@dataclass(slots=True)
class _Catalog:
    known: dict[str, ModelSpec]
    entries: tuple[ChatGPTModel, ...] = ()
    models: ModelRegistry = field(default_factory=ModelRegistry)
    loaded: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatGPTTransport:
    """A completion transport pinned to one authenticated account session.

    The caller owns the HTTP and account sessions. with_model() shares both and
    the account catalog. with_account() starts a new catalog without changing
    existing transports. No API-key, embedding, or realtime interface is exposed.
    """

    display_name: ClassVar[str] = "ChatGPT"
    env_vars: ClassVar[tuple[str, ...]] = ()
    auto_select: ClassVar[bool] = False
    supports_temperature: ClassVar[bool] = False
    supports_max_tokens: ClassVar[bool] = False
    supports_debug: ClassVar[bool] = False

    auth: ClassVar[ChatGPTAuth] = ChatGPTAuth()
    supports_thinking: ClassVar[bool] = True
    default_reasoning_effort: ClassVar[str] = "medium"
    supported_thinking_levels = staticmethod(supported_reasoning_efforts)

    name = "ChatGPT"

    def __init__(
        self,
        *,
        account: ChatGPTSession,
        session: aiohttp.ClientSession,
        model: str | None = None,
        model_specs: Iterable[ModelSpec] = (),
        base_url: str | None = None,
        image_base_url: str = CODEX_BASE_URL,
        max_attempts: int = 3,
        retry_base_delay: float = 2.0,
        reasoning_effort: str | None = None,
        reasoning_summary: Literal["auto", "concise", "detailed"] = "auto",
    ) -> None:
        self._executor = ChatGPTResponses(account, session, base_url, max_attempts, retry_base_delay)
        self.image_base_url = image_base_url
        self._model_id = model
        known = {spec.id: spec for spec in model_specs}
        self._catalog = _Catalog(known)
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.last_usage = Usage(0, 0)

    @property
    def account(self) -> ChatGPTSession:
        return self._executor.account

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._executor.http

    @property
    def models(self) -> ModelRegistry:
        """Visible server models followed by native image presets; empty until fetched."""
        return self._catalog.models

    @property
    def catalog(self) -> tuple[ChatGPTModel, ...]:
        """Server entries and native image presets, with visibility and display names."""
        return self._catalog.entries

    @property
    def model(self) -> ModelSpec | None:
        return self.models.get(self._model_id) if self._model_id is not None else None

    async def fetch_models(self) -> None:
        """Refresh this account's catalog. Failure invalidates it and propagates."""
        async with self._catalog.lock:
            await self._fetch_models()

    async def _fetch_models(self) -> None:
        self._catalog.loaded = False
        self._catalog.entries = ()
        self.models.clear()
        body = await self._executor.fetch_models()
        entries = parse_models(body, self._catalog.known)
        if self.account.supports_images:
            image_models = tuple(m for m in CHATGPT_IMAGE_MODELS.values() if isinstance(m, ChatGPTImageModel))
            entries = tuple(e for e in entries if e.id not in CHATGPT_IMAGE_MODELS) + image_models
        self._catalog.entries = entries
        for entry in entries:
            if entry.visibility == "list":
                self.models[entry.id] = entry
        self._catalog.loaded = True
        if self._model_id is None and self.models:
            self._model_id = self.models.first().id

    @staticmethod
    def format_error(error: Exception) -> str:
        """Return safe account recovery text for interactive applications."""
        from .cli import error_message

        message = error_message(error)
        if isinstance(error, ResponsesAPIError):
            if error.recovery.value == "quota":
                message += " Open ChatGPT Usage: https://chatgpt.com/settings/usage."
            elif error.recovery.value in {"permission", "authentication"}:
                message += " Check the selected account and plan permission in Configure ChatGPT."
        return message

    @classmethod
    def open_for_app(
        cls,
        *,
        session: aiohttp.ClientSession,
        client_id: str | None = None,
        account: str | None = None,
        credentials: str | None = None,
    ) -> AbstractAsyncContextManager[ChatGPTTransport]:
        """Open an account session for an application that owns transport lifetimes."""
        from pathlib import Path

        from .accounts import open_chatgpt
        from .credential_stores import FileCredentialStore

        return open_chatgpt(
            FileCredentialStore(Path(credentials) if credentials else None),
            session,
            client_id=account if account is not None else client_id,
        )

    def with_model(self, model: str | ModelSpec) -> ChatGPTTransport:
        """Select a visible catalog model while sharing the current account session."""
        model_id = model.id if isinstance(model, ModelSpec) else model
        if not self._catalog.loaded or model_id not in self.models.keys():
            raise ValueError("Fetch the account catalog and select a visible ChatGPT model.")
        clone = self.with_account(self.account, model=model_id)
        clone._catalog = self._catalog
        return clone

    @staticmethod
    def select_model(transport: ChatGPTTransport, model: ModelSpec) -> ChatGPTTransport:
        """Reset an incompatible reasoning setting when an application changes models."""
        selected = transport.with_model(model)
        if selected.reasoning_effort is not None and selected.reasoning_effort not in supported_reasoning_efforts(
            model
        ):
            selected.reasoning_effort = None
        return selected

    def with_account(self, account: ChatGPTSession, *, model: str | None = None) -> ChatGPTTransport:
        """Create a separately fetched catalog for another selected account."""
        return type(self)(
            account=account,
            session=self.session,
            model=model,
            model_specs=self._catalog.known.values(),
            base_url=self._executor.base_url,
            image_base_url=self.image_base_url,
            max_attempts=self._executor.max_attempts,
            retry_base_delay=self._executor.retry_base_delay,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
        )

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        """Build a request after fetching and selecting a visible account model."""
        selected = self.model
        if selected is None:
            raise ValueError("Select a visible model from the current ChatGPT account catalog.")
        if Capability.image_generation in selected.capabilities:
            raise ValueError("Use generate_images() with a ChatGPT image model.")
        payload = build_chatgpt_payload(
            messages,
            tools,
            system,
            selected,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
        )
        if self.account.supports_images:
            # Native Codex uses top-level function declarations, unlike plan sharing.
            if tools:
                payload["input"] = payload["input"][1:]
                payload["tools"] = convert_tools(tools)
            payload["tool_choice"] = "auto"
        return payload

    def image_transport(self, model: str | ModelSpec = "gpt-image-2") -> ChatGPTImageTransport:
        """Create a standalone generator sharing this account and HTTP session."""
        return ChatGPTImageTransport(
            account=self.account, session=self.session, model=model, base_url=self.image_base_url
        )

    async def generate_images(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        """Generate with the selected image model, or the default GPT Image model."""
        selected = self.model
        model_id = (
            model
            if model is not None
            else (
                selected.id
                if selected is not None and Capability.image_generation in selected.capabilities
                else "gpt-image-2"
            )
        )
        return await self.image_transport(model_id).generate_images(prompt, n=n)

    async def stream(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncGenerator[StreamEvent, None]:
        self.last_usage = Usage(0, 0)
        async with self._catalog.lock:
            if not self._catalog.loaded:
                await self._fetch_models()
        payload = self.build_payload(messages, tools, system)
        started = False
        async with aclosing(self._executor.stream(payload)) as events:
            async for event in events:
                if not started:
                    started = True
                    if not isinstance(event, IterationStart):
                        yield IterationStart(iteration=0)
                if isinstance(event, IterationEnd):
                    self.last_usage = event.usage
                yield event
