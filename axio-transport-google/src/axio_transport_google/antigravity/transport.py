"""Cloud Code envelopes around the shared Gemini request and stream formats."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import aiohttp
from axio.events import IterationEnd, ProviderEvent, StreamEvent
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.retry import is_retryable, retry_delay
from axio.tool import Tool
from axio.types import StopReason, Usage
from axio_sse import Payload, payloads

from axio_transport_google import (
    _BLOCKED,
    GenerateContentChunk,
    GoogleTransport,
    _build_contents_json,
    _build_tools_json,
    _Turn,
    _usage,
    valid_thinking_levels,
)

from .app_auth import AntigravityAuth
from .oauth import AntigravityOAuth
from .session import AntigravityFileCredentials, AntigravitySession, resolve_account

BASE_URL = "https://daily-cloudcode-pa.googleapis.com"
USER_AGENT = "antigravity/cli/1.1.13 (aidev_client; auth_method=consumer)"


class AntigravityAPIError(StreamError):
    """A classified failure without tokens, account details, or request contents."""

    def __init__(self, status: int, code: str = "", *, retry_after: float | None = None) -> None:
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.category = {
            401: "authentication",
            403: "permission",
            429: "quota",
        }.get(status, "transient" if status >= 500 else "request")
        help_text = {
            "authentication": "Run axio-antigravity login to renew authorization.",
            "permission": "Check Antigravity access for the selected Google account.",
            "quota": "Wait for the account quota to reset, then retry.",
        }.get(self.category, "Retry later." if status >= 500 else "Check the request and selected model.")
        super().__init__(f"Antigravity HTTP {status} ({self.category}; {code or 'unknown'}). {help_text}")


def _limit(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default


def _model(model_id: str, data: dict[str, Any]) -> ModelSpec:
    caps = {Capability.text, Capability.tool_use}
    for key, cap in (
        ("supportsImages", Capability.vision),
        ("supportsVideo", Capability.video),
        ("supportsThinking", Capability.reasoning),
    ):
        if data.get(key) is True:
            caps.add(cap)
    mime_types = data.get("supportedMimeTypes", {})
    if isinstance(mime_types, dict) and any(k.startswith("audio/") and v is True for k, v in mime_types.items()):
        caps.add(Capability.audio)
    return ModelSpec(
        model_id, frozenset(caps), _limit(data, "maxOutputTokens", 8192), _limit(data, "maxTokens", 128000)
    )


@dataclass(slots=True)
class AntigravityTransport:
    """Gemini completion through one pinned Google OAuth account.

    The caller owns the aiohttp session. Model clones share the account session,
    request session ID, and Gemini signature state. No API key is used.
    """

    display_name: ClassVar[str] = "Antigravity"
    env_vars: ClassVar[tuple[str, ...]] = ()
    auto_select: ClassVar[bool] = False
    supports_debug: ClassVar[bool] = False

    auth: ClassVar[AntigravityAuth] = AntigravityAuth()

    name: str = "Antigravity (Gemini)"
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    account: AntigravitySession | None = field(default=None, repr=False, compare=False)
    model: ModelSpec = field(default_factory=lambda: ModelSpec(""))
    models: ModelRegistry = field(default_factory=ModelRegistry)
    catalog: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    base_url: str = BASE_URL
    project: str = field(default="", repr=False)
    max_retries: int = 3
    retry_base_delay: float = 2.0
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_level: str | None = None
    thinking_budget: int | None = None
    last_usage: Usage | None = field(default=None, repr=False)
    _session_id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)
    _reader: GoogleTransport = field(default_factory=GoogleTransport, repr=False)

    @classmethod
    @asynccontextmanager
    async def open_for_app(
        cls,
        *,
        session: aiohttp.ClientSession,
        account_id: str | None = None,
        credentials: str | None = None,
        account: str | None = None,
    ) -> AsyncIterator[AntigravityTransport]:
        """Open a pinned account using settings that contain no tokens."""
        store = AntigravityFileCredentials(Path(credentials) if credentials else None)
        selected = await resolve_account(store, account if account is not None else account_id)
        transport = cls(session=session, account=AntigravitySession(store, AntigravityOAuth(session), selected))
        await transport.fetch_models()
        yield transport

    async def _account(self) -> AntigravitySession:
        if self.account is None:
            if self.session is None:
                raise ValueError("AntigravityTransport requires an aiohttp session.")
            store = AntigravityFileCredentials()
            self.account = AntigravitySession(store, AntigravityOAuth(self.session), await resolve_account(store))
        return self.account

    async def _payloads(
        self, method: str, body: dict[str, Any], *, streaming: bool = False
    ) -> AsyncGenerator[Payload, None]:
        if self.session is None:
            raise ValueError("AntigravityTransport requires an aiohttp session.")
        account = await self._account()
        token = await account.access_token()
        refreshed = False
        sent = False
        for attempt in range(1, max(1, self.max_retries) + 1):
            delay = 0.0
            try:
                url = f"{self.base_url}/v1internal:{method}" + ("?alt=sse" if streaming else "")
                async with self.session.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=None, connect=30, sock_read=180),
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        try:
                            error = await response.json()
                        except (ValueError, aiohttp.ContentTypeError):
                            error = {}
                        detail = error.get("error", {}) if isinstance(error, dict) else {}
                        code = detail.get("status", "") if isinstance(detail, dict) else ""
                        # Only the protocol status is safe for diagnostics; messages can echo inputs.
                        code = code if isinstance(code, str) and code.isupper() and len(code) < 80 else ""
                        delay = retry_delay(response, attempt, base=self.retry_base_delay)
                        if response.status == 401 and not refreshed and attempt < self.max_retries:
                            token = await account.access_token(rejected=token)
                            refreshed = True
                            delay = 0
                        elif not (is_retryable(response.status) and attempt < self.max_retries):
                            raise AntigravityAPIError(response.status, code, retry_after=delay)
                    elif streaming:
                        async for payload in payloads(response.content.iter_any()):
                            sent = True
                            yield payload
                        return
                    else:
                        data = await response.json()
                        if not isinstance(data, dict):
                            raise StreamError("Antigravity returned a non-object response.")
                        yield Payload(data)
                        return
            except (aiohttp.ClientError, TimeoutError) as error:
                if sent or attempt >= self.max_retries:
                    raise StreamError("Antigravity connection failed or the stream was interrupted.") from error
                delay = retry_delay(None, attempt, base=self.retry_base_delay)
            except ValueError as error:
                raise StreamError("Antigravity returned invalid JSON.") from error
            await asyncio.sleep(delay)

    async def _json(self, method: str, body: dict[str, Any]) -> Payload:
        async with aclosing(self._payloads(method, body)) as responses:
            async for data in responses:
                return data
        raise StreamError("Antigravity returned no response.")

    async def fetch_models(self) -> ModelRegistry:
        profile = await self._json("loadCodeAssist", {"metadata": {"ideType": "ANTIGRAVITY"}})
        project = profile.get("cloudaicompanionProject")
        if isinstance(project, dict):
            project = project.get("id")
        if not isinstance(project, str) or not project:
            raise StreamError("Complete account setup in Antigravity first: no companion project is available.")
        data = await self._json("fetchAvailableModels", {"project": project})
        raw_models = data.get("models")
        if not isinstance(raw_models, dict):
            raise StreamError("Antigravity returned no model catalogue.")
        catalog = {
            key: {k: v for k, v in value.items() if k != "modelExperiments"}
            for key, value in raw_models.items()
            if key.startswith("gemini-") and "image" not in key and isinstance(value, dict)
        }
        models = ModelRegistry(_model(key, value) for key, value in catalog.items())
        if not models:
            raise StreamError("No Gemini completion models are available to this Antigravity account.")
        if self.model.id:
            if self.model.id not in models:
                raise StreamError(f"The selected model {self.model.id!r} is unavailable for this account.")
            selected = models[self.model.id]
        else:
            default = data.get("defaultAgentModelId")
            selected = models[default] if isinstance(default, str) and default in models else models.first()
        self.project, self.catalog, self.models, self.model = project, catalog, models, selected
        return models

    def with_model(self, model: ModelSpec) -> AntigravityTransport:
        if model.id not in self.models:
            raise ValueError(f"Unknown Antigravity model: {model.id}")
        return replace(self, model=self.models[model.id], thinking_level=None, thinking_budget=None, last_usage=None)

    def get_thinking_options(self) -> tuple[str, ...] | None:
        return valid_thinking_levels(self.model.id)

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if self.max_output_tokens is not None:
            config["maxOutputTokens"] = self.max_output_tokens
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if Capability.reasoning in self.model.capabilities:
            thinking: dict[str, Any] = {"includeThoughts": True}
            if self.thinking_level:
                levels = self.get_thinking_options()
                if levels is None or self.thinking_level.upper() not in levels:
                    raise ValueError("Unsupported thinking level for the selected Antigravity model.")
                thinking["thinkingLevel"] = self.thinking_level.upper()
            elif self.thinking_budget is not None:
                thinking["thinkingBudget"] = self.thinking_budget
            config["thinkingConfig"] = thinking
        request: dict[str, Any] = {
            "contents": _build_contents_json(messages, self._reader._thought_signatures),
            "sessionId": self._session_id,
            "generationConfig": config,
        }
        if system:
            request["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            request["tools"] = _build_tools_json(tools)
        return {
            "project": self.project,
            "model": self.model.id,
            "request": request,
            "userAgent": "antigravity",
            "requestType": "agent",
            "requestId": f"agent/{uuid.uuid4()}",
        }

    async def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        if not self.project or not self.model.id:
            await self.fetch_models()
        self.last_usage = None
        self._reader._streams += 1
        turn = _Turn(seq=self._reader._streams)
        responses = self._payloads(
            "streamGenerateContent", self.build_payload(messages, tools, system), streaming=True
        )
        async with aclosing(responses):
            async for envelope in responses:
                if "error" in envelope:
                    raise StreamError("Antigravity returned an error inside the stream.")
                response = envelope.get("response")
                if not isinstance(response, dict):
                    raise StreamError("Antigravity returned an invalid stream envelope.")
                payload = Payload(response)
                payload.setdefault("modelVersion", self.model.id)
                for event in self._reader._chunk_events(GenerateContentChunk.read(payload), turn):
                    self.last_usage = turn.usage
                    yield event
                extra = {k: v for k, v in envelope.items() if k != "response"}
                if extra:
                    yield ProviderEvent(provider="antigravity", kind="envelope", data=extra)
        if not turn.finished:
            raise StreamError("Antigravity stream ended without a finishReason.")
        reason = turn.stop_reason
        if turn.has_tool_calls and reason not in _BLOCKED:
            reason = StopReason.tool_use
        if reason is StopReason.error:
            raise StreamError(f"Antigravity stopped with {turn.reason or 'an error'}.")
        usage = _usage(turn.counts, final=True) if turn.counts is not None else turn.usage
        self.last_usage = usage
        yield IterationEnd(iteration=0, stop_reason=reason, usage=usage)
