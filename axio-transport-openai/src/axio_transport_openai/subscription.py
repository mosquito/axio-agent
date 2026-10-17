"""Authenticated Responses execution and recovery for ChatGPT plan requests."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import aiohttp
from axio.events import StreamEvent
from axio.retry import retry_delay
from axio_responses import (
    Recovery,
    Responses,
    ResponsesAPIError,
    ResponsesIncompleteError,
    ResponsesInterruptedError,
    request_id,
)

from .session import ChatGPTSession


@dataclass(frozen=True, slots=True)
class ChatGPTResponses:
    """Send subscription request bodies with a pinned account session.

    The caller builds a subscription-compatible body. This executor owns HTTP,
    SSE completion checks, and bounded recovery. It never changes credentials
    after an inference error or falls back to another account or billing path.
    ``max_attempts`` includes the first request. Any emitted event prevents replay.
    The caller owns both sessions and closes them after stream consumers finish.
    """

    account: ChatGPTSession
    http: aiohttp.ClientSession
    base_url: str = "https://api.openai.com/v1"
    max_attempts: int = 3
    retry_base_delay: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        if not math.isfinite(self.retry_base_delay) or self.retry_base_delay < 0:
            raise ValueError("retry_base_delay must be finite and nonnegative.")

    async def fetch_models(self) -> Any:
        """Read the selected account's catalog without an inference quota gate.

        A model picker remains available when inference is paused. Authentication
        still requires the selected account's plan permission. No static fallback
        or credential replacement is attempted after an admission error.
        """
        for attempt in range(1, self.max_attempts + 1):
            token = await self.account.access_token()
            response: aiohttp.ClientResponse | None = None
            try:
                async with self.http.get(
                    f"{self.base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {token}"},
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        raise ResponsesAPIError.from_http(
                            await response.text(), status=response.status, headers=response.headers
                        )
                    try:
                        return await response.json()
                    except (ValueError, aiohttp.ContentTypeError) as exc:
                        raise ResponsesAPIError(
                            "ChatGPT model catalog is not JSON.",
                            status=response.status,
                            request_id=request_id(response.headers),
                            raw_body=await response.text(),
                        ) from exc
            except ResponsesAPIError as exc:
                if not exc.retryable or attempt == self.max_attempts:
                    raise
            except (aiohttp.ClientError, TimeoutError) as exc:
                if attempt == self.max_attempts:
                    raise ResponsesInterruptedError(
                        "ChatGPT model catalog request failed.",
                        status=response.status if response is not None else None,
                        request_id=request_id(response.headers) if response is not None else None,
                    ) from exc
            await asyncio.sleep(retry_delay(response, attempt, base=self.retry_base_delay))
        raise AssertionError("A positive attempt count must return or raise.")

    async def stream(self, payload: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        """Read through completion, or raise a structured error without replaying output."""
        for attempt in range(1, self.max_attempts + 1):
            if self.account.plan_usage_error is not None:
                raise self.account.plan_usage_error
            token = await self.account.access_token()
            headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
            sent = False
            response: aiohttp.ClientResponse | None = None
            try:
                async with self.http.post(
                    f"{self.base_url.rstrip('/')}/responses",
                    json=payload,
                    headers=headers,
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        raise ResponsesAPIError.from_http(
                            await response.text(), status=response.status, headers=response.headers
                        )
                    turn = Responses(status=response.status, request_id=request_id(response.headers))
                    async for event in turn.over(response.content.iter_any(), until="[DONE]"):
                        sent = True
                        yield event
                    end = turn.finished()
                    terminal = turn.terminal_event
                    assert terminal is not None
                    if (
                        terminal.string("type") != "response.completed"
                        or terminal.obj("response").string("status") != "completed"
                    ):
                        raise ResponsesIncompleteError(
                            "ChatGPT response did not complete.",
                            status=response.status,
                            request_id=request_id(response.headers),
                            body=dict(terminal),
                            event_type=terminal.string("type"),
                        )
                    yield end
                    return
            except ResponsesAPIError as exc:
                if exc.recovery is Recovery.quota:
                    self.account.plan_usage_error = exc
                if sent or not exc.retryable or attempt == self.max_attempts:
                    raise
            except (aiohttp.ClientError, TimeoutError) as exc:
                error = ResponsesInterruptedError(
                    "ChatGPT connection failed before response completion.",
                    status=response.status if response is not None else None,
                    request_id=request_id(response.headers) if response is not None else None,
                )
                if sent or attempt == self.max_attempts:
                    raise error from exc
            await asyncio.sleep(retry_delay(response, attempt, base=self.retry_base_delay))
