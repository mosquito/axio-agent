"""Responses diagnostics and recovery without an HTTP client dependency."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from axio.exceptions import StreamError
from axio.retry import is_retryable


class Recovery(StrEnum):
    """The action indicated by a provider failure, before considering replay safety."""

    retry = "retry"
    quota = "quota"
    authentication = "authentication"
    permission = "permission"
    invalid_request = "invalid_request"
    unknown = "unknown"


_RECOVERY = {
    "subscription_sharing_usage_limit_exceeded": Recovery.quota,
    "insufficient_quota": Recovery.quota,
    "subscription_sharing_usage_unavailable": Recovery.retry,
    "subscription_sharing_user_unavailable": Recovery.retry,
    "subscription_sharing_user_not_eligible": Recovery.permission,
    "subscription_sharing_unsupported_capability": Recovery.invalid_request,
    "subscription_sharing_route_not_supported": Recovery.invalid_request,
    "subscription_sharing_invalid_user": Recovery.authentication,
    "chatpass_v2_scope_not_authorized": Recovery.permission,
    "chatpass_v2_invalid_authorization_context": Recovery.permission,
}


def request_id(headers: object) -> str | None:
    """Read the HTTP request ID without retaining authorization headers."""
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).lower() == "x-request-id" and isinstance(value, str):
                return value
    return None


class ResponsesAPIError(StreamError):
    """A provider error with its original diagnostic body and transport metadata.

    ``status`` is the actual HTTP status, including 200 for an SSE failure.
    ``body`` retains JSON values without normalizing their shape. ``raw_body``
    retains HTTP text when available. The exception text contains the diagnostic message.
    Authentication errors alone do not prove revocation or authorize token removal.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
        body: Any = None,
        raw_body: str | None = None,
        event_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.param = param
        self.status = status
        self.request_id = request_id
        self.body = body
        self.raw_body = raw_body
        self.event_type = event_type

    @property
    def recovery(self) -> Recovery:
        if self.code in _RECOVERY:
            return _RECOVERY[self.code]
        if self.status == 401:
            return Recovery.authentication
        if self.status == 403:
            return Recovery.permission
        if self.status in (400, 404, 405, 422):
            return Recovery.invalid_request
        if self.status is not None and is_retryable(self.status):
            return Recovery.retry
        return Recovery.unknown

    @property
    def retryable(self) -> bool:
        """Whether recovery permits retry; callers must also prevent output replay."""
        return self.recovery is Recovery.retry

    @classmethod
    def from_body(
        cls,
        body: Any,
        *,
        status: int | None = None,
        request_id: str | None = None,
        event_type: str | None = None,
        raw_body: str | None = None,
    ) -> ResponsesAPIError:
        envelope = body if isinstance(body, dict) else {}
        response = envelope.get("response")
        container = response if isinstance(response, dict) else envelope
        error = container.get("error")
        details = error if isinstance(error, dict) else container
        message = details.get("message")
        if not isinstance(message, str) or not message:
            detail = envelope.get("detail")
            message = detail if isinstance(detail, str) else body if isinstance(body, str) else "Unknown error"
        code = details.get("code")
        param = details.get("param")
        body_request_id = envelope.get("request_id")
        return cls(
            f"Responses API error{f' {status}' if status is not None else ''}: {message}",
            code=code if isinstance(code, str) else None,
            param=param if isinstance(param, str) else None,
            status=status,
            request_id=request_id or (body_request_id if isinstance(body_request_id, str) else None),
            body=body,
            raw_body=raw_body,
            event_type=event_type,
        )

    @classmethod
    def from_http(cls, text: str, *, status: int, headers: object = None) -> ResponsesAPIError:
        try:
            body = json.loads(text)
        except ValueError:
            body = text
        return cls.from_body(body, status=status, request_id=request_id(headers), raw_body=text)


class ResponsesInterruptedError(ResponsesAPIError):
    """The stream ended before a terminal response event arrived."""


class ResponsesIncompleteError(ResponsesAPIError):
    """A terminal response did not complete; inspect its body for the reason."""
