"""Preserve provider diagnostics and distinguish recovery from HTTP status."""

from __future__ import annotations

import json
from typing import Any

import pytest
from axio.exceptions import StreamError
from axio_sse import Event

from axio_responses import Recovery, Responses, ResponsesAPIError, ResponsesInterruptedError


@pytest.mark.parametrize(
    "code,status,recovery",
    [
        ("subscription_sharing_usage_limit_exceeded", 429, Recovery.quota),
        ("insufficient_quota", 429, Recovery.quota),
        ("subscription_sharing_usage_unavailable", 503, Recovery.retry),
        ("subscription_sharing_user_unavailable", 503, Recovery.retry),
        ("subscription_sharing_user_not_eligible", 403, Recovery.permission),
        ("subscription_sharing_unsupported_capability", 400, Recovery.invalid_request),
        ("subscription_sharing_route_not_supported", 403, Recovery.invalid_request),
        ("subscription_sharing_invalid_user", 401, Recovery.authentication),
        ("chatpass_v2_scope_not_authorized", 403, Recovery.permission),
        ("chatpass_v2_invalid_authorization_context", 403, Recovery.permission),
        (None, 401, Recovery.authentication),
        (None, 403, Recovery.permission),
        (None, 400, Recovery.invalid_request),
        (None, 429, Recovery.retry),
        (None, 503, Recovery.retry),
        (None, 200, Recovery.unknown),
    ],
)
def test_recovery(code: str | None, status: int, recovery: Recovery) -> None:
    body = {"error": {"message": "diagnostic", "code": code, "param": "tools"}, "extra": [1, 2]}
    raw = json.dumps(body)
    error = ResponsesAPIError.from_http(raw, status=status, headers={"X-Request-ID": "req-test"})
    assert isinstance(error, StreamError)
    assert error.code == code
    assert error.param == "tools"
    assert error.status == status
    assert error.request_id == "req-test"
    assert error.body == body
    assert error.raw_body == raw
    assert error.recovery is recovery
    assert error.retryable == (recovery is Recovery.retry)


@pytest.mark.parametrize("body", [{"detail": "routing unavailable"}, {"detail": ["region"]}, [], None, 5, "bad"])
def test_nonstandard_json_retains_shape(body: Any) -> None:
    error = ResponsesAPIError.from_http(json.dumps(body), status=503)
    assert error.body == body
    assert error.code is None
    assert error.param is None
    assert error.retryable


def test_plain_text_is_not_parsed_as_a_code() -> None:
    text = "subscription_sharing_usage_limit_exceeded"
    error = ResponsesAPIError.from_http(text, status=503)
    assert error.body == text
    assert error.raw_body == text
    assert error.code is None
    assert error.retryable


@pytest.mark.parametrize("kind", ["response.failed", "error"])
def test_stream_failures_preserve_error_and_envelope(kind: str) -> None:
    details = {"message": "limit", "code": "subscription_sharing_usage_limit_exceeded", "param": "model"}
    body: dict[str, Any] = {"type": kind, "request_id": "body-id", "sequence_number": 2}
    if kind == "response.failed":
        body["response"] = {"id": "resp-test", "error": details, "status": "failed"}
    else:
        body.update(details)
    reader = Responses(status=200, request_id="http-id")
    list(reader.read(Event(data=json.dumps({"type": "response.output_text.delta", "delta": "partial"}))))
    with pytest.raises(ResponsesAPIError) as raised:
        list(reader.read(Event(data=json.dumps(body))))
    error = raised.value
    assert error.code == details["code"]
    assert error.param == "model"
    assert error.status == 200
    assert error.request_id == "http-id"
    assert error.body == body
    assert error.event_type == kind
    assert not error.retryable


def test_malformed_fields_remain_available_in_body() -> None:
    body = {"error": {"code": ["wrong"], "param": 1, "message": {}}, "request_id": "req-body"}
    error = ResponsesAPIError.from_body(body)
    assert error.body == body
    assert error.code is None and error.param is None
    assert error.request_id == "req-body"


def test_refusal_delta_does_not_prove_completion() -> None:
    reader = Responses(status=200, request_id="req-cut")
    list(reader.read(Event(data=json.dumps({"type": "response.refusal.delta", "delta": "Cannot"}))))
    with pytest.raises(ResponsesInterruptedError) as raised:
        reader.finished()
    assert raised.value.request_id == "req-cut"
    assert raised.value.status == 200


def test_refusal_cannot_hide_failed_completion() -> None:
    reader = Responses()
    list(reader.read(Event(data=json.dumps({"type": "response.refusal.delta", "delta": "Cannot"}))))
    body = {"type": "response.completed", "response": {"status": "failed", "error": {"code": "failure"}}}
    with pytest.raises(ResponsesAPIError) as raised:
        list(reader.read(Event(data=json.dumps(body))))
    assert raised.value.code == "failure"
    assert raised.value.body == body
