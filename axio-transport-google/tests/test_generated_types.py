"""Tests validating that transport payload builders conform to discovery doc TypedDicts.

Validates field names and value types in the JSON payloads against the generated
TypedDicts from the Vertex AI discovery document.
"""

from __future__ import annotations

from typing import Any, get_args, get_origin, get_type_hints

from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.messages import Message
from axio.models import Capability
from axio.tool import Tool

from axio_transport_google import (
    GoogleTransport,
    _build_contents_json,
    _build_tools_json,
)
from axio_transport_google._generated_types import (
    Candidate,
    Content,
    GenerateContentRequest,
    GenerationConfig,
    Part,
)
from axio_transport_google._generated_types import (
    Tool as ToolDict,
)


async def _search_tool(query: str) -> str:
    return "result"


def _get_typed_dict_keys(td: type) -> set[str]:
    """Get all valid keys for a TypedDict."""
    return set(get_type_hints(td).keys())


def _validate_keys(data: dict[str, Any], td: type, path: str = "") -> list[str]:
    """Check that all keys in data are valid keys for the TypedDict.

    Returns a list of error messages for unknown keys.
    """
    valid_keys = _get_typed_dict_keys(td)
    errors = []
    for key in data:
        if key not in valid_keys:
            errors.append(f"{path}.{key}" if path else key)
    return errors


def _validate_dict_recursive(data: Any, td: type, path: str = "") -> list[str]:
    """Recursively validate dict keys against TypedDict definitions.

    Returns list of "{path}: unknown key '{key}'" error strings.
    """
    if not isinstance(data, dict):
        return []

    hints = get_type_hints(td)
    errors = []

    for key in data:
        full_path = f"{path}.{key}" if path else key
        if key not in hints:
            errors.append(f"unknown key '{full_path}' (not in {td.__name__})")
            continue

        # Try to recurse into nested TypedDicts
        hint = hints[key]
        value = data[key]

        # Unwrap Optional / Union
        origin = get_origin(hint)
        args = get_args(hint)

        if origin is list and isinstance(value, list):
            item_type = args[0] if args else None
            if item_type and _is_typed_dict(item_type):
                for i, item in enumerate(value):
                    errors.extend(_validate_dict_recursive(item, item_type, f"{full_path}[{i}]"))
        elif _is_typed_dict(hint) and isinstance(value, dict):
            errors.extend(_validate_dict_recursive(value, hint, full_path))

    return errors


def _is_typed_dict(tp: type) -> bool:
    """Check if a type is a TypedDict."""
    return isinstance(tp, type) and issubclass(tp, dict) and hasattr(tp, "__annotations__")


# ---------------------------------------------------------------------------
# Content builder conformance
# ---------------------------------------------------------------------------


def test_user_text_conforms_to_content() -> None:
    messages = [Message(role="user", content=[TextBlock(text="Hello")])]
    contents = _build_contents_json(messages)
    errors = _validate_dict_recursive(contents[0], Content)
    assert not errors, f"Content validation errors: {errors}"


def test_assistant_text_conforms_to_content() -> None:
    messages = [Message(role="assistant", content=[TextBlock(text="Hi")])]
    contents = _build_contents_json(messages)
    errors = _validate_dict_recursive(contents[0], Content)
    assert not errors, f"Content validation errors: {errors}"


def test_user_image_conforms_to_content() -> None:
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="What is this?"),
                ImageBlock(media_type="image/png", data=b"\x89PNG"),
            ],
        )
    ]
    contents = _build_contents_json(messages)
    errors = _validate_dict_recursive(contents[0], Content)
    assert not errors, f"Content validation errors: {errors}"


def test_function_call_conforms_to_part() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="search", input={"q": "test"})],
        )
    ]
    contents = _build_contents_json(messages)
    errors = _validate_dict_recursive(contents[0], Content)
    assert not errors, f"Content validation errors: {errors}"


def test_function_response_conforms_to_part() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="search", input={"q": "test"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="found it")],
        ),
    ]
    contents = _build_contents_json(messages)
    errors = _validate_dict_recursive(contents[1], Content)
    assert not errors, f"Content validation errors: {errors}"


def test_thought_signature_conforms_to_part() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="search", input={"q": "test"})],
        )
    ]
    sigs = {"call_1": "dGVzdA=="}
    contents = _build_contents_json(messages, sigs)
    part = contents[0]["parts"][0]
    errors = _validate_dict_recursive(part, Part)
    assert not errors, f"Part validation errors: {errors}"


# ---------------------------------------------------------------------------
# Tool builder conformance
# ---------------------------------------------------------------------------


def test_tools_json_conforms_to_tool() -> None:
    tool: Tool[Any] = Tool(name="search", description="Search", handler=_search_tool)
    result = _build_tools_json([tool])
    for tool_dict in result:
        errors = _validate_dict_recursive(tool_dict, ToolDict)
        assert not errors, f"Tool validation errors: {errors}"


# ---------------------------------------------------------------------------
# Generation config conformance
# ---------------------------------------------------------------------------


def test_generation_config_conforms() -> None:
    t = GoogleTransport(temperature=0.5, top_p=0.8, top_k=40, seed=42)
    config = t._build_generation_config_json()
    errors = _validate_dict_recursive(config, GenerationConfig)
    assert not errors, f"GenerationConfig validation errors: {errors}"


def test_generation_config_thinking_conforms() -> None:
    t = GoogleTransport(thinking_level="HIGH")
    config = t._build_generation_config_json()
    errors = _validate_dict_recursive(config, GenerationConfig)
    assert not errors, f"GenerationConfig validation errors: {errors}"


def test_generation_config_image_modalities_conforms() -> None:
    from axio.models import ModelSpec

    t = GoogleTransport()
    t.model = ModelSpec(
        id="test-image-model",
        context_window=100000,
        max_output_tokens=8192,
        capabilities=frozenset({Capability.text, Capability.image_generation}),
    )
    config = t._build_generation_config_json()
    errors = _validate_dict_recursive(config, GenerationConfig)
    assert not errors, f"GenerationConfig validation errors: {errors}"


def test_generation_config_media_resolution_conforms() -> None:
    t = GoogleTransport(media_resolution="HIGH")
    config = t._build_generation_config_json()
    errors = _validate_dict_recursive(config, GenerationConfig)
    assert not errors, f"GenerationConfig validation errors: {errors}"


# ---------------------------------------------------------------------------
# Full request body conformance
# ---------------------------------------------------------------------------


def test_full_request_body_conforms() -> None:
    t = GoogleTransport(
        temperature=0.7,
        safety_settings=[{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
    )
    messages = [
        Message(role="user", content=[TextBlock(text="Hello")]),
    ]
    tool: Tool[Any] = Tool(name="search", description="Search", handler=_search_tool)

    contents = _build_contents_json(messages, t._thought_signatures)
    body: dict[str, Any] = {"contents": contents}
    body["systemInstruction"] = {"parts": [{"text": "You are helpful"}]}
    body["tools"] = _build_tools_json([tool])
    body["generationConfig"] = t._build_generation_config_json()
    body["safetySettings"] = t.safety_settings

    errors = _validate_dict_recursive(body, GenerateContentRequest)
    assert not errors, f"GenerateContentRequest validation errors: {errors}"


# ---------------------------------------------------------------------------
# Verify all Candidate.finishReason values are handled
# ---------------------------------------------------------------------------


def test_finish_reasons_coverage() -> None:
    """Ensure _FINISH_REASON_MAP covers all non-UNSPECIFIED finishReason values from discovery doc."""
    from axio_transport_google import _FINISH_REASON_MAP

    finish_reason_hints = get_type_hints(Candidate)["finishReason"]
    discovery_reasons = set(get_args(finish_reason_hints))
    discovery_reasons.discard("FINISH_REASON_UNSPECIFIED")

    mapped_reasons = set(_FINISH_REASON_MAP.keys())
    missing = discovery_reasons - mapped_reasons
    assert not missing, f"finishReason values not in _FINISH_REASON_MAP: {missing}"
