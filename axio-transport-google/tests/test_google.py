"""Tests for Google GenAI transport — message conversion and tool handling."""

from __future__ import annotations

import base64
from typing import Any

from axio.blocks import AudioBlock, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock, VideoBlock
from axio.messages import Message
from axio.models import Capability, ModelRegistry
from axio.tool import Tool

from axio_transport_google import (
    GENAI_MODELS,
    GoogleTransport,
    _build_contents_json,
    _build_tools_json,
    _get_anthropic_models,
    _tool_name_from_id,
)


async def _dummy_tool(query: str) -> str:
    return "ok"


# ---------------------------------------------------------------------------
# _tool_name_from_id
# ---------------------------------------------------------------------------


def test_tool_name_from_id_found() -> None:
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="read_file", input={"filename": "a.txt"})]),
    ]
    assert _tool_name_from_id("c1", messages) == "read_file"


def test_tool_name_from_id_not_found() -> None:
    assert _tool_name_from_id("missing", []) == "unknown"


# ---------------------------------------------------------------------------
# _build_contents_json — basic user/assistant
# ---------------------------------------------------------------------------


def test_convert_user_text() -> None:
    messages = [Message(role="user", content=[TextBlock(text="Hello")])]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert any(p.get("text") == "Hello" for p in contents[0]["parts"])


def test_convert_assistant_text() -> None:
    messages = [Message(role="assistant", content=[TextBlock(text="Hi there")])]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "model"
    assert any(p.get("text") == "Hi there" for p in contents[0]["parts"])


# ---------------------------------------------------------------------------
# _build_contents_json — images, audio, and video in user messages
# ---------------------------------------------------------------------------


def test_convert_user_image() -> None:
    img_data = b"\x89PNG\r\n\x1a\nfake"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="What is this?"),
                ImageBlock(media_type="image/png", data=img_data),
            ],
        )
    ]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "What is this?"
    idata = parts[1]["inlineData"]
    assert idata["mimeType"] == "image/png"
    assert base64.b64decode(idata["data"]) == img_data


def test_convert_user_audio() -> None:
    audio_data = b"\xff\xfb\x90\x00fake-mp3"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="Transcribe this"),
                AudioBlock(media_type="audio/mp3", data=audio_data),
            ],
        )
    ]
    contents = _build_contents_json(messages)
    parts = contents[0]["parts"]
    assert len(parts) == 2
    idata = parts[1]["inlineData"]
    assert idata["mimeType"] == "audio/mp3"
    assert base64.b64decode(idata["data"]) == audio_data


def test_convert_user_video() -> None:
    video_data = b"\x00\x00\x00\x1cftypisom"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="Describe this video"),
                VideoBlock(media_type="video/mp4", data=video_data),
            ],
        )
    ]
    contents = _build_contents_json(messages)
    parts = contents[0]["parts"]
    assert len(parts) == 2
    idata = parts[1]["inlineData"]
    assert idata["mimeType"] == "video/mp4"
    assert base64.b64decode(idata["data"]) == video_data


# ---------------------------------------------------------------------------
# _build_contents_json — tool calls and results
# ---------------------------------------------------------------------------


def test_convert_assistant_tool_call() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="get_weather", input={"location": "Paris"})],
        )
    ]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "model"
    part = contents[0]["parts"][0]
    fc = part["functionCall"]
    assert fc["name"] == "get_weather"
    assert fc["args"] == {"location": "Paris"}
    assert fc["id"] == "call_1"


def test_convert_tool_result_text() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="get_weather", input={"location": "Paris"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="22°C sunny")],
        ),
    ]
    contents = _build_contents_json(messages)
    assert len(contents) == 2
    fr = contents[1]["parts"][0]["functionResponse"]
    assert fr["name"] == "get_weather"
    assert fr["id"] == "call_1"
    assert fr["response"]["result"] == "22°C sunny"


def test_convert_tool_result_with_image() -> None:
    img_data = b"\xff\xd8\xff\xe0fake-jpeg"
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="read_file", input={"filename": "photo.jpg"})],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[
                        TextBlock(text="Image file: photo.jpg"),
                        ImageBlock(media_type="image/jpeg", data=img_data),
                    ],
                )
            ],
        ),
    ]
    contents = _build_contents_json(messages)
    fr = contents[1]["parts"][0]["functionResponse"]
    assert fr["response"] == {"result": "Image file: photo.jpg"}
    # Media is a sibling inlineData part, not nested inside functionResponse
    media_part = contents[1]["parts"][1]
    assert media_part["inlineData"]["mimeType"] == "image/jpeg"
    assert base64.b64decode(media_part["inlineData"]["data"]) == img_data


def test_convert_tool_result_with_audio() -> None:
    audio_data = b"OggS\x00\x02fake-ogg"
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="read_file", input={"filename": "audio.ogg"})],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[
                        TextBlock(text="Audio file: audio.ogg"),
                        AudioBlock(media_type="audio/ogg", data=audio_data),
                    ],
                )
            ],
        ),
    ]
    contents = _build_contents_json(messages)
    parts = contents[1]["parts"]
    assert parts[0]["functionResponse"]["response"] == {"result": "Audio file: audio.ogg"}
    assert parts[1]["inlineData"]["mimeType"] == "audio/ogg"
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == audio_data


def test_convert_tool_result_error() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="shell", input={"command": "bad"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="command failed", is_error=True)],
        ),
    ]
    contents = _build_contents_json(messages)
    fr = contents[1]["parts"][0]["functionResponse"]
    assert fr["response"]["error"] == "command failed"


# ---------------------------------------------------------------------------
# _build_contents_json — thought signatures
# ---------------------------------------------------------------------------


def test_convert_thought_signature_roundtrip() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="read_file", input={"filename": "a.txt"})],
        )
    ]
    sigs = {"call_1": "dGVzdF9zaWduYXR1cmU="}
    contents = _build_contents_json(messages, sigs)
    part = contents[0]["parts"][0]
    assert part["thoughtSignature"] == "dGVzdF9zaWduYXR1cmU="


# ---------------------------------------------------------------------------
# _build_tools_json
# ---------------------------------------------------------------------------


def test_build_tools_json() -> None:
    tool: Tool[Any] = Tool(name="search", description="Search the web", handler=_dummy_tool)
    result = _build_tools_json([tool])
    assert len(result) == 1
    declarations = result[0]["functionDeclarations"]
    assert len(declarations) == 1
    assert declarations[0]["name"] == "search"
    assert declarations[0]["description"] == "Search the web"
    assert "query" in declarations[0]["parameters"].get("properties", {})


# ---------------------------------------------------------------------------
# GoogleTransport basics
# ---------------------------------------------------------------------------


def test_transport_defaults() -> None:
    t = GoogleTransport()
    assert t.model.id == "gemini-3.1-flash-lite-preview"
    assert t.name == "Google GenAI"


def test_transport_to_from_dict() -> None:
    t = GoogleTransport(api_key="test-key")
    d = t.to_dict()
    assert d["api_key"] == "test-key"
    t2 = GoogleTransport.from_dict(d)
    assert t2.api_key == "test-key"


def test_transport_vertexai_to_from_dict() -> None:
    t = GoogleTransport(vertexai=True, project="my-project", location="us-central1")
    d = t.to_dict()
    assert d["vertexai"] is True
    assert d["project"] == "my-project"
    assert d["location"] == "us-central1"
    t2 = GoogleTransport.from_dict(d)
    assert t2.vertexai is True
    assert t2.project == "my-project"
    assert t2.location == "us-central1"


def test_transport_non_vertexai_no_extra_fields(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = GoogleTransport(api_key="test-key")
    d = t.to_dict()
    assert "vertexai" not in d
    assert "project" not in d


def test_genai_models_registry() -> None:
    assert isinstance(GENAI_MODELS, ModelRegistry)
    assert "gemini-3-flash-preview" in GENAI_MODELS
    assert "gemini-3.1-pro-preview" in GENAI_MODELS
    assert "gemini-3.1-flash-lite-preview" in GENAI_MODELS


def test_vertexai_anthropic_models_registry() -> None:
    models = _get_anthropic_models()
    assert isinstance(models, ModelRegistry)
    assert "anthropic/claude-sonnet-4-6" in models
    assert "anthropic/claude-opus-4-6" in models
    assert "anthropic/claude-haiku-4-5" in models


# ---------------------------------------------------------------------------
# Image / video generation model capabilities
# ---------------------------------------------------------------------------


def test_nano_banana_model_has_image_generation() -> None:
    spec = GENAI_MODELS["gemini-3.1-flash-image-preview"]
    assert Capability.image_generation in spec.capabilities
    assert Capability.vision in spec.capabilities


def test_dedicated_gen_models_not_in_chat_registry() -> None:
    """Imagen/Veo use dedicated :predict / :predictLongRunning endpoints —
    must not be selectable as chat models."""
    for model_id in GENAI_MODELS.keys():
        assert not model_id.startswith("imagen-"), f"{model_id} should not be in GENAI_MODELS"
        assert not model_id.startswith("veo-"), f"{model_id} should not be in GENAI_MODELS"


# ---------------------------------------------------------------------------
# Config parameters round-trip
# ---------------------------------------------------------------------------


def test_config_params_to_from_dict() -> None:
    t = GoogleTransport(temperature=0.7, top_p=0.9, seed=42, thinking_budget=4096, service_tier="flex")
    d = t.to_dict()
    assert d["temperature"] == 0.7
    assert d["seed"] == 42
    assert d["thinking_budget"] == 4096
    assert d["service_tier"] == "flex"
    t2 = GoogleTransport.from_dict(d)
    assert t2.temperature == 0.7
    assert t2.top_p == 0.9
    assert t2.seed == 42
    assert t2.thinking_budget == 4096
    assert t2.service_tier == "flex"


def test_string_settings_coercion() -> None:
    """Settings from TUI SQLite DB arrive as strings — __post_init__ must coerce."""
    t = GoogleTransport(
        vertexai="true",  # type: ignore[arg-type]
        temperature="0.7",  # type: ignore[arg-type]
        top_p="0.9",  # type: ignore[arg-type]
        seed="42",  # type: ignore[arg-type]
        thinking_budget="4096",  # type: ignore[arg-type]
    )
    assert t.vertexai is True
    assert t.temperature == 0.7
    assert t.top_p == 0.9
    assert t.seed == 42
    assert t.thinking_budget == 4096


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


def test_build_url_developer_api(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = GoogleTransport(api_key="test-key")
    url = t._build_url("models/gemini-test:streamGenerateContent", "alt=sse")
    assert "generativelanguage.googleapis.com" in url
    assert "key=test-key" in url
    assert "alt=sse" in url
    assert "gemini-test" in url


def test_build_url_vertex_ai() -> None:
    t = GoogleTransport(vertexai=True, project="my-proj", location="us-central1")
    url = t._build_url("publishers/google/models/gemini-test:streamGenerateContent", "alt=sse")
    assert "us-central1-aiplatform.googleapis.com" in url
    assert "projects/my-proj" in url
    assert "alt=sse" in url


def test_build_url_vertex_ai_global() -> None:
    t = GoogleTransport(vertexai=True, project="my-proj", location="global")
    url = t._build_url("publishers/google/models/gemini-test:streamGenerateContent")
    assert "aiplatform.googleapis.com" in url
    assert "global-aiplatform" not in url


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------


def test_generation_config_basic() -> None:
    t = GoogleTransport(temperature=0.5, top_p=0.8, seed=123)
    config = t._build_generation_config_json()
    assert config["temperature"] == 0.5
    assert config["topP"] == 0.8
    assert config["seed"] == 123
    assert config["maxOutputTokens"] == t.model.max_output_tokens


def test_generation_config_thinking() -> None:
    t = GoogleTransport(thinking_level="HIGH")
    config = t._build_generation_config_json()
    assert config["thinkingConfig"]["includeThoughts"] is True
    assert config["thinkingConfig"]["thinkingLevel"] == "HIGH"
