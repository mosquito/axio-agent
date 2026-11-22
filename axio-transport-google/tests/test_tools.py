"""Tests for axio_transport_google.tools (image, video, search)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from axio.blocks import ImageBlock, TextBlock, VideoBlock

from axio_transport_google import GoogleTransport, tools
from axio_transport_google.tools import generate_image, generate_video, google_search


@pytest.fixture(autouse=True)
async def _reset_tools_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tools.transport, credentials, and env are clean for each test."""
    monkeypatch.setattr(tools, "transport", None)
    monkeypatch.setattr(tools, "_search_transport", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Isolate tests from any real antigravity credentials on the machine
    from axio_transport_google.antigravity import AntigravityFileCredentials

    monkeypatch.setattr(AntigravityFileCredentials, "list_records", AsyncMock(return_value=[]))
    if tools._search_session is not None and not tools._search_session.closed:
        await tools._search_session.close()
    monkeypatch.setattr(tools, "_search_session", None)


async def test_tools_unconfigured_error() -> None:
    img_res = await generate_image("a prompt")
    assert len(img_res) == 1
    assert isinstance(img_res[0], TextBlock)
    assert "not configured" in img_res[0].text

    vid_res = await generate_video("a prompt")
    assert len(vid_res) == 1
    assert isinstance(vid_res[0], TextBlock)
    assert "not configured" in vid_res[0].text

    search_res = await google_search("a query")
    assert "not configured" in search_res


async def test_google_search_with_bound_transport() -> None:
    mock_transport = AsyncMock()
    mock_transport.search = AsyncMock(
        return_value="Python is a language.\n\nSources:\n- [python.org](https://python.org)"
    )

    tools.transport = mock_transport
    result = await google_search("what is python")

    mock_transport.search.assert_awaited_once_with("what is python")
    assert "Python is a language." in result
    assert "https://python.org" in result


async def test_google_search_error_handling() -> None:
    mock_transport = AsyncMock()
    mock_transport.search = AsyncMock(side_effect=RuntimeError("API quota exceeded"))

    tools.transport = mock_transport
    result = await google_search("query")

    assert "Google search error: API quota exceeded" in result


async def test_google_search_fallback_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-api-key")

    mock_search = AsyncMock(return_value="Answer from env transport")
    monkeypatch.setattr(GoogleTransport, "search", mock_search)

    result = await google_search("query via env")
    assert result == "Answer from env transport"
    mock_search.assert_awaited_once_with("query via env")


async def test_google_search_fallback_to_antigravity(monkeypatch: pytest.MonkeyPatch) -> None:
    from axio_transport_google.antigravity import AntigravityFileCredentials, AntigravityTransport

    fake_account = MagicMock(account_id="acc_123")
    monkeypatch.setattr(AntigravityFileCredentials, "list_records", AsyncMock(return_value=[fake_account]))
    monkeypatch.setattr("axio_transport_google.antigravity.resolve_account", AsyncMock(return_value="acc_123"))
    monkeypatch.setattr(AntigravityTransport, "fetch_models", AsyncMock())
    mock_search = AsyncMock(return_value="Answer from antigravity transport")
    monkeypatch.setattr(AntigravityTransport, "search", mock_search)

    result = await google_search("query via antigravity")
    assert result == "Answer from antigravity transport"
    mock_search.assert_awaited_once_with("query via antigravity")


async def test_antigravity_transport_bind_tools() -> None:
    from axio_transport_google.antigravity import AntigravityTransport

    mock_transport = MagicMock(spec=AntigravityTransport)
    AntigravityTransport.bind_tools(mock_transport)
    assert tools.transport is mock_transport
    AntigravityTransport.bind_tools(None)
    assert tools.transport is None


async def test_generate_image_success() -> None:
    mock_transport = AsyncMock()
    mock_transport.generate_images = AsyncMock(return_value=[b"fake_png_data"])

    tools.transport = mock_transport
    result = await generate_image("a sunset")

    assert len(result) == 2
    assert isinstance(result[0], TextBlock)
    assert isinstance(result[1], ImageBlock)
    assert result[1].data == b"fake_png_data"


async def test_generate_video_success() -> None:
    mock_transport = AsyncMock()
    mock_transport.generate_videos = AsyncMock(return_value=[b"fake_mp4_data"])

    tools.transport = mock_transport
    result = await generate_video("a timelapse")

    assert len(result) == 2
    assert isinstance(result[0], TextBlock)
    assert isinstance(result[1], VideoBlock)
    assert result[1].data == b"fake_mp4_data"


async def test_google_transport_search_end_to_end() -> None:
    class FakeResponse:
        status = 200

        async def json(self) -> dict[str, Any]:
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Found it on Google."}]},
                        "groundingMetadata": {
                            "groundingChunks": [
                                {"web": {"uri": "https://example.com", "title": "Example"}},
                            ],
                        },
                    }
                ]
            }

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    mock_session = MagicMock()
    mock_session.post.return_value = FakeResponse()

    t = GoogleTransport(api_key="key", session=mock_session)
    result = await t.search("test search")

    assert "Found it on Google." in result
    assert "- [Example](https://example.com)" in result
    mock_session.post.assert_called_once()
