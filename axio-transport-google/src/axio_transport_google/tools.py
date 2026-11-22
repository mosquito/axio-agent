"""Image, video, and search tools for Google GenAI transport."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

import aiohttp
from axio.blocks import ImageBlock, TextBlock, VideoBlock
from axio.field import StrictStr
from axio.transport import ImageGenTransport, SearchTransport, VideoGenTransport

logger = logging.getLogger(__name__)


@runtime_checkable
class _GenTransport(ImageGenTransport, VideoGenTransport, Protocol):
    """Combined Protocol — provides both image and video generation."""


# Set by the host app (e.g. axio-tui) to the active Google transport.
# ``None`` disables the tools with a configuration error message.
transport: Any | None = None

_search_session: aiohttp.ClientSession | None = None
_search_transport: SearchTransport | None = None


async def _resolve_search_transport() -> SearchTransport | None:
    global _search_session, _search_transport
    if transport is not None and (isinstance(transport, SearchTransport) or hasattr(transport, "search")):
        return transport  # type: ignore[no-any-return]

    # Check for logged-in Antigravity account
    try:
        from .antigravity import (
            AntigravityFileCredentials,
            AntigravityOAuth,
            AntigravitySession,
            AntigravityTransport,
            resolve_account,
        )

        store = AntigravityFileCredentials()
        records = await store.list_records()
        if records:
            if _search_transport is None or not isinstance(_search_transport, AntigravityTransport):
                if _search_session is None or _search_session.closed:
                    _search_session = aiohttp.ClientSession()
                account_id = await resolve_account(store)
                oauth = AntigravityOAuth(_search_session)
                account_session = AntigravitySession(store, oauth, account_id)
                t = AntigravityTransport(session=_search_session, account=account_session)
                await t.fetch_models()
                _search_transport = t
            return _search_transport
    except Exception as exc:
        logger.debug("Failed to initialize Antigravity for search: %s", exc)

    # Check for GEMINI_API_KEY in environment
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        from . import GoogleTransport

        if _search_transport is None or not isinstance(_search_transport, GoogleTransport):
            if _search_session is None or _search_session.closed:
                _search_session = aiohttp.ClientSession()

            _search_transport = GoogleTransport(api_key=api_key, session=_search_session)
        return _search_transport
    return None


async def generate_image(
    prompt: StrictStr,
    model: StrictStr = "gemini-3-pro-image",
    n: int = 1,
) -> list[TextBlock | ImageBlock]:
    """Generate images from a text prompt using Google Gemini image models
    (Nano Banana family). Only use when the user explicitly asks to generate
    an image — never for screenshots, UI testing, or verifying application
    output. Use a descriptive, detailed prompt for best results."""
    if transport is None:
        return [TextBlock(text="Error: image generation transport not configured")]
    try:
        images = await transport.generate_images(prompt, model=model, n=n)
    except Exception as exc:
        logger.error("Image generation failed: %s", exc, exc_info=True)
        return [TextBlock(text=f"Image generation error: {exc}")]
    if not images:
        return [TextBlock(text="No images generated")]
    result: list[TextBlock | ImageBlock] = [TextBlock(text=f"Generated {len(images)} image(s) for: {prompt}")]
    for img_bytes in images:
        result.append(ImageBlock(media_type="image/png", data=img_bytes))
    return result


async def generate_video(
    prompt: StrictStr,
    model: StrictStr = "veo-3.1-fast-generate-001",
    duration_seconds: int = 6,
    aspect_ratio: StrictStr = "16:9",
) -> list[TextBlock | VideoBlock]:
    """Generate a video from a text prompt using Google Veo models.
    Only use when the user explicitly asks to generate a video — never for
    screen recordings, UI testing, or verifying application output.
    This is an async operation that may take 1-3 minutes."""
    if transport is None:
        return [TextBlock(text="Error: video generation transport not configured")]
    try:
        videos = await transport.generate_videos(
            prompt,
            model=model,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
    except Exception as exc:
        logger.error("Video generation failed: %s", exc, exc_info=True)
        return [TextBlock(text=f"Video generation error: {exc}")]
    if not videos:
        return [TextBlock(text="No videos generated")]
    result: list[TextBlock | VideoBlock] = [TextBlock(text=f"Generated {len(videos)} video(s) for: {prompt}")]
    for vid_bytes in videos:
        result.append(VideoBlock(media_type="video/mp4", data=vid_bytes))
    return result


async def close_tools() -> None:
    """Close any ad-hoc HTTP session opened by tools."""
    global _search_session, _search_transport
    if _search_session is not None and not _search_session.closed:
        await _search_session.close()
    _search_session = None
    _search_transport = None


async def google_search(
    query: StrictStr,
) -> str:
    """Search the web using Google Search via Gemini grounding.
    Returns grounded factual search results with sources and URLs."""
    active_transport = await _resolve_search_transport()
    if active_transport is None:
        return (
            "Error: Google Search transport not configured (no active Google/Antigravity transport, "
            "logged-in Antigravity account, or GEMINI_API_KEY)."
        )
    try:
        return await active_transport.search(query)
    except Exception as exc:
        logger.error("Google search failed: %s", exc, exc_info=True)
        return f"Google search error: {exc}"
