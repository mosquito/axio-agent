"""Native ChatGPT image generation using the selected account session."""

from __future__ import annotations

import base64
import binascii
import uuid
from typing import Any, Literal

import aiohttp
from axio.blocks import ImageBlock, ImageMediaType
from axio.models import ModelRegistry, ModelSpec
from axio_responses import Recovery, ResponsesAPIError, ResponsesInterruptedError, request_id

from .chatgpt_models import CHATGPT_IMAGE_MODELS
from .session import ChatGPTSession
from .subscription import CODEX_BASE_URL


def image_media_type(data: bytes) -> ImageMediaType:
    """Identify a PNG, JPEG, or WebP result without relying on a file extension."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("The image response is not PNG, JPEG, or WebP.")


class ChatGPTImageError(ResponsesAPIError):
    """Image endpoint diagnostics, including native ChatGPT quota failures."""

    @property
    def recovery(self) -> Recovery:
        error = self.body.get("error") if isinstance(self.body, dict) else None
        if self.code == "usage_limit_reached" or (
            isinstance(error, dict) and error.get("type") == "usage_limit_reached"
        ):
            return Recovery.quota
        return super().recovery


class ChatGPTImageTransport:
    """Generate image bytes using native ChatGPT OAuth, with no API-key fallback.

    The caller owns both sessions. Models are supported image endpoint presets,
    not a promise of account quota. POST requests are never automatically replayed:
    a lost response, cancellation, or server error may hide a completed generation.
    """

    name = "ChatGPT Images"

    def __init__(
        self,
        *,
        account: ChatGPTSession,
        session: aiohttp.ClientSession,
        model: str | ModelSpec = "gpt-image-2",
        base_url: str = CODEX_BASE_URL,
    ) -> None:
        self.account = account
        self.session = session
        self.base_url = base_url
        self.models = ModelRegistry(CHATGPT_IMAGE_MODELS.values())
        model_id = model.id if isinstance(model, ModelSpec) else model
        if model_id not in self.models:
            raise ValueError(f"Unsupported ChatGPT image model: {model_id}")
        self.model = self.models[model_id]

    def with_model(self, model: str | ModelSpec) -> ChatGPTImageTransport:
        """Select an image model while sharing account renewal and quota state."""
        return type(self)(account=self.account, session=self.session, model=model, base_url=self.base_url)

    async def generate_images(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: Literal["auto", "1024x1024", "1536x1024", "1024x1536"] = "auto",
        quality: Literal["auto", "low", "medium", "high"] = "auto",
        transparent_background: bool = False,
    ) -> list[bytes]:
        """Return complete images; use image_media_type() to identify their format."""
        payload = self._payload(prompt, model, n, size, quality, transparent_background)
        return await self._request("generations", payload, n)

    async def edit_images(
        self,
        prompt: str,
        *,
        images: list[ImageBlock],
        mask: ImageBlock | None = None,
        model: str | None = None,
        n: int = 1,
        size: Literal["auto", "1024x1024", "1536x1024", "1024x1536"] = "auto",
        quality: Literal["auto", "low", "medium", "high"] = "auto",
        transparent_background: bool = False,
    ) -> list[bytes]:
        """Edit one or more PNG/JPEG/WebP sources using native ChatGPT Images.

        The native Codex protocol does not expose masks. Passing one fails before
        authorization or HTTP; sources are never silently treated as a text prompt.
        """
        payload = self._payload(prompt, model, n, size, quality, transparent_background)
        if mask is not None:
            raise ValueError("The native ChatGPT image edit route does not support masks.")
        if not isinstance(images, list) or not images:
            raise ValueError("Image editing requires a nonempty list of ImageBlock sources.")
        payload["images"] = [image_reference(image) for image in images]
        return await self._request("edits", payload, n)

    def _payload(
        self, prompt: str, model: str | None, n: int, size: str, quality: str, transparent_background: bool
    ) -> dict[str, Any]:
        model_id = self.model.id if model is None else model
        if model_id not in self.models:
            raise ValueError(f"Unsupported ChatGPT image model: {model_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The image prompt must not be empty.")
        if type(n) is not int or n != 1:
            raise ValueError("This transport currently supports one image per request (n=1).")
        if size not in {"auto", "1024x1024", "1536x1024", "1024x1536"}:
            raise ValueError("Unsupported image size.")
        if quality not in {"auto", "low", "medium", "high"}:
            raise ValueError("Unsupported image quality.")
        if type(transparent_background) is not bool:
            raise ValueError("transparent_background must be a boolean.")
        return {
            "model": model_id,
            "prompt": prompt,
            "background": "transparent" if transparent_background else "opaque",
            "quality": quality,
            "size": size,
        }

    async def _request(self, operation: str, payload: dict[str, Any], n: int) -> list[bytes]:
        if self.account.plan_usage_error is not None:
            raise self.account.plan_usage_error
        headers = await self.account.request_headers(images=True)
        headers["x-codex-image-turn-id"] = str(uuid.uuid4())
        response: aiohttp.ClientResponse | None = None
        try:
            async with self.session.post(
                f"{self.base_url.rstrip('/')}/images/{operation}",
                headers=headers,
                json=payload,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise ChatGPTImageError.from_http(
                        await response.text(), status=response.status, headers=response.headers
                    )
                try:
                    body = await response.json()
                    if isinstance(body, dict) and "error" in body:
                        raise ChatGPTImageError.from_body(
                            body, status=response.status, request_id=request_id(response.headers)
                        )
                    return decode_images(body, n)
                except (ValueError, aiohttp.ContentTypeError) as exc:
                    raise ChatGPTImageError(
                        "ChatGPT returned an invalid or incomplete image result.",
                        status=response.status,
                        request_id=request_id(response.headers),
                    ) from exc
        except ChatGPTImageError as exc:
            if exc.recovery is Recovery.quota:
                self.account.plan_usage_error = exc
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise ResponsesInterruptedError(
                "Image request was interrupted; its outcome is unknown. The request was not retried.",
                status=response.status if response is not None else None,
                request_id=request_id(response.headers) if response is not None else None,
            ) from exc


def image_reference(image: ImageBlock) -> dict[str, str]:
    """Encode a supported source without file reads or remote URL fetching."""
    if not isinstance(image, ImageBlock) or not isinstance(image.data, bytes):
        raise ValueError("Image editing requires ImageBlock sources containing bytes.")
    if image_media_type(image.data) != image.media_type:
        raise ValueError("Image source media_type does not match its contents.")
    encoded = base64.b64encode(image.data).decode("ascii")
    return {"image_url": f"data:{image.media_type};base64,{encoded}"}


def decode_images(body: Any, n: int) -> list[bytes]:
    if not isinstance(body, dict) or not isinstance(body.get("data"), list) or len(body["data"]) != n:
        raise ValueError("The response did not contain the requested number of images.")
    images = []
    for item in body["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("b64_json"), str):
            raise ValueError("Missing base64 image result.")
        try:
            data = base64.b64decode(item["b64_json"], validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("Invalid base64 image result.") from None
        image_media_type(data)
        images.append(data)
    return images
