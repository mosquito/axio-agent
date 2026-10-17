"""Separate generation and editing tools bound to one conversation's images."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from axio.blocks import ImageBlock, TextBlock
from axio.field import StrictStr
from axio.tool import Tool
from axio.tool_access import Access, tool_access

from .chatgpt_images import ChatGPTImageTransport, image_media_type, image_reference


class ChatGPTImageTools:
    """Bind two model-facing tools to an image transport and a local image catalog.

    The host registers attachments with add_image() and tells the agent their IDs.
    Tool results contain new image IDs and ImageBlocks, allowing subsequent edits
    without model-generated base64 or filesystem access. Create one instance per
    conversation. The host owns session lifetime and persists images if desired.
    """

    def __init__(self, transport: ChatGPTImageTransport, *, images: Mapping[str, ImageBlock] | None = None) -> None:
        self.transport = transport
        self._images: dict[str, ImageBlock] = {}
        for image_id, image in (images or {}).items():
            self.add_image(image, image_id=image_id)
        self.tools: list[Tool[Any]] = [
            Tool(name="generate_image", handler=self.generate_image),
            Tool(name="edit_image", handler=self.edit_image),
        ]

    @property
    def images(self) -> Mapping[str, ImageBlock]:
        """Read-only catalog of attachments and generated results, keyed by ID."""
        return MappingProxyType(self._images)

    def add_image(self, image: ImageBlock, *, image_id: str | None = None) -> str:
        """Register an attachment and return its ID for the agent's prompt."""
        image_reference(image)
        key = image_id if image_id is not None else f"image_{uuid.uuid4().hex}"
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Image IDs must be nonempty strings.")
        if key in self._images:
            raise ValueError(f"Image ID already exists: {key}")
        self._images[key] = image
        return key

    def _result(self, images: list[bytes]) -> list[TextBlock | ImageBlock]:
        blocks = [ImageBlock(media_type=image_media_type(data), data=data) for data in images]
        result: list[TextBlock | ImageBlock] = []
        for block in blocks:
            key = self.add_image(block)
            result.extend([TextBlock(text=f"Image ID: {key}"), block])
        return result

    @tool_access(lambda arguments: (Access("external"),))
    async def generate_image(self, prompt: StrictStr) -> list[TextBlock | ImageBlock]:
        """Create a new image from a description when the user requests generation.
        To change an existing image or use reference images, call edit_image instead.
        Returns an image and its ID, which can be used in later edits.
        """
        return self._result(await self.transport.generate_images(prompt))

    @tool_access(lambda arguments: (Access("external"),))
    async def edit_image(self, prompt: StrictStr, images: list[str]) -> list[TextBlock | ImageBlock]:
        """Edit existing images according to the requested changes.
        images must contain IDs registered by the application or returned by earlier
        image tool calls. Keep their order when referring to first/second images.
        Returns a new image and ID; source images remain available. Masks are unsupported.
        """
        if not isinstance(images, list) or not images:
            raise ValueError("Provide at least one source image ID.")
        sources = []
        for key in images:
            if not isinstance(key, str) or key not in self._images:
                raise ValueError("Unknown source image ID; use a registered attachment or earlier tool result.")
            sources.append(self._images[key])
        return self._result(await self.transport.edit_images(prompt, images=sources))
