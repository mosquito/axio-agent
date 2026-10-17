"""Expose MCP resource metadata and contents without interpreting their URI schemes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from aiohttp_tiny_mcp.protocol.models import ResourceDef, ResourceTemplateDef

from .session import MCPSession


@dataclass(slots=True)
class ResourceCatalog:
    session: MCPSession
    resources: list[ResourceDef] = field(default_factory=list)
    templates: list[ResourceTemplateDef] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Keep all advertised resources and templates, including skill duplicates."""
        return {
            "server": self.session.config.name,
            "resources": [item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in self.resources],
            "resourceTemplates": [
                item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in self.templates
            ],
        }

    async def refresh(self) -> None:
        self.resources = await self.session.list_resources()
        self.templates = await self.session.list_resource_templates()

    async def list_resources(self) -> str:
        """Refresh and list this server's resources and URI templates, including skills."""
        await self.refresh()
        return json.dumps(self.snapshot(), ensure_ascii=False)

    async def read_resource(self, uri: str) -> str:
        """Read an exact resource URI from this server, including URIs built from its templates.

        Pass the URI unchanged. The server resolves it; it is not a local path.
        Returns MCP contents as JSON, with text or base64 blobs and MIME types.
        """
        result = await self.session.read_resource(uri)
        return result.model_dump_json(by_alias=True, exclude_none=True)
