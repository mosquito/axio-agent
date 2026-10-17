"""Keep MCP resource contents intact and collect paginated URI templates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp_tiny_mcp.client.base import BaseClient
from aiohttp_tiny_mcp.protocol.core import Operation
from aiohttp_tiny_mcp.protocol.models import ReadResourceResult

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.resources import ResourceCatalog
from axio_tools_mcp.session import MCPSession


async def test_resource_read_preserves_multiple_text_and_binary_contents(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MCPSession(MCPServerConfig(name="s", url="http://localhost"))
    contents = [
        {"uri": "custom://part/text", "text": "Some text", "mimeType": "text/plain"},
        {"uri": "custom://part/blob", "blob": "AAEC", "mimeType": "application/octet-stream"},
    ]
    read = AsyncMock(return_value=ReadResourceResult.model_validate({"contents": contents}))
    monkeypatch.setattr(session, "read_resource", read)
    result = json.loads(await ResourceCatalog(session).read_resource("custom://bundle?key=a%2Fb"))
    assert result["contents"] == contents
    read.assert_awaited_once_with("custom://bundle?key=a%2Fb")


async def test_template_pages_hold_stdio_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MCPSession(MCPServerConfig(name="s", command="unused"))
    session.lock = asyncio.Lock()
    client = AsyncMock(spec=BaseClient)
    session.client = client

    async def pages(operation: Operation) -> AsyncIterator[dict[str, Any]]:
        assert operation == Operation.LIST_RESOURCE_TEMPLATES
        assert session.lock is not None and session.lock.locked()
        yield {"resourceTemplates": [{"uriTemplate": "notes://{name}", "name": "notes"}], "nextCursor": "next"}
        assert session.lock.locked()
        yield {"resourceTemplates": [{"uriTemplate": "files://{path}", "name": "files"}]}

    monkeypatch.setattr(client, "pages", pages)
    templates = await session.list_resource_templates()
    assert [item.uri_template for item in templates] == ["notes://{name}", "files://{path}"]
    assert not session.lock.locked()


async def test_disconnected_template_request_fails() -> None:
    session = MCPSession(MCPServerConfig(name="s", command="unused"))
    with pytest.raises(RuntimeError, match="Not connected"):
        await session.list_resource_templates()
