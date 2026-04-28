"""Tests for axio.permission: guards."""

from __future__ import annotations

from typing import Any

import pytest

from axio.exceptions import GuardError
from axio.permission import AllowAllGuard, DenyAllGuard, PermissionGuard
from axio.tool import Tool


async def _noop() -> str:
    return ""


_tool: Tool[Any] = Tool(name="t", description="t", handler=_noop)


class TestAllowAllGuard:
    async def test_returns_kwargs(self) -> None:
        guard = AllowAllGuard()
        result = await guard.check(_tool, x=1, y="hello")
        assert result == {"x": 1, "y": "hello"}

    def test_satisfies_protocol(self) -> None:
        assert isinstance(AllowAllGuard(), PermissionGuard)


class TestDenyAllGuard:
    async def test_raises(self) -> None:
        guard = DenyAllGuard()
        with pytest.raises(GuardError, match="denied"):
            await guard.check(_tool)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(DenyAllGuard(), PermissionGuard)
