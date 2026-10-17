from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from axio.exceptions import GuardError
from axio.permission_policy import Approval, PolicyGuard
from axio.tool import Tool

from axio_tools_local.file_search import search_files


async def test_search_checks_symlink_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.chdir(root)
    (root / "file").write_text("hello")
    prompt = AsyncMock(return_value=Approval())
    policy = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("search", search_files, guards=(policy,))
    assert "hello" in await tool(query="hello")
    prompt.assert_not_awaited()
    outside = tmp_path / "secret"
    outside.write_text("private")
    (root / "link").symlink_to(outside)
    with pytest.raises(GuardError):
        await tool(query="private")
    assert prompt.await_args is not None
    assert str(outside) in {access.path for access in prompt.await_args.args[0].accesses}
