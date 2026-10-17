from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from axio.exceptions import GuardError
from axio.permission_policy import Approval, PolicyGuard
from axio.tool import Tool

from axio_tools_local.list_files import list_files
from axio_tools_local.patch_file import patch_file
from axio_tools_local.read_file import read_file
from axio_tools_local.shell import shell
from axio_tools_local.write_file import write_file


async def test_real_file_tools_separate_read_and_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file").write_text("before")
    prompt = AsyncMock(return_value=Approval())
    policy = PolicyGuard(tmp_path, prompt)
    reader: Tool[Any] = Tool("read", read_file, guards=(policy,))
    writer: Tool[Any] = Tool("write", write_file, guards=(policy,))
    patcher: Tool[Any] = Tool("patch", patch_file, guards=(policy,))
    assert "before" in await reader(filename="file")
    prompt.assert_not_awaited()
    with pytest.raises(GuardError):
        await writer(file_path="file", content="after")
    with pytest.raises(GuardError):
        await patcher(file_path="file", content="after", from_line=1, to_line=1)
    assert (tmp_path / "file").read_text() == "before"
    prompt.return_value = Approval("allow", "chat")
    await writer(file_path="file", content="after")
    assert (tmp_path / "file").read_text() == "after"
    prompt.return_value = Approval()
    with pytest.raises(GuardError):
        await writer(file_path="other", content="no")
    assert not (tmp_path / "other").exists()


async def test_listing_symlink_does_not_access_external_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    link = root / "outside-link"
    link.symlink_to(tmp_path / "outside")
    original = Path.stat

    def stat(path: Path, *, follow_symlinks: bool = True) -> Any:
        if path == link and follow_symlinks:
            pytest.fail("Listing must not follow the symlink")
        return original(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat)
    prompt = AsyncMock(return_value=Approval())
    policy = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("list", list_files, guards=(policy,))
    assert "outside-link" in await tool()
    prompt.assert_not_awaited()
    with pytest.raises(GuardError):
        await replace(tool, handler=read_file, access=None)(filename="outside-link")


async def test_shell_inside_project_is_not_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prompt = AsyncMock(return_value=Approval())
    policy = PolicyGuard(tmp_path, prompt)
    tool: Tool[Any] = Tool("shell", shell, guards=(policy,))
    with pytest.raises(GuardError):
        await tool(command="touch should-not-exist", cwd=str(tmp_path))
    assert not (tmp_path / "should-not-exist").exists()
    assert prompt.await_args is not None
    assert prompt.await_args.args[0].accesses[0].operation == "execute"
