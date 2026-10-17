from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from axio.exceptions import GuardError
from axio.permission import PermissionGuard
from axio.permission_policy import Approval, PermissionRequest, PermissionRule, PolicyGuard
from axio.tool import Tool
from axio.tool_access import Access, file_access, tool_access


@tool_access(file_access("read", "path"))
async def read(path: str = ".") -> str:
    return path


@tool_access(file_access("write", "path"))
async def write(path: str, content: str = "") -> str:
    return content


async def opaque(command: str, cwd: str = ".") -> str:
    return command


class MemoryRules:
    def __init__(self) -> None:
        self.saved: dict[str, PermissionRule] = {}

    async def load(self) -> list[PermissionRule]:
        return list(self.saved.values())

    async def add(self, rule: PermissionRule) -> None:
        self.saved[rule.id] = rule

    async def remove(self, rule_id: str) -> None:
        self.saved.pop(rule_id, None)


class Prompts:
    def __init__(self, answer: Approval = Approval()) -> None:
        self.answer = answer
        self.requests: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> Approval:
        self.requests.append(request)
        return self.answer


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project.with.dots"
    project.mkdir()
    monkeypatch.chdir(project)
    return project


async def test_read_only_inside_root_and_default_path(root: Path) -> None:
    prompt = Prompts()
    guard = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("read", read, guards=(guard,))
    assert await tool(path="file_without_extension") == "file_without_extension"
    assert await tool() == "."
    assert not prompt.requests


@pytest.mark.parametrize("via_link", [False, True])
async def test_outside_read_needs_approval(root: Path, via_link: bool) -> None:
    prompt = Prompts()
    outside = root.parent / "secret"
    outside.write_text("private")
    (root / "link").symlink_to(outside)
    guard = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("read", read, guards=(guard,))
    with pytest.raises(GuardError):
        await tool(path="link" if via_link else "../secret")
    assert prompt.requests[0].accesses == (Access("read", str(outside)),)


async def test_once_read_does_not_allow_next_call_or_write(root: Path) -> None:
    prompt = Prompts(Approval("allow"))
    guard = PolicyGuard(root, prompt)
    reader: Tool[Any] = Tool("file", read, guards=(guard,))
    writer: Tool[Any] = Tool("file", write, guards=(guard,))
    await reader(path="../secret")
    await reader(path="../secret")
    await writer(path="inside", content="changed")
    assert len(prompt.requests) == 3


async def test_directory_grant_has_explicit_scope_and_lifetime(root: Path) -> None:
    prompt = Prompts(Approval("allow", "chat", directories=True))
    guard = PolicyGuard(root, prompt)
    guard.set_session("first")
    writer: Tool[Any] = Tool("write", write, guards=(guard,))
    await writer(path="src/a.py")
    await writer(path="src/lib/b.py")
    assert len(prompt.requests) == 1
    await writer(path="src_other/b.py")
    assert len(prompt.requests) == 2
    guard.set_session("second")
    await writer(path="src/a.py")
    assert len(prompt.requests) == 3
    guard.set_session("first")
    await writer(path="src/a.py")
    assert len(prompt.requests) == 4


async def test_file_grant_does_not_follow_retargeted_symlink(root: Path) -> None:
    prompt = Prompts(Approval("allow", "chat"))
    guard = PolicyGuard(root, prompt)
    link = root / "link"
    link.symlink_to(root / "first")
    writer: Tool[Any] = Tool("write", write, guards=(guard,))
    await writer(path="link")
    link.unlink()
    link.symlink_to(root / "second")
    await writer(path="link")
    assert len(prompt.requests) == 2


async def test_opaque_rule_binds_arguments_cwd_and_provider(root: Path) -> None:
    prompt = Prompts(Approval("allow", "chat"))
    guard = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("shell", opaque, guards=(guard,), permission_identity="server-one")
    await tool(command="ls")
    await tool(command="ls")
    await tool(command="rm file")
    await tool(command="ls", cwd="elsewhere")
    await replace(tool, permission_identity="server-two")(command="ls")
    assert len(prompt.requests) == 4
    assert not prompt.requests[0].files_only


async def test_deny_precedes_auto_read_and_allow_for_all_resources(root: Path) -> None:
    store = MemoryRules()
    guard = PolicyGuard(root, Prompts(Approval("allow")), store=store)
    tool: Tool[Any] = Tool("read", read, access=lambda args: (Access("read", "a"), Access("read", "secret")))
    request = guard.describe(tool, {})
    allow = PermissionRule.from_request(request, Approval("allow", "project"))
    deny = PermissionRule.from_request(replace(request, accesses=(request.accesses[1],)), Approval("deny", "project"))
    await store.add(allow)
    await store.add(deny)
    assert (await guard.evaluate(request))[0] == "deny"
    with pytest.raises(GuardError, match="Denied by"):
        await guard.check(tool)
    await guard.revoke(deny.id)
    assert (await guard.evaluate(request))[0] == "allow"


async def test_concurrent_calls_recheck_rules_and_reads_do_not_wait(root: Path) -> None:
    entered, answer = asyncio.Event(), asyncio.Event()
    prompts = 0

    async def prompt(request: PermissionRequest) -> Approval:
        nonlocal prompts
        prompts += 1
        entered.set()
        await answer.wait()
        return Approval("allow", "chat")

    guard = PolicyGuard(root, prompt)
    writer: Tool[Any] = Tool("write", write, guards=(guard,))
    reader: Tool[Any] = Tool("read", read, guards=(guard,))
    first = asyncio.create_task(writer(path="file"))
    await asyncio.wait_for(entered.wait(), 1)
    second = asyncio.create_task(writer(path="file"))
    assert await asyncio.wait_for(reader(path="local"), 1) == "local"
    answer.set()
    await asyncio.wait_for(asyncio.gather(first, second), 1)
    assert prompts == 1


async def test_cancelled_prompt_does_not_grant_or_block_next_call(root: Path) -> None:
    entered = asyncio.Event()

    async def prompt(request: PermissionRequest) -> Approval:
        entered.set()
        await asyncio.Event().wait()
        return Approval("allow", "chat")

    guard = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("write", write, guards=(guard,))
    task = asyncio.create_task(tool(path="file"))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await guard.rules() == []
    guard.prompt = Prompts()
    with pytest.raises(GuardError, match="User denied"):
        await asyncio.wait_for(tool(path="file"), 1)


async def test_changed_path_and_session_while_prompting(root: Path) -> None:
    link = root / "link"
    link.symlink_to(root / "first")
    requests = []

    async def prompt(request: PermissionRequest) -> Approval:
        requests.append(request)
        if len(requests) == 1:
            link.unlink()
            link.symlink_to(root / "second")
            return Approval("allow")
        guard.set_session("another-chat")
        return Approval("allow", "chat")

    guard = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("write", write, guards=(guard,))
    with pytest.raises(GuardError, match="Conversation changed"):
        await tool(path="link")
    assert len(requests) == 2
    assert await guard.rules() == []


async def test_existing_guard_rewrites_arguments_before_policy(root: Path) -> None:
    class Redirect(PermissionGuard):
        async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
            return {**kwargs, "path": "../secret"}

    prompt = Prompts()
    policy = PolicyGuard(root, prompt)
    tool: Tool[Any] = Tool("read", read, guards=(Redirect(), policy))
    with pytest.raises(GuardError):
        await tool(path="safe")
    assert prompt.requests[0].accesses == (Access("read", str(root.parent / "secret")),)


async def test_failed_review_or_safe_advice_cannot_authorize(root: Path) -> None:
    async def advice(tool: Tool[Any], kwargs: Any) -> str:
        return "SAFE"

    prompt = Prompts()
    guard = PolicyGuard(root, prompt, advisor=advice)
    tool: Tool[Any] = Tool("write", write, guards=(guard,))
    with pytest.raises(GuardError):
        await tool(path="file")
    assert "SAFE" in prompt.requests[0].reason

    async def failure(tool: Tool[Any], kwargs: Any) -> str:
        raise RuntimeError("review failed")

    guard.advisor = failure
    with pytest.raises(GuardError):
        await tool(path="file")
    assert "Review unavailable" in prompt.requests[1].reason


async def test_streaming_cannot_bypass_policy(root: Path) -> None:
    tool: Tool[Any] = Tool("write", write, guards=(PolicyGuard(root, Prompts()),))
    with pytest.raises(GuardError):
        async for _ in tool.call_streaming(path="file"):
            pytest.fail("Denied call produced output")


async def test_project_rule_round_trip_revoke_and_failed_prompt(root: Path) -> None:
    store = MemoryRules()
    prompt = Prompts(Approval("allow", "project"))
    guard = PolicyGuard(root, prompt, store=store)
    tool: Tool[Any] = Tool("write", write, guards=(guard,))
    await tool(path="file")
    rule = next(iter(store.saved.values()))
    assert PermissionRule.from_json(rule.to_json()) == rule
    restarted = PolicyGuard(root, Prompts(), store=store)
    await replace(tool, guards=(restarted,))(path="file")
    await restarted.revoke(rule.id)
    with pytest.raises(GuardError):
        await replace(tool, guards=(restarted,))(path="file")

    async def broken(request: PermissionRequest) -> Approval:
        raise ImportError("dialog unavailable")

    restarted.prompt = broken
    with pytest.raises(GuardError, match="dialog unavailable"):
        await replace(tool, guards=(restarted,))(path="file")
