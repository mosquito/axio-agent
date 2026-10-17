"""The instruction contract works with independent provider implementations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from axio.exceptions import GuardError
from axio.instructions import Instruction, LoadedInstruction
from axio.permission import DenyAllGuard
from axio.testing import make_echo_tool
from axio.tool import Tool


class FileInstruction(Instruction):
    def __init__(self, path: Path) -> None:
        super().__init__(path.as_uri(), "Project conventions", source=str(path))
        self.path = path

    async def load(self) -> LoadedInstruction:
        return LoadedInstruction(await asyncio.to_thread(self.path.read_text, encoding="utf-8"))


class EchoInstruction(Instruction):
    async def load(self) -> LoadedInstruction:
        return LoadedInstruction("Use echo to repeat the supplied message.", (make_echo_tool(),))


def test_load_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Instruction("rules", "Rules")  # type: ignore[abstract]


async def test_metadata_does_not_load_and_each_load_reads_current_content(tmp_path: Path) -> None:
    path = tmp_path / "rules.txt"
    instruction: Instruction = FileInstruction(path)
    assert instruction.summary == "Project conventions"
    assert instruction.source == str(path)
    assert instruction.id == path.as_uri()
    with pytest.raises(FileNotFoundError):
        await instruction.load()
    path.write_text("Use UTC.", encoding="utf-8")
    first = await instruction.load()
    path.write_text("Use ISO dates.", encoding="utf-8")
    second = await instruction.load()
    assert first.text == "Use UTC."
    assert second.text == "Use ISO dates."
    assert first.tools == second.tools == ()


async def test_instruction_can_supply_executable_tools() -> None:
    instruction: Instruction = EchoInstruction("example:echo", "Repeat a message")
    loaded = await instruction.load()
    assert loaded.text == "Use echo to repeat the supplied message."
    assert len(loaded.tools) == 1
    assert "hello" in await loaded.tools[0](msg="hello")


async def test_loading_does_not_execute_tools_or_remove_guards() -> None:
    calls: list[str] = []

    async def write(value: str) -> str:
        calls.append(value)
        return value

    tool: Tool[object] = Tool(name="write", handler=write, guards=(DenyAllGuard(),))

    class WriteInstruction(Instruction):
        async def load(self) -> LoadedInstruction:
            return LoadedInstruction("Write the result.", (tool,))

    loaded = await WriteInstruction("example:write", "Write a result").load()
    assert calls == []
    assert loaded.tools[0] is tool
    with pytest.raises(GuardError, match="denied"):
        await loaded.tools[0](value="result")
    assert calls == []


async def test_cancelled_provider_load_releases_resources() -> None:
    started = asyncio.Event()
    released = asyncio.Event()

    class WaitingInstruction(Instruction):
        async def load(self) -> LoadedInstruction:
            try:
                started.set()
                await asyncio.Event().wait()
                return LoadedInstruction("Complete")
            finally:
                released.set()

    instruction = WaitingInstruction("remote:rules", "Remote rules")
    task = asyncio.create_task(instruction.load())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()


@pytest.mark.parametrize("id,summary", [("", "Rules"), ("  ", "Rules"), ("rules", ""), ("rules", " ")])
def test_catalog_metadata_must_not_be_empty(id: str, summary: str) -> None:
    with pytest.raises(ValueError):
        EchoInstruction(id, summary)
