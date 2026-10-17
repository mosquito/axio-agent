"""A general instruction provider independent of both applications."""

from __future__ import annotations

from typing import Any

from axio import Tool
from axio.instructions import Instruction, LoadedInstruction

opened: list[Provider] = []


async def answer(message: str) -> str:
    return "fixture tool answer"


class Guide(Instruction):
    def __init__(self, id: str, tools: tuple[Tool[Any], ...] = ()) -> None:
        super().__init__(id, "Fixture guide summary")
        self.tools = tools

    async def load(self) -> LoadedInstruction:
        return LoadedInstruction("Full fixture guide", self.tools)


class Provider:
    def __init__(self) -> None:
        self.closed = False
        self.items = (Guide("fixture:plain"), Guide("fixture:tools", (Tool("fixture_answer", answer),)))

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        opened.append(self)

    @property
    def all_tools(self) -> list[Tool[Any]]:
        return []

    def get_instructions(self) -> tuple[Instruction, ...]:
        return self.items

    async def close(self) -> None:
        self.closed = True
