"""Provider-independent instructions with content loaded on demand."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .tool import Tool
from .tool_access import Access


@dataclass(frozen=True, slots=True)
class LoadedInstruction:
    """A complete instruction and the tools needed to follow it.

    Tools retain their guards and access declarations. Their provider owns any
    resources they use and must keep those resources open while they are in use.
    Returning a tool does not execute it or grant permission to call it.
    """

    text: str
    tools: tuple[Tool[Any], ...] = ()


class Instruction(ABC):
    """An instruction whose provider implements asynchronous content loading.

    ``id`` identifies the instruction within a catalog. Providers must include
    enough source identity to distinguish instructions with the same name.
    ``summary`` is the short description available before loading the content.
    ``source`` is an optional display location; core does not interpret it.

    Rules, reference material, procedures, and skills use the same contract.
    The caller decides when loading is required. This class does not infer
    applicability from a filename, URI, or instruction type.
    """

    def __init__(self, id: str, summary: str, *, source: str = "") -> None:
        if not id.strip():
            raise ValueError("Instruction id must not be empty")
        if not summary.strip():
            raise ValueError("Instruction summary must not be empty")
        self.id = id
        self.summary = summary
        self.source = source

    def load_access(self) -> tuple[Access, ...]:
        """Declare loading effects for application guards; unknown unless overridden."""
        return (Access("unknown"),)

    @abstractmethod
    async def load(self) -> LoadedInstruction:
        """Return complete content and any tools needed to follow it.

        Implementations prepare dependencies before returning. They must not
        change an agent or execute the procedure described by the instruction.
        Exceptions, including cancellation, propagate to the caller. Providers
        must clean up resources acquired by a failed or cancelled load.

        Each call requests the current content. A provider may use a cache if
        it can validate its freshness. Core does not cache this method or share
        activation state between agents. Providers retain ownership of any
        sessions used by the returned tools.
        """
        ...


LOAD_INSTRUCTION_TOOL = "load_instruction"


class InstructionState:
    """One agent's catalog, loaded content, and changes awaiting an iteration.

    Providers retain their resources. This object only owns the agent's view of
    their instructions. A successful load stages changes; ``commit`` makes them
    available together at the next model boundary.
    """

    def __init__(self) -> None:
        self.catalog: dict[str, Instruction] = {}
        self.loaded: dict[str, LoadedInstruction] = {}
        self.pending: dict[str, LoadedInstruction] = {}
        self._lock = asyncio.Lock()

    def synchronize(self, instructions: Iterable[Instruction]) -> None:
        """Replace catalog metadata and discard removed or replaced content."""
        catalog: dict[str, Instruction] = {}
        for instruction in instructions:
            if instruction.id in catalog and catalog[instruction.id] is not instruction:
                raise ValueError(f"Conflicting instruction id: {instruction.id!r}")
            catalog[instruction.id] = instruction
        keep = {key for key, value in catalog.items() if self.catalog.get(key) is value}
        self.loaded = {key: value for key, value in self.loaded.items() if key in keep}
        self.pending = {key: value for key, value in self.pending.items() if key in keep}
        self.catalog = catalog

    @staticmethod
    def collect_tools(base: Iterable[Tool[Any]], content: Iterable[LoadedInstruction]) -> list[Tool[Any]]:
        """Preserve tool identities and reject conflicting names before activation."""
        tools: dict[str, Tool[Any]] = {}
        for tool in (*base, *(tool for item in content for tool in item.tools)):
            if tool.name in tools and tools[tool.name] is not tool:
                raise ValueError(f"Conflicting tool name: {tool.name!r}")
            tools[tool.name] = tool
        return list(tools.values())

    async def load(self, id: str, base: Iterable[Tool[Any]]) -> LoadedInstruction:
        """Stage one complete provider result; failure leaves the state unchanged."""
        async with self._lock:
            instruction = self.catalog.get(id)
            if instruction is None:
                raise ValueError(f"Unknown instruction: {id!r}")
            result = await instruction.load()
            if self.catalog.get(id) is not instruction:
                raise ValueError(f"Instruction changed during loading: {id!r}")
            self.collect_tools(base, {**self.loaded, **self.pending, id: result}.values())
            previous = self.loaded.get(id)
            if (
                previous is not None
                and previous.text == result.text
                and len(previous.tools) == len(result.tools)
                and all(a is b for a, b in zip(previous.tools, result.tools, strict=True))
            ):
                self.pending.pop(id, None)
            else:
                self.pending[id] = result
            return result

    def commit(self, base: Iterable[Tool[Any]]) -> list[Tool[Any]]:
        """Apply staged content atomically and return the current tool set."""
        loaded = {**self.loaded, **self.pending}
        tools = self.collect_tools(base, loaded.values())
        self.loaded = loaded
        self.pending.clear()
        return tools

    def render(self) -> str:
        """Render the short catalog and the content currently active for this agent."""
        if not self.catalog:
            return ""
        entries = [{"id": item.id, "summary": item.summary, "source": item.source} for item in self.catalog.values()]
        parts = [
            f"Available instructions. Use {LOAD_INSTRUCTION_TOOL}(id) to load full content and associated tools.",
            json.dumps(entries, ensure_ascii=False),
        ]
        parts.extend(f"Loaded instruction {json.dumps(id)}:\n{content.text}" for id, content in self.loaded.items())
        return "\n\n".join(parts)
