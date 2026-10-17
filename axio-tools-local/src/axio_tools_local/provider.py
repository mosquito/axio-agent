"""Local tools and instruction sources, independent of an assistant application."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, get_type_hints

from axio import Agent, Tool
from axio.blocks import ToolResultBlock, ToolUseBlock
from axio.instructions import Instruction, LoadedInstruction
from axio.tool_access import Access

from .file_search import search_files
from .instructions import DEFAULT_MAX_BYTES, Instructions, load_instructions
from .list_files import list_files
from .patch_file import patch_file
from .paths import LocalPaths
from .read_file import read_file
from .run_python import run_python
from .shell import shell
from .skills import LocalSkill, LocalSkills
from .write_file import write_file


@dataclass(frozen=True, slots=True)
class LocalOptions:
    """Explicit inputs for one local provider. No process directory is changed."""

    cwd: Path
    roots: tuple[Path, ...]
    isolated: bool = False
    instruction_files: tuple[Path, ...] = ()
    skill_directories: tuple[Path, ...] = ()
    max_instruction_bytes: int = DEFAULT_MAX_BYTES


def _bind_tool(handler: Callable[..., Awaitable[Any]], cwd: Path, parameter: str) -> Tool[Any]:
    """Bind this provider's path parameter for both execution and effect reporting."""
    signature = inspect.signature(handler)
    declaration = getattr(handler, "__axio_access__")

    def arguments(values: Mapping[str, Any]) -> dict[str, Any]:
        bound = signature.bind(**values)
        bound.apply_defaults()
        result = dict(bound.arguments)
        value = result[parameter]
        if not isinstance(value, str):
            raise ValueError(f"Invalid path: {parameter}")
        # Retain symlink spelling so preflight can observe later retargeting.
        result[parameter] = str(cwd / value)
        return result

    @wraps(handler)
    async def call(**values: Any) -> Any:
        return await handler(**arguments(values))

    call.__annotations__ = get_type_hints(handler, include_extras=True)

    def access(values: Mapping[str, Any]) -> tuple[Access, ...]:
        effects: tuple[Access, ...] = declaration(arguments(values))
        return effects

    stream = getattr(handler, "stream", None)
    if stream is not None:

        async def bound_stream(**values: Any) -> AsyncIterator[tuple[str, str]]:
            async for key, text in stream(**arguments(values)):
                yield key, text

        setattr(call, "stream", bound_stream)
    return Tool(name=handler.__name__, handler=call, access=access)


def local_tools(cwd: Path) -> list[Tool[Any]]:
    """Create stable tools whose relative paths use the supplied directory."""
    return [
        _bind_tool(read_file, cwd, "filename"),
        _bind_tool(write_file, cwd, "file_path"),
        _bind_tool(patch_file, cwd, "file_path"),
        _bind_tool(list_files, cwd, "directory"),
        _bind_tool(search_files, cwd, "path"),
        _bind_tool(shell, cwd, "cwd"),
        _bind_tool(run_python, cwd, "cwd"),
    ]


class LocalSkillInstruction(Instruction):
    """Load an Agent Skills document and attach its local execution tools."""

    def __init__(self, skill: LocalSkill, tools: Sequence[Tool[Any]]) -> None:
        summary = f"{skill.name}: {skill.description}"
        if skill.compatibility:
            summary += f" Compatibility: {skill.compatibility}"
        super().__init__(f"local:{skill.source}", summary, source=str(skill.source))
        self.skill = skill
        self.tools = tuple(tools)

    def load_access(self) -> tuple[Access, ...]:
        return (Access("read", str(self.skill.source)),)

    async def load(self) -> LoadedInstruction:
        document = json.loads(await asyncio.to_thread(LocalSkills.read_file, self.skill, "SKILL.md"))
        text: str = document.pop("text")
        return LoadedInstruction(
            "Resolve relative script and resource paths against the returned skill directory. "
            "Read related text with read_local_skill(location, path). Check compatibility before execution.\n"
            + json.dumps(document, ensure_ascii=False)
            + "\n\n"
            + text,
            self.tools,
        )


class ScopedInstruction(Instruction):
    """One agent's complete snapshot of required local rules."""

    def __init__(self, paths: LocalPaths, options: LocalOptions) -> None:
        super().__init__("local:scoped-instructions", "Required file instructions for accessed paths")
        self.paths = paths
        self.options = options
        self.targets = {paths.cwd}
        self.snapshot = Instructions((), options.max_instruction_bytes)

    async def load(self) -> LoadedInstruction:
        snapshot = await asyncio.to_thread(
            load_instructions,
            self.paths,
            targets=tuple(self.targets),
            isolated=self.options.isolated,
            files=self.options.instruction_files,
            max_bytes=self.options.max_instruction_bytes,
        )
        self.snapshot = snapshot
        return LoadedInstruction(snapshot.render())

    def writes_instructions(self, effects: Sequence[Access]) -> bool:
        sources = {block.source for block in self.snapshot.blocks}
        for effect in effects:
            if effect.operation != "write" or effect.path is None:
                continue
            path = Path(effect.path)
            resolved = self.paths.resolve(path)
            if path.name == "AGENTS.md" or resolved.name == "AGENTS.md" or resolved in sources:
                return True
            if resolved.exists() and any(resolved.samefile(source) for source in sources if source.exists()):
                return True
        return False


class LocalBinding:
    """Compose required file instructions and preflight hooks for one agent.

    Effect declarations are a planning boundary. They cannot observe paths
    hidden inside shell commands, Python code, or undeclared MCP operations.
    """

    def __init__(self, provider: LocalToolProvider, agent: Agent) -> None:
        self.provider = provider
        self.agent = agent
        self.instruction = ScopedInstruction(provider.paths, provider.options)
        self._skill_ids: set[str] = set()

    def describe(self) -> str:
        return self.provider.paths.describe()

    async def before_turn(self, iteration: int) -> None:
        if self.provider in self.agent.providers:
            return
        catalog = self.provider.get_instructions()
        old = self._skill_ids
        self.agent.instructions = [item for item in self.agent.instructions if item.id not in old]
        self.agent.instructions.extend(catalog)
        self._skill_ids = {item.id for item in catalog}

    def effects(self, blocks: list[ToolUseBlock]) -> dict[str, tuple[Access, ...]]:
        result: dict[str, tuple[Access, ...]] = {}
        for block in blocks:
            tool = self.agent.get_tool(block.name)
            if tool is not None and tool.access is not None:
                result[block.id] = tool.access(block.input)
        return result

    async def before_tools(self, blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        effects = await asyncio.to_thread(self.effects, blocks)
        previous = self.instruction.targets
        self.instruction.targets = previous | {
            self.provider.paths.cwd / effect.path
            for accesses in effects.values()
            for effect in accesses
            if effect.path is not None
        }
        try:
            await self.agent.load_instruction(self.instruction.id)
        except BaseException:
            self.instruction.targets = previous
            raise
        edits = [block for block in blocks if self.instruction.writes_instructions(effects.get(block.id, ()))]
        if edits and len(blocks) > 1:
            return [
                ToolResultBlock(
                    tool_use_id=block.id,
                    content=(
                        "Deferred: only the first instruction edit can run. Review updated instructions next turn."
                    ),
                    is_error=True,
                )
                for block in blocks
                if block.id != edits[0].id
            ]
        return []


class LocalToolProvider:
    """Own local tool identities and the reloadable local skill catalog."""

    def __init__(self) -> None:
        self._options: LocalOptions | None = None
        self._paths: LocalPaths | None = None
        self._skills: LocalSkills | None = None
        self._tools: list[Tool[Any]] = []
        self._instructions: tuple[Instruction, ...] = ()

    @property
    def options(self) -> LocalOptions:
        if self._options is None:
            raise RuntimeError("Local provider is not initialized")
        return self._options

    @property
    def paths(self) -> LocalPaths:
        if self._paths is None:
            raise RuntimeError("Local provider is not initialized")
        return self._paths

    @property
    def skills(self) -> LocalSkills:
        if self._skills is None:
            raise RuntimeError("Local provider is not initialized")
        return self._skills

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        options = config.get("local") if isinstance(config, Mapping) else config
        if options is None:
            cwd = Path.cwd()
            options = LocalOptions(cwd, (cwd,))
        if not isinstance(options, LocalOptions):
            raise TypeError("Local provider configuration must be LocalOptions")
        if not options.roots:
            raise ValueError("At least one local root is required")
        self._paths = LocalPaths.discover(options.cwd, options.roots[0], options.roots[1:])
        self._options = options
        self._tools = local_tools(self.paths.cwd)
        self._skills = LocalSkills(
            self.paths.cwd, directories=list(options.skill_directories), isolated=options.isolated
        )
        await self.reload()

    async def refresh(self, *, config: Any = None) -> None:
        await self.reload()

    async def reload(self) -> None:
        await self.skills.reload()
        tools = (*self._tools, self.skills.reader)
        self._instructions = tuple(
            LocalSkillInstruction(skill, tools) for skill in self.skills.catalog.skills.values()
        )

    @property
    def all_tools(self) -> list[Tool[Any]]:
        return list(self._tools)

    def get_instructions(self) -> Sequence[Instruction]:
        return self._instructions

    def bind(self, agent: Agent) -> LocalBinding:
        """Bind a fresh snapshot; call separately for each agent, including copies."""
        previous = [
            getattr(hook, "__self__")
            for hook in agent.before_turn_hooks
            if isinstance(getattr(hook, "__self__", None), LocalBinding)
        ]
        removed_ids = {id for binding in previous for id in binding._skill_ids}
        agent.instructions = [
            item
            for item in agent.instructions
            if not isinstance(item, ScopedInstruction) and item.id not in removed_ids
        ]
        agent.before_turn_hooks = tuple(
            hook for hook in agent.before_turn_hooks if not isinstance(getattr(hook, "__self__", None), LocalBinding)
        )
        agent.before_tools_hooks = tuple(
            hook for hook in agent.before_tools_hooks if not isinstance(getattr(hook, "__self__", None), LocalBinding)
        )
        agent.system_contributions = tuple(
            part
            for part in agent.system_contributions
            if not isinstance(getattr(part, "__self__", None), LocalBinding)
        )
        binding = LocalBinding(self, agent)
        agent.system_contributions += (binding.describe,)
        agent.instructions.append(binding.instruction)
        agent.required_instructions = tuple(dict.fromkeys((*agent.required_instructions, binding.instruction.id)))
        agent.before_turn_hooks += (binding.before_turn,)
        agent.before_tools_hooks += (binding.before_tools,)
        return binding

    async def close(self) -> None:
        """Local tools do not retain open files or subprocesses between calls."""
