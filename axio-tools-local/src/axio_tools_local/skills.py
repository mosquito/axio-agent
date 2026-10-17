"""Discover local Agent Skills and load their instructions on demand."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from axio.tool import Tool
from axio.tool_access import Access

from .paths import config_directory

MAX_HEADER_BYTES = 64 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_DIRECTORIES = 2000
MAX_DEPTH = 6
SKIP_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__"}
LOCAL_SKILL_TOOL = "read_local_skill"


def default_directories() -> tuple[Path, ...]:
    """Search known skill locations, without walking the entire home directory."""
    home = Path.home()
    return (
        config_directory() / "skills",
        home / ".agents/skills",
        home / ".claude/skills",
        home / ".codex/skills",
        home / ".cursor/skills",
        config_directory().parent / "opencode/skills",
    )


def read_header(source: Path) -> tuple[dict[str, Any], str]:
    """Read bounded YAML frontmatter without loading the Markdown body."""
    if not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("SKILL.md must be a regular file")
    with source.open("rb") as stream:
        first = stream.readline(MAX_HEADER_BYTES + 1)
        if first.decode("utf-8-sig").strip() != "---":
            raise ValueError("SKILL.md must start with YAML frontmatter")
        header = bytearray(first)
        lines: list[str] = []
        while len(header) <= MAX_HEADER_BYTES:
            line = stream.readline(MAX_HEADER_BYTES + 1)
            if not line:
                raise ValueError("Missing closing frontmatter delimiter")
            header.extend(line)
            if len(header) > MAX_HEADER_BYTES:
                break
            text = line.decode("utf-8")
            if text.strip() == "---":
                try:
                    data = yaml.safe_load("".join(lines))
                except yaml.YAMLError as error:
                    raise ValueError("Invalid YAML frontmatter") from error
                if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
                    raise ValueError("Frontmatter must be a mapping with string keys")
                return data, hashlib.sha256(header).hexdigest()
            lines.append(text)
    raise ValueError("Skill frontmatter exceeds 64 KiB")


@dataclass(frozen=True, slots=True)
class LocalSkill:
    source: Path
    name: str
    description: str
    compatibility: str
    header_digest: str


@dataclass(frozen=True, slots=True)
class LocalCatalog:
    skills: dict[str, LocalSkill]
    diagnostics: tuple[str, ...]


def content_digest(source: Path) -> str | None:
    """Hash a bounded regular file; return None when equality cannot be checked."""
    try:
        info = source.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            return None
        with source.open("rb") as stream:
            content = stream.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            return None
        return hashlib.sha256(content).hexdigest()
    except OSError:
        return None


def discover(directories: tuple[tuple[Path, bool], ...]) -> LocalCatalog:
    """Return metadata and diagnostics; a broken skill does not hide its peers."""
    skills: dict[str, LocalSkill] = {}
    diagnostics: list[str] = []
    seen: dict[Path, int] = {}
    pending = [(path, 0, explicit) for path, explicit in reversed(directories)]
    while pending:
        directory, depth, explicit = pending.pop()
        try:
            if not directory.exists() and not explicit:
                continue
            directory = directory.resolve(strict=True)
            if directory in seen and seen[directory] <= depth:
                continue
            if directory not in seen and len(seen) >= MAX_DIRECTORIES:
                diagnostics.append("Local skill discovery stopped at the 2000-directory limit")
                break
            seen[directory] = depth
            children = sorted(directory.iterdir(), key=lambda item: item.name)
            source = next((item for item in children if item.name == "SKILL.md"), None)
            if source is not None:
                source = source.resolve(strict=True)
                if str(source) not in skills:
                    metadata, digest = read_header(source)
                    for key in ("name", "description"):
                        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                            raise ValueError(f"Skill requires a non-empty {key}")
                    compatibility = metadata.get("compatibility", "")
                    if not isinstance(compatibility, str):
                        raise ValueError("Skill compatibility must be text")
                    extra = metadata.get("metadata")
                    auto = extra.get("opencode/autoinvoke") if isinstance(extra, dict) else None
                    if metadata.get("disable-model-invocation") is True or auto in (False, "false"):
                        diagnostics.append(f"{source}: automatic invocation is disabled; skill not exposed")
                        continue
                    name = metadata["name"]
                    if (
                        name != source.parent.name
                        or len(name) > 64
                        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
                    ):
                        diagnostics.append(f"{source}: nonstandard skill name; loaded using its explicit path")
                    if len(metadata["description"]) > 1024:
                        diagnostics.append(f"{source}: description exceeds 1024 characters")
                    skills[str(source)] = LocalSkill(source, name, metadata["description"], compatibility, digest)
            subdirectories = [item for item in children if item.name not in SKIP_DIRECTORIES and item.is_dir()]
            if depth == MAX_DEPTH:
                if subdirectories:
                    diagnostics.append(f"{directory}: local skill discovery reached the depth limit ({MAX_DEPTH})")
            else:
                pending.extend((item, depth + 1, False) for item in reversed(subdirectories))
        except (OSError, ValueError, RuntimeError) as error:
            diagnostics.append(f"{directory}: {error}")
    names: dict[str, list[str]] = {}
    for location, skill in skills.items():
        names.setdefault(skill.name, []).append(location)
    for name, locations in names.items():
        if len(locations) > 1:
            digests = {content_digest(skills[location].source) for location in locations}
            if len(digests) == 1 and None not in digests:
                continue
            diagnostics.append(f"Duplicate local skill name {name!r}; use the explicit SKILL.md path")
    return LocalCatalog(skills, tuple(diagnostics))


class LocalSkills:
    """Keep one replaceable catalog for the local skill tool and system prompt."""

    def __init__(self, cwd: Path, *, directories: list[Path], isolated: bool = False) -> None:
        automatic = () if isolated else tuple((path, False) for path in default_directories())
        explicit = tuple((cwd / path.expanduser(), True) for path in directories)
        self.directories = (*automatic, *explicit)
        self.catalog = LocalCatalog({}, ())
        self.reader: Tool[Any] = Tool(name=LOCAL_SKILL_TOOL, handler=self.read, access=self.access)

    async def reload(self) -> None:
        self.catalog = await asyncio.to_thread(discover, self.directories)

    @property
    def tools(self) -> list[Tool[Any]]:
        return [self.reader] if self.catalog.skills else []

    def access(self, arguments: Mapping[str, Any]) -> tuple[Access, ...]:
        skill = self.catalog.skills.get(arguments["location"])
        if skill is None:
            raise ValueError("Unknown local skill location")
        return (
            Access("read", str(skill.source)),
            Access("read", str(skill.source.parent / arguments.get("path", "SKILL.md"))),
        )

    async def read(self, location: str, path: str = "SKILL.md") -> str:
        """Read a local skill or a UTF-8 file relative to its directory. Use a listed SKILL.md location."""
        skill = self.catalog.skills.get(location)
        if skill is None:
            raise ValueError("Unknown local skill location; use a path from the skill catalog")
        return await asyncio.to_thread(self.read_file, skill, path)

    @staticmethod
    def read_file(skill: LocalSkill, path: str) -> str:
        directory = skill.source.parent
        if Path(path).is_absolute():
            raise ValueError("Skill file paths must be relative to the skill directory")
        target = (directory / path).resolve(strict=True)
        if not target.is_relative_to(directory):
            raise ValueError("Skill file is outside the skill directory")
        if read_header(skill.source)[1] != skill.header_digest:
            raise ValueError("Skill metadata changed; use /skills reload before reading it")
        if not stat.S_ISREG(target.stat().st_mode):
            raise ValueError("Skill content must be a regular file")
        with target.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("Skill file exceeds 16 MiB")
        return json.dumps(
            {
                "source": str(skill.source),
                "directory": str(directory),
                "path": str(target),
                "text": data.decode("utf-8"),
            },
            ensure_ascii=False,
        )
