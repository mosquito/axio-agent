"""Skills compatibility resources for local HTTP and stdio integration tests."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from aiohttp_tiny_mcp import Registry, run_stdio
from axio_tools_mcp.skills import SKILLS_CATALOG, SKILLS_ROOT
from pydantic import BaseModel

SKILL_URI = "skill://planning/SKILL.md"
INSTRUCTIONS = "---\nname: planning\ndescription: Plan a task\n---\nRead references/details.md.\n"
REFERENCE = "Keep user requirements. Проверка UTF-8.\n"


class Nothing(BaseModel):
    pass


class Topic(BaseModel):
    name: str


def skill_entry() -> dict[str, Any]:
    return {
        "uri": SKILL_URI,
        "frontmatter": {"name": "planning", "description": "Plan a task"},
        "resources": [
            {"uri": uri, "digest": "sha256:" + hashlib.sha256(text.encode()).hexdigest(), "size": len(text.encode())}
            for uri, text in ((SKILL_URI, INSTRUCTIONS), ("skill://planning/references/details.md", REFERENCE))
        ],
    }


def skill_registry(reads: list[str], *, broken: bool = False, direct_skills: bool = True) -> Registry:
    registry = Registry("skills-fixture", "1")

    @registry.tool
    async def ping(args: Nothing) -> str:
        return "pong"

    async def skills(args: Nothing) -> list[dict[str, Any]]:
        entry = skill_entry()
        if broken:
            entry.pop("resources")
        return [entry]

    async def instructions(args: Nothing) -> str:
        reads.append("SKILL.md")
        return INSTRUCTIONS

    async def reference(args: Nothing) -> str:
        reads.append("references/details.md")
        return REFERENCE

    async def guide(args: Nothing) -> str:
        return "A plain MCP resource."

    async def topic(args: Topic) -> str:
        return f"Topic: {args.name}"

    registry.resource(SKILLS_CATALOG)(skills)
    if direct_skills:
        registry.resource(SKILL_URI)(instructions)
        registry.resource("skill://planning/references/details.md")(reference)
    registry.resource(f"{SKILLS_ROOT}/planning/SKILL.md")(instructions)
    registry.resource(f"{SKILLS_ROOT}/planning/references/details.md")(reference)
    registry.resource("notes://guide", description="A general guide")(guide)
    registry.resource("notes://topics/{name}", description="Look up a topic")(topic)
    return registry


if __name__ == "__main__":
    asyncio.run(run_stdio(skill_registry([])))
