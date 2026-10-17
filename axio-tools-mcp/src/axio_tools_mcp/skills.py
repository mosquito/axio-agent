"""Read text skills through the MCP Skills compatibility resource bridge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from aiohttp_tiny_mcp.protocol.models import ReadResourceResult, ResourceDef, TextResourceContents
from pydantic import BaseModel, ConfigDict, Field

from .session import MCPSession

SKILLS_ROOT = "mcp-extensions://io.modelcontextprotocol/skills"
SKILLS_CATALOG = f"{SKILLS_ROOT}/skills"
SKILLS_LIST = f"{SKILLS_CATALOG}/list"
MAX_SKILL_BYTES = 16 * 1024 * 1024
MAX_SKILL_FILES = 512


class SkillResource(BaseModel):
    model_config = ConfigDict(strict=True)

    uri: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size: int = Field(ge=0)


class Skill(BaseModel):
    model_config = ConfigDict(strict=True)

    uri: str
    frontmatter: dict[str, Any]
    resources: list[SkillResource]

    @property
    def name(self) -> str:
        return str(self.frontmatter["name"])

    @property
    def description(self) -> str:
        return str(self.frontmatter["description"])

    @property
    def root(self) -> str:
        return self.uri.removesuffix("SKILL.md")

    def check(self) -> None:
        """Reject incomplete manifests before exposing the skill to the model."""
        if not self.uri.endswith("/SKILL.md"):
            raise ValueError("Skill URI must end in /SKILL.md")
        for key in ("name", "description"):
            if not isinstance(self.frontmatter.get(key), str) or not self.frontmatter[key].strip():
                raise ValueError(f"Skill frontmatter requires {key}")
        uris = {item.uri for item in self.resources}
        if len(uris) != len(self.resources) or self.uri not in uris:
            raise ValueError("Skill manifest must include SKILL.md and have unique resource URIs")
        if any(not uri.startswith(self.root) for uri in uris):
            raise ValueError("Skill resource is outside the skill directory")
        if len(uris) > MAX_SKILL_FILES or sum(item.size for item in self.resources) > MAX_SKILL_BYTES:
            raise ValueError("Skill exceeds the file count or size limit")


def resource_text(result: ReadResourceResult, uri: str) -> str:
    """Require one text resource so its exact UTF-8 bytes can be verified."""
    if len(result.contents) != 1:
        raise ValueError("Expected one text resource")
    content = result.contents[0]
    if not isinstance(content, TextResourceContents) or content.uri != uri:
        raise ValueError("Expected text content for the requested resource URI")
    if len(content.text.encode("utf-8")) > MAX_SKILL_BYTES:
        raise ValueError("Skill resource exceeds the size limit")
    return content.text


@dataclass(slots=True)
class SkillCatalog:
    """Keep skill identities and all reads bound to their originating session."""

    session: MCPSession
    skills: dict[str, Skill]

    @classmethod
    async def discover(cls, session: MCPSession, resources: list[ResourceDef] | None = None) -> SkillCatalog:
        catalog = cls(session, {})
        if "resources" not in session.server_capabilities:
            return catalog
        if resources is None:
            resources = await session.list_resources()
        uris = {resource.uri for resource in resources}
        if SKILLS_LIST in uris:
            uri = SKILLS_LIST
            cursors: set[str] = set()
            while True:
                page = json.loads(resource_text(await session.read_resource(uri), uri))
                if not isinstance(page, dict):
                    raise ValueError("Expected a skills/list result")
                catalog.add(page.get("skills"))
                cursor = page.get("nextCursor")
                if cursor is None:
                    break
                if not isinstance(cursor, str) or not cursor or cursor in cursors:
                    raise ValueError("Invalid skills/list cursor")
                cursors.add(cursor)
                uri = SKILLS_LIST + "?params=" + quote(json.dumps({"cursor": cursor}), safe="")
        elif SKILLS_CATALOG in uris:
            catalog.add(json.loads(resource_text(await session.read_resource(SKILLS_CATALOG), SKILLS_CATALOG)))
        return catalog

    def add(self, entries: Any) -> None:
        if not isinstance(entries, list):
            raise ValueError("Expected a skill catalog array")
        for entry in entries:
            skill = Skill.model_validate(entry)
            skill.check()
            if skill.uri in self.skills:
                raise ValueError("Duplicate skill URI")
            self.skills[skill.uri] = skill

    async def read(self, uri: str, path: str = "SKILL.md") -> str:
        """Read instructions or a manifest-listed text file relative to the skill root."""
        skill = self.skills.get(uri)
        if skill is None:
            raise ValueError("Unknown skill URI; use a URI from the skill catalog")
        target = skill.root + path
        resource = next((item for item in skill.resources if item.uri == target), None)
        if resource is None:
            raise ValueError("File is not in the skill manifest")
        text = resource_text(await self.session.read_resource(target), target)
        data = text.encode("utf-8")
        if len(data) != resource.size or "sha256:" + hashlib.sha256(data).hexdigest() != resource.digest:
            raise ValueError("Skill content changed or failed integrity verification; reload the MCP catalog")
        return json.dumps(
            {
                "server": self.session.config.name,
                "skill": skill.uri,
                "resource": resource.uri,
                "text": text,
                "files": [item.uri.removeprefix(skill.root) for item in skill.resources],
            },
            ensure_ascii=False,
        )
