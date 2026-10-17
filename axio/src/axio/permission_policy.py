"""Scoped tool permissions and serialized approval requests, without a UI dependency."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from .exceptions import GuardError
from .permission import PermissionGuard
from .tool import Tool
from .tool_access import Access

type Decision = Literal["allow", "ask", "deny"]
type Lifetime = Literal["once", "chat", "project"]


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool: str
    accesses: tuple[Access, ...]
    arguments: str
    signature: str
    cwd: str
    reason: str = "No matching permission rule."

    @property
    def files_only(self) -> bool:
        return bool(self.accesses) and all(
            a.path is not None and a.operation in {"read", "write"} for a in self.accesses
        )

    def describe_scope(self, *, directories: bool = False) -> str:
        if not self.files_only:
            return f"Exact call to {self.tool} in {self.cwd}:\n{self.arguments}"
        return "\n".join(
            f"{a.operation}: {Path(a.path).parent if directories else a.path}"
            + (" (including descendants)" if directories else " (exact path)")
            for a in self.accesses
            if a.path is not None
        )


@dataclass(frozen=True, slots=True)
class Approval:
    decision: Literal["allow", "deny"] = "deny"
    lifetime: Lifetime = "once"
    directories: bool = False

    def __post_init__(self) -> None:
        if self.decision not in {"allow", "deny"} or self.lifetime not in {"once", "chat", "project"}:
            raise ValueError("Invalid approval decision or lifetime")
        if not isinstance(self.directories, bool):
            raise ValueError("Invalid directory scope")
        if self.lifetime == "once" and self.directories:
            raise ValueError("One-time approval cannot grant directory access")


@dataclass(frozen=True, slots=True)
class PermissionRule:
    id: str
    tool: str
    accesses: tuple[Access, ...]
    signature: str | None
    decision: Literal["allow", "deny"]
    lifetime: Literal["chat", "project"]
    directories: bool = False

    @classmethod
    def from_request(cls, request: PermissionRequest, answer: Approval) -> PermissionRule:
        if answer.lifetime == "once":
            raise ValueError("A one-time answer is not a rule")
        if answer.directories and not request.files_only:
            raise ValueError("Directory rules require declared file operations")
        accesses = request.accesses
        if answer.directories:
            accesses = tuple(replace(a, path=str(Path(a.path).parent)) for a in accesses if a.path is not None)
        return cls(
            uuid4().hex,
            request.tool,
            accesses,
            None if request.files_only else request.signature,
            answer.decision,
            answer.lifetime,
            answer.directories,
        )

    def matches(self, request: PermissionRequest) -> bool:
        if self.tool != request.tool:
            return False
        if self.signature is not None:
            return self.signature == request.signature
        if not request.files_only:
            return False

        def covered(access: Access) -> bool:
            return any(
                a.operation == access.operation
                and a.path is not None
                and access.path is not None
                and (a.path == access.path or (self.directories and Path(access.path).is_relative_to(a.path)))
                for a in self.accesses
            )

        # A denial of one resource blocks a call that touches several resources.
        return any(map(covered, request.accesses)) if self.decision == "deny" else all(map(covered, request.accesses))

    def describe(self) -> str:
        scope = "\n".join(f"{a.operation}: {a.path}" for a in self.accesses)
        if self.signature is not None:
            scope = f"Exact call {self.signature[:12]}"
        elif self.directories:
            scope += "\nIncluding descendants"
        return f"{self.decision.upper()} {self.tool} — {self.lifetime}\n{scope}"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, value: str) -> PermissionRule:
        data = json.loads(value)
        if not isinstance(data, dict) or set(data) != {
            "id",
            "tool",
            "accesses",
            "signature",
            "decision",
            "lifetime",
            "directories",
        }:
            raise ValueError("Invalid permission rule")
        if (
            not isinstance(data["id"], str)
            or not data["id"]
            or not isinstance(data["tool"], str)
            or not data["tool"]
            or data["decision"] not in {"allow", "deny"}
            or data["lifetime"] != "project"
            or not isinstance(data["directories"], bool)
            or not isinstance(data["accesses"], list)
        ):
            raise ValueError("Invalid permission rule fields")
        accesses = []
        for item in data["accesses"]:
            if not isinstance(item, dict) or set(item) != {"operation", "path"}:
                raise ValueError("Invalid permission access")
            if item["operation"] not in {"read", "write", "execute", "external", "unknown"}:
                raise ValueError("Invalid permission operation")
            path = item["path"]
            if path is not None and (not isinstance(path, str) or not Path(path).is_absolute()):
                raise ValueError("Permission paths must be absolute")
            accesses.append(Access(**item))
        signature = data["signature"]
        if signature is not None:
            if not isinstance(signature, str) or len(signature) != 64 or data["directories"]:
                raise ValueError("Invalid exact-call rule")
        elif not accesses or any(a.path is None or a.operation not in {"read", "write"} for a in accesses):
            raise ValueError("File rules require file accesses")
        return cls(**{**data, "accesses": tuple(accesses)})


class RuleStore(Protocol):
    async def load(self) -> list[PermissionRule]: ...
    async def add(self, rule: PermissionRule) -> None: ...
    async def remove(self, rule_id: str) -> None: ...


type Prompt = Callable[[PermissionRequest], Awaitable[Approval]]
type Advisor = Callable[[Tool[Any], Mapping[str, Any]], Awaitable[str]]


class PolicyGuard(PermissionGuard):
    """Apply explicit rules, allow project-local reads, and ask for other calls.

    The lock serializes prompts, not tool execution. Waiting calls re-evaluate
    rules after acquiring it. Callers must attach this guard after guards that
    can change arguments. Filesystem checks are preflight checks, not isolation
    against concurrent external changes to the filesystem.
    """

    def __init__(
        self, project_root: Path, prompt: Prompt, *, store: RuleStore | None = None, advisor: Advisor | None = None
    ) -> None:
        self.project_root = project_root.resolve()
        self.prompt = prompt
        self.store = store
        self.advisor = advisor
        self._chat_rules: list[PermissionRule] = []
        self._session = ""
        self._generation = 0
        self._lock = asyncio.Lock()

    def set_session(self, session: str) -> None:
        if session != self._session:
            self._session = session
            self._generation += 1
            self._chat_rules.clear()

    async def rules(self) -> list[PermissionRule]:
        try:
            saved = await self.store.load() if self.store is not None else []
        except Exception as error:
            raise GuardError(
                "Cannot load permission rules; repair or remove the invalid rule in Permissions."
            ) from error
        return [*saved, *self._chat_rules]

    async def revoke(self, rule_id: str) -> None:
        if self.store is not None:
            await self.store.remove(rule_id)
        self._chat_rules = [rule for rule in self._chat_rules if rule.id != rule_id]

    def describe(self, tool: Tool[Any], kwargs: Mapping[str, Any]) -> PermissionRequest:
        cwd = Path.cwd().resolve()
        declared = tool.access(kwargs) if tool.access is not None else (Access("unknown"),)
        accesses = tuple(
            dict.fromkeys(
                replace(a, path=str((cwd / a.path).resolve())) if a.path is not None else a for a in declared
            )
        )
        arguments = json.dumps(dict(kwargs), sort_keys=True, ensure_ascii=False, allow_nan=False)
        # Opaque effects are bound to every argument, the working directory, and
        # provider identity. A renamed/replaced declaration cannot reuse the call.
        signature = hashlib.sha256(
            json.dumps(
                [
                    tool.name,
                    tool.permission_identity,
                    tool.handler.__module__,
                    tool.handler.__qualname__,
                    tool.description,
                    tool.input_schema,
                    str(cwd),
                    arguments,
                    [asdict(a) for a in accesses],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return PermissionRequest(tool.name, accesses, arguments, signature, str(cwd))

    async def evaluate(self, request: PermissionRequest) -> tuple[Decision, str]:
        matches = [rule for rule in await self.rules() if rule.matches(request)]
        for rule in matches:
            if rule.decision == "deny":
                return "deny", f"Denied by {rule.lifetime} rule {rule.id}."
        if matches:
            return "allow", f"Allowed by {matches[0].lifetime} rule {matches[0].id}."
        if not request.accesses:
            return "allow", "Provider declares no external effects."
        if all(
            a.operation == "read" and a.path is not None and Path(a.path).is_relative_to(self.project_root)
            for a in request.accesses
        ):
            return "allow", "Read-only access inside the project."
        return "ask", "This call is not covered by project-local read access or an explicit rule."

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        generation = self._generation
        request = await asyncio.to_thread(self.describe, tool, kwargs)
        decision, reason = await self.evaluate(request)
        if generation != self._generation:
            raise GuardError("Conversation changed while checking permission")
        if decision == "deny":
            raise GuardError(reason)
        if decision == "allow":
            return kwargs
        async with self._lock:
            while True:
                if generation != self._generation:
                    raise GuardError("Conversation changed while awaiting permission")
                request = await asyncio.to_thread(self.describe, tool, kwargs)
                decision, reason = await self.evaluate(request)
                if generation != self._generation:
                    raise GuardError("Conversation changed while checking permission")
                if decision == "deny":
                    raise GuardError(reason)
                if decision == "allow":
                    return kwargs
                if self.advisor is not None:
                    try:
                        advice = await self.advisor(tool, kwargs)
                    except Exception:
                        advice = "Review unavailable; explicit permission is required."
                    if advice:
                        reason += f"\nReview: {advice}"
                answer = await self.prompt(replace(request, reason=reason))
                if not isinstance(answer, Approval):
                    raise GuardError("Permission prompt did not return a decision")
                if generation != self._generation:
                    raise GuardError("Conversation changed while awaiting permission")
                # A symlink or cwd can change while the dialog is open. The answer
                # covers only the request the user saw. Ask again for changed effects.
                current = await asyncio.to_thread(self.describe, tool, kwargs)
                if current != request:
                    continue
                decision, reason = await self.evaluate(current)
                if generation != self._generation:
                    raise GuardError("Conversation changed while checking permission")
                if decision == "deny":
                    raise GuardError(reason)
                if answer.lifetime != "once":
                    rule = PermissionRule.from_request(request, answer)
                    if rule.lifetime == "project":
                        if self.store is None:
                            raise GuardError("Project permission storage is unavailable")
                        await self.store.add(rule)
                    else:
                        self._chat_rules.append(rule)
                if generation != self._generation:
                    raise GuardError("Conversation changed while saving permission")
                if answer.decision != "allow":
                    raise GuardError(f"User denied {tool.name}")
                return kwargs
