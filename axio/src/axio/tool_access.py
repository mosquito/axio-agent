"""Provider declarations of tool effects, independent of permission policy and UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Access:
    """One effect. Paths describe local files; other effects require exact-call approval.

    A recursive read must enumerate any symlink targets it can follow as separate
    accesses. This declaration is a provider contract, not an OS sandbox.
    """

    operation: Literal["read", "write", "execute", "external", "unknown"]
    path: str | None = None


type AccessDescriptor = Callable[[Mapping[str, Any]], tuple[Access, ...]]


def tool_access[F: Callable[..., Any]](describe: AccessDescriptor) -> Callable[[F], F]:
    """Attach a declaration to an entry-point handler without wrapping it.

    An explicit empty tuple means the handler has no external effects. An absent
    declaration means unknown effects. Descriptors must not perform tool actions.
    """

    def decorate(handler: F) -> F:
        setattr(handler, "__axio_access__", describe)
        return handler

    return decorate


def file_access(
    operation: Literal["read", "write"], parameter: str, *, default: str | None = None
) -> AccessDescriptor:
    """Declare a single file or directory parameter owned by this provider."""

    def describe(arguments: Mapping[str, Any]) -> tuple[Access, ...]:
        path = arguments.get(parameter, default)
        if not isinstance(path, str):
            raise ValueError(f"Missing or invalid path: {parameter}")
        return (Access(operation, path),)

    return describe
