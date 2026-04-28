"""Field metadata for tool handler functions - lightweight replacement for pydantic.Field."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, get_args, get_origin

# Types for which basic isinstance checks are applied in non-strict mode.
_PRIMITIVE_TYPES: frozenset[type] = frozenset({str, int, float, bool, list, dict})


class MissingSentinel:
    """Singleton sentinel - distinguishes 'no default' from ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


#: Sentinel value meaning "this field has no default and is required".
MISSING: Final[MissingSentinel] = MissingSentinel()


@dataclass(frozen=True)
class FieldInfo:
    """Metadata attached to a handler parameter via ``Annotated[T, FieldInfo(...)]``."""

    description: str = ""
    default: Any = MISSING
    ge: int | float | None = None  # minimum (≥)
    le: int | float | None = None  # maximum (≤)
    strict: bool = False  # reject implicit type coercion

    def validate(self, value: Any, name: str, hint: Any) -> None:
        """Validate *value* against this FieldInfo's constraints, raising if invalid."""
        # Unwrap Annotated to reach the real type annotation.
        inner = get_args(hint)[0] if get_origin(hint) is typing.Annotated else hint
        # Unwrap Optional / Union to find the base type.
        inner_origin = get_origin(inner)
        is_optional = False
        if inner_origin is types.UnionType or inner_origin is typing.Union:
            inner_args = get_args(inner)
            non_none = [a for a in inner_args if a is not type(None)]
            is_optional = len(non_none) < len(inner_args)
            if len(non_none) == 1:
                inner = non_none[0]
                inner_origin = get_origin(inner)

        # None is valid for Optional fields; no further checks needed.
        if value is None and is_optional:
            return

        # Literal: value must be one of the declared constants.
        if inner_origin is Literal:
            allowed = get_args(inner)
            if value not in allowed:
                raise ValueError(f"Field '{name}' must be one of {list(allowed)!r}, got {value!r}")
        elif self.strict:
            b = bare_type(hint)
            if not isinstance(value, b):
                raise TypeError(f"Field '{name}' requires {b.__name__}, got {type(value).__name__}")
        else:
            # Non-strict: isinstance check for known primitive types.
            # None is accepted for Optional fields (handled above).
            # int is accepted for float (JSON "number" covers both).
            b = bare_type(hint)
            if b is float and isinstance(value, int):
                pass  # valid JSON number
            elif b in _PRIMITIVE_TYPES and value is not None and not isinstance(value, b):
                raise TypeError(f"Field '{name}' requires {b.__name__}, got {type(value).__name__}")

        if self.ge is not None and value < self.ge:
            raise ValueError(f"Field '{name}' must be >= {self.ge}")
        if self.le is not None and value > self.le:
            raise ValueError(f"Field '{name}' must be <= {self.le}")


# noinspection PyPep8Naming
def Field(
    description: str = "",
    default: Any = MISSING,
    ge: int | float | None = None,
    le: int | float | None = None,
) -> FieldInfo:
    """Annotate a handler parameter with metadata (description, default, constraints).

    Usage::

        async def search(
            query: Annotated[str, Field(description="Search query")],
            limit: Annotated[int, Field(default=10, ge=1, le=100)],
        ) -> str: ...
    """
    return FieldInfo(description=description, default=default, ge=ge, le=le)


#: Drop-in replacement for ``from pydantic import StrictStr``.
#: Rejects non-string values (e.g. integers) without coercion.
StrictStr = Annotated[str, FieldInfo(strict=True)]


def is_classvar(annotation: Any) -> bool:
    """Return True if *annotation* is ``ClassVar`` or ``ClassVar[X]``."""
    return get_origin(annotation) is typing.ClassVar or annotation is typing.ClassVar


def get_field_info(annotation: Any) -> FieldInfo | None:
    """Extract a ``FieldInfo`` from an ``Annotated`` annotation, or return ``None``."""
    if get_origin(annotation) is not typing.Annotated:
        return None
    return next((a for a in get_args(annotation)[1:] if isinstance(a, FieldInfo)), None)


def bare_type(hint: Any) -> type:
    """Return the base Python type, stripping ``Annotated``, ``Optional``, and generic wrappers."""
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is typing.Annotated:
        return bare_type(args[0])
    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        return bare_type(non_none[0]) if len(non_none) == 1 else object
    if origin is not None:
        # Generic alias: list[int] → list, dict[str, Any] → dict, etc.
        return origin if isinstance(origin, type) else object
    return hint if isinstance(hint, type) else object
