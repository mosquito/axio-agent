"""Validate and serialize typed credential dataclasses without provider codecs."""

from __future__ import annotations

import hashlib
import json
import math
import types
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from .types import CredentialError


def record_id(record: object) -> str:
    """Find the one dataclass field marked with metadata={\"credential_key\": True}."""
    keys: list[object] = []

    def visit(value: object) -> None:
        if not is_dataclass(value) or isinstance(value, type):
            return
        for item in fields(value):
            child = getattr(value, item.name)
            if item.metadata.get("credential_key"):
                keys.append(child)
            else:
                visit(child)

    visit(record)
    if len(keys) != 1 or not isinstance(keys[0], str) or not keys[0]:
        raise CredentialError("Credentials require one nonempty string key.")
    return keys[0]


def record_key(key: str) -> str:
    """Hash record identifiers to keep them out of filesystem paths."""
    if not isinstance(key, str) or not key:
        raise CredentialError("A nonempty credential key is required.")
    # Keep the original prefix so existing stores use the same process locks.
    return "account-" + hashlib.sha256(key.encode()).hexdigest()


def json_value(value: Any) -> Any:
    """Convert dataclass containers to JSON values without stringifying secrets."""
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError


def restore(value: Any, annotation: Any) -> Any:
    """Restore declared JSON-compatible types without coercing invalid values."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return restore(value, args[0])
    if origin in (types.UnionType, Union):
        for member in args:
            try:
                return restore(value, member)
            except (TypeError, ValueError):
                pass
        raise ValueError
    if origin is Literal:
        if any(type(value) is type(choice) and value == choice for choice in args):
            return value
        raise ValueError
    if annotation is type(None) and value is None:
        return None
    if annotation in (str, int, bool) and type(value) is annotation:
        return value
    if annotation is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        for member in annotation:
            if type(value) is type(member.value) and value == member.value:
                return member
        raise ValueError
    if origin in (list, set, frozenset, tuple) and isinstance(value, list):
        if origin is tuple and (not args or args[-1] is not Ellipsis):
            if len(value) != len(args):
                raise ValueError
            return tuple(restore(item, kind) for item, kind in zip(value, args, strict=True))
        return origin(restore(item, args[0]) for item in value)
    if origin is dict and args[0] is str and isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError
        return {key: restore(item, args[1]) for key, item in value.items()}
    if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
        declared = {item.name for item in fields(annotation) if item.init}
        if value.keys() != declared:
            raise ValueError
        hints = get_type_hints(annotation, include_extras=True)
        return annotation(**{name: restore(value[name], hints[name]) for name in declared})
    raise ValueError


class CredentialCodec[T]:
    """Versioned dataclass records with safe errors and strict type checks."""

    def __init__(self, record_type: type[T]) -> None:
        if not isinstance(record_type, type) or not is_dataclass(record_type):
            raise TypeError("Credential records must be dataclasses.")
        self.record_type = record_type

    def from_dict(self, data: Any, key: str | None = None) -> T:
        try:
            if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
                raise ValueError
            record = cast(
                T, restore({name: value for name, value in data.items() if name != "version"}, self.record_type)
            )
            identity = record_id(record)
            if key is not None and identity != key:
                raise ValueError
            return record
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            raise CredentialError("Invalid stored credential record.") from None

    def to_dict(self, record: T) -> dict[str, Any]:
        try:
            if not isinstance(record, self.record_type) or not is_dataclass(record) or isinstance(record, type):
                raise ValueError
            data: dict[str, Any] = {"version": 1, **json_value(asdict(record))}
            self.from_dict(data, record_id(record))
            return data
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise CredentialError("Invalid credential record.") from None

    def decode(self, raw: str, key: str | None = None) -> T:
        try:
            return self.from_dict(json.loads(raw), key)
        except (ValueError, TypeError, RecursionError):
            raise CredentialError("Invalid stored credential record.") from None

    def encode(self, record: T) -> str:
        return json.dumps(self.to_dict(record), allow_nan=False, separators=(",", ":"))
