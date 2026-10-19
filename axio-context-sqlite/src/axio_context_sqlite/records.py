"""Persistent message identity and snapshots for asynchronous archive consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from axio.messages import Message


class ContextConflictError(RuntimeError):
    """The record changed since the caller read its revision."""


@dataclass(frozen=True, slots=True)
class ContextRecord:
    """A current message snapshot; metadata belongs to the application."""

    message_id: str
    session_id: str
    project: str
    position: int
    message: Message = field(repr=False)
    status: str = "completed"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    revision: int = 1
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class ContextChange:
    """A committed invalidation and the current record, or a deletion tombstone.

    The record can be newer than this invalidation. This is a replayable
    invalidation journal, not a historical snapshot log. Apply the current
    record idempotently, then persist the sequence after the consumer commits.
    """

    sequence: int
    session_id: str
    project: str
    message_id: str
    record: ContextRecord | None = field(repr=False)


RECORD_COLUMNS = (
    "m.message_id, m.session_id, m.project, m.position, m.role, "
    "CASE WHEN m.id IS NOT NULL THEN decompress_payload(m.content) END, "
    "m.status, m.metadata, m.revision, m.created_at"
)


def decode_record(row: Sequence[Any]) -> ContextRecord:
    metadata = json.loads(row[7])
    if not isinstance(metadata, dict):
        raise ValueError("Stored context metadata must be a JSON object")
    return ContextRecord(
        message_id=row[0],
        session_id=row[1],
        project=row[2],
        position=row[3],
        message=Message.from_dict({"role": row[4], "content": json.loads(row[5])}),
        status=row[6],
        metadata=MappingProxyType(metadata),
        revision=row[8],
        created_at=row[9],
    )
