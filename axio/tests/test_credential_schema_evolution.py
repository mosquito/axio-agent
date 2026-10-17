"""Credential schema additions preserve old records without hiding missing state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from axio.credentials import CredentialCodec, CredentialError


@dataclass(frozen=True, slots=True)
class Account:
    key: str = field(metadata={"credential_key": True})
    native: bool = field(default=False, metadata={"credential_optional": True})
    tags: list[str] = field(default_factory=list, metadata={"credential_optional": True})


@dataclass(frozen=True, slots=True)
class Record:
    account: Account
    operation: str | None = None


def test_explicit_optional_fields_restore_defaults_and_round_trip() -> None:
    codec = CredentialCodec(Record)
    old = {"version": 1, "account": {"key": "account"}, "operation": "refresh"}
    restored = codec.from_dict(old)
    assert restored == Record(Account("account"), operation="refresh")
    assert codec.decode(codec.encode(restored)) == restored
    assert codec.from_dict(old).account.tags is not restored.account.tags
    assert old["account"] == {"key": "account"}


@pytest.mark.parametrize(
    "data",
    [
        {"version": 1, "account": {"key": "account"}},
        {"version": 1, "account": {"key": "account", "unknown": False}, "operation": None},
        {"version": 1, "account": {"key": "account", "native": "false"}, "operation": None},
        {"version": 1, "account": {"key": "account", "native": None}, "operation": None},
        {"version": 1, "account": {"key": "account", "tags": [1]}, "operation": None},
        {"version": 1, "account": {}, "operation": None},
    ],
)
def test_optional_fields_do_not_weaken_record_validation(data: dict[str, Any]) -> None:
    with pytest.raises(CredentialError):
        CredentialCodec(Record).from_dict(data)


def test_optional_metadata_without_default_does_not_remove_requirement() -> None:
    @dataclass(frozen=True, slots=True)
    class Required:
        key: str = field(metadata={"credential_key": True, "credential_optional": True})

    with pytest.raises(CredentialError):
        CredentialCodec(Required).from_dict({"version": 1})
