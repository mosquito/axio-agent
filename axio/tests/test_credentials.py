"""Shared credential storage works with unrelated provider data types."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, assert_type

import pytest

from axio.credentials import CredentialCodec, CredentialError, Credentials, FileCredentialStore, SQLiteCredentialStore


@dataclass(frozen=True, slots=True)
class APIKey:
    name: str = field(metadata={"credential_key": True})
    secret: str = field(repr=False)


class TokenKind(StrEnum):
    bearer = "bearer"


@dataclass(frozen=True, slots=True)
class Tokens:
    access: str = field(repr=False)
    scopes: frozenset[str]
    expires: float
    kind: TokenKind = TokenKind.bearer


@dataclass(frozen=True, slots=True)
class Registration:
    identity: str = field(metadata={"credential_key": True})
    tokens: Tokens | None = field(repr=False)
    revision: int = 0
    operation: Literal["renew", "revoke"] | None = None
    labels: dict[str, list[str]] = field(default_factory=dict)
    coordinates: tuple[int, str] = (1, "primary")
    flags: set[str] = field(default_factory=set)


class ServiceCredentials(FileCredentialStore[APIKey]):
    filename = "service.auth"


class InheritedCredentials(ServiceCredentials):
    pass


@pytest.mark.skipif(os.name != "posix", reason="Local credential backends require POSIX locks")
@pytest.mark.parametrize("backend", ["file", "sqlite"])
async def test_distinct_provider_types_round_trip(tmp_path: Path, backend: str) -> None:
    keys: Credentials[APIKey]
    registrations: Credentials[Registration]
    if backend == "file":
        keys = FileCredentialStore[APIKey](tmp_path / "key.auth")
        registrations = FileCredentialStore[Registration](tmp_path / "session.auth")
    else:
        keys = SQLiteCredentialStore[APIKey](tmp_path / "key.db")
        registrations = SQLiteCredentialStore[Registration](tmp_path / "session.db")
    key = APIKey("work", "secret-key")
    registration = Registration(
        "work",
        Tokens("secret-token", frozenset({"read", "write"}), 1000),
        operation="renew",
        labels={"regions": ["eu", "us"]},
        flags={"active"},
    )
    async with keys.lock("work"):
        await keys.save(key)
    async with registrations.lock("work"):
        await registrations.save(registration)
    await keys.select_account("work")
    assert await keys.selected_account() == "work"
    assert await registrations.selected_account() is None
    assert await keys.list_records() == [key]
    assert await registrations.list_records() == [registration]

    # Reopen with explicit record_type to cover both ways of retaining T at runtime.
    reopened: Credentials[Registration]
    if backend == "file":
        reopened = FileCredentialStore(tmp_path / "session.auth", record_type=Registration)
    else:
        reopened = SQLiteCredentialStore(tmp_path / "session.db", record_type=Registration)
    async with reopened.lock("work"):
        saved = await reopened.load("work")
        assert_type(saved, Registration | None)
        assert saved == registration
        assert saved is not None and isinstance(saved.tokens, Tokens)
        assert saved.tokens.kind is TokenKind.bearer
        await reopened.save(Registration("work", None))
    async with registrations.lock("work"):
        assert await registrations.load("work") == Registration("work", None)
    async with keys.lock("work"):
        assert await keys.load("work") == key


@pytest.mark.skipif(os.name != "posix", reason="Local credential backends require POSIX locks")
async def test_typed_subclass_only_supplies_type_and_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    store = InheritedCredentials()
    assert store.path == tmp_path / "axio" / "service.auth"
    async with store.lock("personal"):
        await store.save(APIKey("personal", "secret"))
    restarted = ServiceCredentials()
    async with restarted.lock("personal"):
        saved = await restarted.load("personal")
        assert_type(saved, APIKey | None)
        assert saved == APIKey("personal", "secret")


@pytest.mark.parametrize(
    "change",
    [
        "boolean_integer",
        "infinite_float",
        "wrong_secret",
        "wrong_scopes",
        "bad_literal",
        "bad_enum",
        "missing_field",
        "extra_field",
        "wrong_key",
        "boolean_version",
    ],
)
def test_invalid_nested_records_fail_without_disclosing_data(change: str) -> None:
    codec = CredentialCodec(Registration)
    value = codec.to_dict(Registration("work", Tokens("secret-token", frozenset({"read"}), 1000)))
    if change == "boolean_integer":
        value["revision"] = True
    elif change == "infinite_float":
        value["tokens"]["expires"] = float("inf")
    elif change == "wrong_secret":
        value["tokens"]["access"] = 123
    elif change == "wrong_scopes":
        value["tokens"]["scopes"] = "secret-token"
    elif change == "bad_literal":
        value["operation"] = "unknown"
    elif change == "bad_enum":
        value["tokens"]["kind"] = "unknown"
    elif change == "missing_field":
        del value["operation"]
    elif change == "extra_field":
        value["unexpected"] = "secret-token"
    elif change == "wrong_key":
        value["identity"] = "other"
    else:
        value["version"] = True
    with pytest.raises(CredentialError) as error:
        codec.decode(json.dumps(value), "work")
    assert "secret-token" not in str(error.value)


@dataclass(frozen=True, slots=True)
class ValidatedKey:
    name: str = field(metadata={"credential_key": True})
    secret: str = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.secret) < 8:
            raise ValueError(f"Invalid secret: {self.secret}")


def test_dataclass_validation_errors_are_sanitized() -> None:
    codec = CredentialCodec(ValidatedKey)
    with pytest.raises(CredentialError) as error:
        codec.from_dict({"version": 1, "name": "work", "secret": "hidden"})
    assert "hidden" not in str(error.value)


@pytest.mark.parametrize("key", ["", None, True, 42])
def test_record_keys_require_nonempty_strings(key: Any) -> None:
    with pytest.raises(CredentialError):
        CredentialCodec(APIKey).to_dict(APIKey(key, "secret"))


@pytest.mark.skipif(os.name != "posix", reason="Local credential backends require POSIX locks")
@pytest.mark.parametrize("value", [None, [], "secret"])
async def test_present_invalid_value_is_not_treated_as_missing(tmp_path: Path, value: Any) -> None:
    store = FileCredentialStore[APIKey](tmp_path / "keys.auth")
    # A malformed metadata value must not be silently replaced with a new host ID.
    store.directory.write(store.path.name, json.dumps({"version": 1, "values": {"host": value}}))
    with pytest.raises(CredentialError):
        await store.host_id()
