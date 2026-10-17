"""ChatGPT credential data and registration identity checks."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Literal

from axio.credentials import CredentialConflict as CredentialConflict
from axio.credentials import CredentialError as CredentialError
from axio.credentials import CredentialStore as Store
from axio.credentials import complete_operation as complete_operation

from .oauth import DYNAMIC_CLIENT_ID, PLAN_SCOPE, ChatGPTAccount, ChatGPTTokens


@dataclass(frozen=True, slots=True)
class CredentialRecord:
    """One complete ChatGPT account, including durable recovery state."""

    account: ChatGPTAccount = field(repr=False)
    revision: str = field(default_factory=lambda: uuid.uuid4().hex)
    operation: Literal["refresh", "logout"] | None = None

    def __post_init__(self) -> None:
        try:
            account = self.account
            if not isinstance(account, ChatGPTAccount):
                raise ValueError
            for value in (account.client_id, account.host_id, account.issuer, account.subject):
                if not isinstance(value, str) or not value:
                    raise ValueError
            if account.client_id == DYNAMIC_CLIENT_ID:
                raise ValueError
            if account.codex_account_id is not None and (
                not isinstance(account.codex_account_id, str)
                or not account.codex_account_id
                or account.client_id != f"codex:{account.codex_account_id}:{account.subject}"
            ):
                raise ValueError
            for optional in (account.email, account.name):
                if optional is not None and not isinstance(optional, str):
                    raise ValueError
            tokens = account.tokens
            if tokens is not None:
                if (
                    not isinstance(tokens, ChatGPTTokens)
                    or not isinstance(tokens.id_token, str)
                    or not tokens.id_token
                ):
                    raise ValueError
                if type(tokens.codex) is not bool or tokens.codex != (account.codex_account_id is not None):
                    raise ValueError
                if not isinstance(tokens.scopes, frozenset) or not all(
                    isinstance(scope, str) and scope for scope in tokens.scopes
                ):
                    raise ValueError
                for optional in (tokens.access_token, tokens.refresh_token, tokens.token_type):
                    if optional is not None and (not isinstance(optional, str) or not optional):
                        raise ValueError
                for timestamp in (tokens.received_at, tokens.expires_at):
                    if timestamp is not None and (
                        isinstance(timestamp, bool)
                        or not isinstance(timestamp, (int, float))
                        or not math.isfinite(timestamp)
                    ):
                        raise ValueError
                if tokens.received_at is None:
                    raise ValueError
                if tokens.access_token is not None:
                    if (
                        tokens.expires_at is None
                        or tokens.expires_at <= tokens.received_at
                        or tokens.token_type is None
                        or tokens.token_type.lower() != "bearer"
                    ):
                        raise ValueError
                elif tokens.expires_at is not None or PLAN_SCOPE in tokens.scopes or tokens.codex:
                    raise ValueError
            if (
                not isinstance(self.revision, str)
                or not self.revision
                or self.operation not in (None, "refresh", "logout")
            ):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise CredentialError("Invalid credential record.") from None

    def public_account(self) -> ChatGPTAccount:
        """Return metadata without possibly spent tokens after an interruption."""
        return replace(self.account, tokens=None) if self.operation else self.account


CredentialStore = Store[CredentialRecord]


async def save_account(
    store: CredentialStore, account: ChatGPTAccount, *, expected: CredentialRecord | None = None
) -> None:
    """Persist a validated sign-in without overwriting a concurrent update.

    For returning sign-in, pass the record loaded before browser authorization
    as expected. Reauthorize expected.public_account() to retain its identity.
    """
    await complete_operation(commit_account(store, account, expected))


async def commit_account(store: CredentialStore, account: ChatGPTAccount, expected: CredentialRecord | None) -> None:
    if not account.client_id or account.client_id == DYNAMIC_CLIENT_ID or not account.subject:
        raise CredentialError("A verified account with an issued client ID is required.")
    async with store.lock(account.client_id):
        current = await store.load(account.client_id)
        if current != expected:
            raise CredentialConflict("The account changed before sign-in could be saved.")
        if current and (
            current.account.issuer,
            current.account.host_id,
            current.account.subject,
        ) != (account.issuer, account.host_id, account.subject):
            raise CredentialConflict("An issued client ID cannot change identity or host.")
        await store.save(CredentialRecord(account))
