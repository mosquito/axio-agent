"""Contracts and commit handling for credential storage."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from axio.exceptions import AxioError


class CredentialError(AxioError):
    """An adapter could not read or commit credentials safely."""


class CredentialConflict(CredentialError):
    """Credentials changed while another operation was in progress."""


class CredentialStore[T](Protocol):
    """Storage contract implemented by the application, with any backend.

    lock() excludes all writers for one record key, including other
    processes and hosts that share these credentials. Keep it held across HTTP
    requests. A distributed lease must remain exclusive until exit; a lease
    that silently expires can cause refresh-token reuse.

    load() reads the latest record, never a stale process-local cache. save()
    atomically and durably replaces the whole record BEFORE returning. Each
    save is committed separately, even inside lock(); lock() is not a database
    transaction that rolls back writes on exit. Recovery markers must survive
    a process crash during the subsequent HTTP exchange.

    Call load/save under lock. Report failures as CredentialError with safe
    messages. Protect credentials through filesystem permissions, database
    access controls, or equivalent. Keep secrets out of logs and repr. Retain
    the host ID separately across restarts and sign-outs. Axio selects neither
    a backend nor a storage path.
    """

    def lock(self, key: str, /) -> AbstractAsyncContextManager[None]: ...

    async def load(self, key: str, /) -> T | None: ...

    async def save(self, record: T) -> None: ...


async def complete_operation[T](operation: Awaitable[T]) -> T:
    """Finish a credential transaction before delivering cancellation."""
    task = asyncio.ensure_future(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()
