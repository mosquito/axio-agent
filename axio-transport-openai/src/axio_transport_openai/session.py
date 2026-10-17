"""Pinned ChatGPT sessions with coordinated foreground and background renewal."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, replace
from types import TracebackType

from axio.exceptions import StreamError
from axio_responses import ResponsesAPIError

from .credentials import CredentialError, CredentialRecord, CredentialStore, complete_operation
from .oauth import TERMINAL_REFRESH_ERRORS, ChatGPTAccount, ChatGPTOAuth, OAuthError, OAuthValidationError


class SignInRequired(OAuthError, StreamError):
    """Reauthorize the same client registration to restore this session."""


class PlanConsentRequired(OAuthError, StreamError):
    """The account did not grant permission to use its ChatGPT plan."""


@dataclass(frozen=True, slots=True)
class SignOutResult:
    """Local tokens were cleared. False means remote revocation is unconfirmed."""

    revocation_confirmed: bool


class ChatGPTSession:
    """Keep one issued client registration current for the lifetime of a run.

    Use as an async context manager to enable background renewal. Without it,
    access_token() still renews on demand. close() stops background work without
    signing out. The caller owns the OAuth client's aiohttp session and must
    keep it open until close() finishes. last_error reports background failures.
    """

    def __init__(
        self,
        store: CredentialStore,
        oauth: ChatGPTOAuth,
        client_id: str,
        *,
        refresh_margin: float = 60,
        retry_delay: float = 5,
    ) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (refresh_margin, retry_delay)):
            raise ValueError("Refresh margin and retry delay must be finite and positive.")
        self.store = store
        self.oauth = oauth
        self._client_id = client_id
        self.refresh_margin = refresh_margin
        self.retry_delay = retry_delay
        self.last_error: OAuthError | CredentialError | None = None
        self.plan_usage_error: ResponsesAPIError | None = None
        self._worker: asyncio.Task[None] | None = None
        self._starting = asyncio.Lock()
        self._requests: set[asyncio.Task[ChatGPTAccount]] = set()
        self._stop = asyncio.Event()
        self._closed = False
        self._signed_out = False
        self._pending: tuple[str, CredentialRecord] | None = None

    @property
    def client_id(self) -> str:
        """The registration pinned when this session was created."""
        return self._client_id

    def resume_plan_usage(self) -> None:
        """Allow another plan request after the caller resolves a reported usage limit.

        A quota response provides no reliable reset time. Resuming is explicit;
        it does not change accounts, credentials, or the billing path.
        """
        self.plan_usage_error = None

    async def __aenter__(self) -> ChatGPTSession:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def start(self) -> None:
        """Start background renewal after checking the selected registration."""
        async with self._starting:
            self._check_open()
            if self._worker is not None and not self._worker.done():
                return
            await self.access_token()
            self._check_open()
            self._worker = asyncio.create_task(self._run(), name="chatgpt-token-renewal")

    async def close(self) -> None:
        """Stop renewal and finish any credential commit already in progress."""
        self._closed = True
        self._stop.set()
        await complete_operation(self._drain())

    async def _drain(self) -> None:
        async with self._starting:
            pass
        if self._worker is not None:
            await self._worker
        if self._requests:
            await asyncio.gather(*self._requests, return_exceptions=True)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("The ChatGPT session is closed.")
        if self._signed_out:
            raise SignInRequired("This session has signed out.")

    async def access_token(self) -> str:
        """Get a current plan token, reloading credentials under the process lock."""
        self._check_open()
        task = asyncio.create_task(self._current())
        self._requests.add(task)
        try:
            account = await complete_operation(task)
        finally:
            self._requests.discard(task)
        assert account.tokens is not None and account.tokens.access_token is not None
        return account.tokens.access_token

    def _refresh_at(self, account: ChatGPTAccount) -> float:
        tokens = account.tokens
        assert tokens is not None and tokens.expires_at is not None
        # Short-lived tokens still get a positive interval before the next refresh.
        margin = min(self.refresh_margin, max(0, (tokens.expires_at - tokens.received_at) / 2))
        return tokens.expires_at - margin

    async def _current(self) -> ChatGPTAccount:
        async with self.store.lock(self.client_id):
            self._check_open()
            record = await self.store.load(self.client_id)
            if record is None:
                raise SignInRequired("The account registration was not found.")
            if record.account.client_id != self.client_id:
                raise CredentialError("The stored registration does not match the selected account.")
            account = record.account
            self.oauth._check_account(account)
            if self._pending is not None:
                revision, pending = self._pending
                if record.revision == revision:
                    await self.store.save(pending)
                    record = pending
                    account = record.account
                self._pending = None
            if record.operation is not None:
                raise SignInRequired("A credential operation was interrupted. Sign in to the same account again.")
            tokens = account.tokens
            if tokens is None:
                raise SignInRequired("Sign in to the selected account.")
            if not tokens.plan_enabled:
                raise PlanConsentRequired("The selected account needs ChatGPT plan consent.")
            if tokens.access_token is None or tokens.expires_at is None:
                raise SignInRequired("The selected account has no usable access token.")
            if time.time() < self._refresh_at(account):
                self.last_error = None
                return account
            if tokens.refresh_token is None:
                raise SignInRequired("The selected account has no renewable session.")
            marker = CredentialRecord(account, operation="refresh")
            await self.store.save(marker)
            try:
                updated = await self.oauth.refresh(account)
            except OAuthError as exc:
                self.last_error = exc
                if not exc.request_not_sent and exc.code in TERMINAL_REFRESH_ERRORS and exc.status in (400, 401):
                    await self.store.save(CredentialRecord(replace(account, tokens=None)))
                    raise SignInRequired("The renewable session ended. Sign in to the same account again.") from None
                if not exc.request_not_sent and (isinstance(exc, OAuthValidationError) or exc.status is None):
                    # A lost or invalid response may hide a successful rotation.
                    # Keep the marker; never replay a possibly spent token.
                    raise
                await self.store.save(CredentialRecord(account))
                if exc.temporary and time.time() < tokens.expires_at:
                    return account
                raise
            except TimeoutError:
                # A deadline after sending gives no evidence that rotation failed.
                error = OAuthError("OAuth refresh timed out.")
                self.last_error = error
                raise error from None
            replacement = CredentialRecord(updated)
            # Keep the replacement in memory until its atomic write completes.
            # Other processes see the marker and cannot reuse the spent token.
            self._pending = (marker.revision, replacement)
            await self.store.save(replacement)
            self._pending = None
            self.last_error = None
            if updated.tokens is None or not updated.tokens.plan_enabled:
                raise PlanConsentRequired("ChatGPT plan consent is no longer granted.")
            return updated

    async def _run(self) -> None:
        delay = 0.0
        failures = 0
        while not self._stop.is_set():
            if delay:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    return
                except TimeoutError:
                    pass
            if self._stop.is_set():
                return
            try:
                account = await complete_operation(self._current())
                if self.last_error is not None:
                    failures += 1
                    delay = min(60, self.retry_delay * 2 ** min(failures - 1, 4))
                else:
                    failures = 0
                    delay = max(0.05, self._refresh_at(account) - time.time())
            except OAuthError as exc:
                self.last_error = exc
                if not exc.temporary or isinstance(exc, (SignInRequired, PlanConsentRequired)):
                    return
                failures += 1
                delay = min(60, self.retry_delay * 2 ** min(failures - 1, 4))
            except CredentialError as exc:
                self.last_error = exc
                # A pending rotated set can be committed without another exchange.
                if self._pending is None:
                    return
                failures += 1
                delay = min(60, self.retry_delay * 2 ** min(failures - 1, 4))
            except RuntimeError:
                if not self._closed:
                    raise
                return

    async def sign_out(self) -> SignOutResult:
        """Revoke the latest renewable session and retain its registration.

        Temporary revocation errors get two retries. Local tokens are cleared
        even if the server cannot confirm revocation. A storage failure raises
        CredentialError; it never reports successful local sign-out.
        """
        self._signed_out = True
        self._stop.set()
        return await complete_operation(self._sign_out())

    async def _sign_out(self) -> SignOutResult:
        if self._worker is not None:
            await self._worker
        async with self.store.lock(self.client_id):
            record = await self.store.load(self.client_id)
            if record is None:
                raise SignInRequired("The account registration was not found.")
            if record.account.client_id != self.client_id:
                raise CredentialError("The stored registration does not match the selected account.")
            self.oauth._check_account(record.account)
            account = record.account
            known_token = record.operation != "refresh"
            if self._pending is not None and self._pending[0] == record.revision:
                account = self._pending[1].account
                known_token = True
            marker = CredentialRecord(account, operation="logout" if known_token else "refresh")
            await self.store.save(marker)
            self._pending = None
            confirmed = False
            if account.tokens is not None and account.tokens.refresh_token is not None:
                for attempt in range(3):
                    try:
                        await self.oauth.revoke(account)
                        confirmed = known_token
                        break
                    except OAuthError as exc:
                        if not exc.temporary or attempt == 2:
                            break
                        await asyncio.sleep(self.retry_delay * 2**attempt)
            await self.store.save(CredentialRecord(replace(account, tokens=None)))
            return SignOutResult(confirmed)
