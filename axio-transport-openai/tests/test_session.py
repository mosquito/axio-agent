"""Session lifecycle against a local OAuth server and an application store."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from axio_transport_openai.credentials import (
    CredentialConflict,
    CredentialError,
    CredentialRecord,
    save_account,
)
from axio_transport_openai.oauth import (
    RESOURCE,
    SCOPES,
    TERMINAL_REFRESH_ERRORS,
    ChatGPTAccount,
    ChatGPTOAuth,
    ChatGPTTokens,
    OAuthError,
    OAuthValidationError,
)
from axio_transport_openai.session import ChatGPTSession, PlanConsentRequired, SignInRequired


class ApplicationStore:
    """A test adapter; production persistence is supplied by the application."""

    def __init__(self) -> None:
        self.records: dict[str, CredentialRecord] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.fail_write = 0
        self.writes = 0

    @asynccontextmanager
    async def lock(self, client_id: str) -> AsyncIterator[None]:
        async with self.locks.setdefault(client_id, asyncio.Lock()):
            yield

    async def load(self, client_id: str) -> CredentialRecord | None:
        assert self.locks[client_id].locked()
        return self.records.get(client_id)

    async def save(self, record: CredentialRecord) -> None:
        assert self.locks[record.account.client_id].locked()
        self.writes += 1
        if self.writes == self.fail_write:
            raise CredentialError("Storage unavailable.")
        self.records[record.account.client_id] = record


@dataclass
class Server:
    issuer: str = ""
    refreshes: list[dict[str, str]] = field(default_factory=list)
    revocations: list[dict[str, str]] = field(default_factory=list)
    status: int = 200
    error: str = ""
    changes: dict[str, Any] = field(default_factory=dict)
    revoke_status: int = 200
    started: asyncio.Event = field(default_factory=asyncio.Event)
    proceed: asyncio.Event = field(default_factory=asyncio.Event)
    delay: bool = False
    bad_json: bool = False
    revoke_failures: int = 0
    lose_response: bool = False
    discovery_status: int = 200
    token_endpoint: str = ""

    async def discovery(self, request: web.Request) -> web.Response:
        if self.discovery_status != 200:
            return web.json_response({"error": "temporarily_unavailable"}, status=self.discovery_status)
        return web.json_response(
            {
                "issuer": self.issuer,
                "token_endpoint": self.token_endpoint or self.issuer + "/token",
                "revocation_endpoint": self.issuer + "/revoke",
            }
        )

    async def token(self, request: web.Request) -> web.Response:
        form = {key: str(value) for key, value in (await request.post()).items()}
        assert set(form) == {"grant_type", "client_id", "refresh_token", "resource"}
        assert form["grant_type"] == "refresh_token" and form["resource"] == RESOURCE
        assert "Authorization" not in request.headers
        self.refreshes.append(form)
        self.started.set()
        if self.lose_response:
            assert request.transport is not None
            request.transport.close()
            return web.Response()
        if self.delay:
            await self.proceed.wait()
        if self.status != 200:
            return web.json_response({"error": self.error, "error_description": "private-secret"}, status=self.status)
        if self.bad_json:
            return web.Response(text="private-secret")
        return web.json_response(
            {
                "access_token": "new-access-" + form["client_id"],
                "refresh_token": "new-refresh-" + form["client_id"],
                "token_type": "Bearer",
                "expires_in": 3600,
                **self.changes,
            }
        )

    async def revoke(self, request: web.Request) -> web.Response:
        form = {key: str(value) for key, value in (await request.post()).items()}
        self.revocations.append(form)
        assert set(form) == {"token", "client_id", "token_type_hint"}
        assert form["token_type_hint"] == "refresh_token"
        if self.revoke_failures:
            self.revoke_failures -= 1
            return web.Response(status=503)
        return web.Response(status=self.revoke_status)


@pytest.fixture
async def server() -> AsyncIterator[Server]:
    server = Server()
    app = web.Application()
    app.router.add_get("/.well-known/openid-configuration", server.discovery)
    app.router.add_post("/token", server.token)
    app.router.add_post("/revoke", server.revoke)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    server.issuer = f"http://127.0.0.1:{runner.addresses[0][1]}"
    try:
        yield server
    finally:
        server.proceed.set()
        await runner.cleanup()


@pytest.fixture
async def oauth(server: Server) -> AsyncIterator[ChatGPTOAuth]:
    async with aiohttp.ClientSession() as http:
        yield ChatGPTOAuth(http, "stable-host", issuer=server.issuer)


@pytest.fixture
def account(server: Server) -> ChatGPTAccount:
    now = time.time()
    return ChatGPTAccount(
        "client-a",
        "stable-host",
        server.issuer,
        "subject-a",
        "owl@example.test",
        "Owl",
        ChatGPTTokens(
            "id-secret", "old-access", "old-refresh", "Bearer", frozenset(SCOPES.split()), now - 3600, now - 1
        ),
    )


@pytest.fixture
async def store(account: ChatGPTAccount) -> ApplicationStore:
    store = ApplicationStore()
    await save_account(store, account)
    return store


def session(store: ApplicationStore, oauth: ChatGPTOAuth) -> ChatGPTSession:
    return ChatGPTSession(store, oauth, "client-a", retry_delay=0.01)


async def test_rotates_and_preserves_omitted_scope_and_id(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    account: ChatGPTAccount,
) -> None:
    assert await session(store, oauth).access_token() == "new-access-client-a"
    record = store.records[account.client_id]
    assert record.operation is None
    assert record.account.tokens is not None and account.tokens is not None
    assert record.account.tokens.refresh_token == "new-refresh-client-a"
    assert record.account.tokens.id_token == account.tokens.id_token
    assert record.account.tokens.scopes == account.tokens.scopes
    assert record.account.tokens.expires_at is not None and record.account.tokens.expires_at > time.time()
    assert server.refreshes[0]["refresh_token"] == "old-refresh"
    assert "secret" not in repr(record) + repr(record.account) + repr(record.account.tokens)


async def test_concurrent_sessions_reload_under_lock(
    store: ApplicationStore, oauth: ChatGPTOAuth, server: Server
) -> None:
    first, stale = session(store, oauth), session(store, oauth)
    values = await asyncio.gather(*(s.access_token() for s in (first, stale, first, stale)))
    assert values == ["new-access-client-a"] * 4
    assert len(server.refreshes) == 1


@pytest.mark.parametrize("code", sorted(TERMINAL_REFRESH_ERRORS))
async def test_terminal_error_clears_only_tokens(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    account: ChatGPTAccount,
    code: str,
) -> None:
    server.status, server.error = 400, code
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()
    assert store.records[account.client_id].account == replace(account, tokens=None)


@pytest.mark.parametrize("status,code", [(503, "invalid_grant"), (429, ""), (400, "invalid_client"), (400, "unknown")])
async def test_failure_preserves_credentials(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    account: ChatGPTAccount,
    status: int,
    code: str,
) -> None:
    server.status, server.error = status, code
    with pytest.raises(OAuthError) as error:
        await session(store, oauth).access_token()
    assert "private-secret" not in repr(error.value)
    assert store.records[account.client_id].account == account
    assert store.records[account.client_id].operation is None


@pytest.mark.parametrize("failure", ["disconnect", "timeout"])
@pytest.mark.parametrize("unexpired", [False, True])
async def test_lost_refresh_response_never_replays_token(
    store: ApplicationStore, server: Server, account: ChatGPTAccount, failure: str, unexpired: bool
) -> None:
    if unexpired:
        assert account.tokens is not None
        account = replace(account, tokens=replace(account.tokens, expires_at=time.time() + 30))
        store.records[account.client_id] = CredentialRecord(account)
    server.lose_response = failure == "disconnect"
    server.delay = failure == "timeout"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.2)) as http:
        oauth = ChatGPTOAuth(http, account.host_id, issuer=server.issuer)
        active = session(store, oauth)
        with pytest.raises(OAuthError):
            await active.access_token()
        assert server.started.is_set()
        assert store.records[account.client_id].operation == "refresh"
        assert store.records[account.client_id].account == account
        assert active.last_error is not None
        for caller in (active, session(store, oauth)):
            with pytest.raises(SignInRequired, match="interrupted"):
                await caller.access_token()
        assert len(server.refreshes) == 1


@pytest.mark.parametrize("failure", ["discovery", "connection"])
async def test_failure_before_refresh_request_can_retry(
    store: ApplicationStore, oauth: ChatGPTOAuth, server: Server, failure: str, unused_tcp_port: int
) -> None:
    if failure == "discovery":
        server.discovery_status = 503
    else:
        server.token_endpoint = f"{server.issuer.rsplit(':', 1)[0]}:{unused_tcp_port}/token"
    with pytest.raises(OAuthError) as error:
        await session(store, oauth).access_token()
    assert error.value.request_not_sent
    assert store.records["client-a"].operation is None
    assert not server.refreshes
    server.discovery_status = 200
    server.token_endpoint = ""
    async with aiohttp.ClientSession() as http:
        fresh_oauth = ChatGPTOAuth(http, "stable-host", issuer=server.issuer)
        assert await session(store, fresh_oauth).access_token() == "new-access-client-a"
    assert len(server.refreshes) == 1


async def test_refresh_deadline_preserves_marker(
    store: ApplicationStore, oauth: ChatGPTOAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def expired(self: ChatGPTOAuth, account: ChatGPTAccount) -> ChatGPTAccount:
        raise TimeoutError

    monkeypatch.setattr(ChatGPTOAuth, "refresh", expired)
    with pytest.raises(OAuthError, match="timed out"):
        await session(store, oauth).access_token()
    assert store.records["client-a"].operation == "refresh"
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()


async def test_transient_failure_can_return_unexpired_access(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    account: ChatGPTAccount,
) -> None:
    assert account.tokens is not None
    store.records[account.client_id] = CredentialRecord(
        replace(account, tokens=replace(account.tokens, expires_at=time.time() + 1))
    )
    server.status = 503
    active = session(store, oauth)
    assert await active.access_token() == "old-access"
    assert isinstance(active.last_error, OAuthError) and active.last_error.temporary


@pytest.mark.parametrize("changes", [{"refresh_token": None}, {"access_token": None}, {"expires_in": -1}])
async def test_bad_rotation_is_not_replayed(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    changes: dict[str, Any],
) -> None:
    server.changes = changes
    with pytest.raises(OAuthValidationError):
        await session(store, oauth).access_token()
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()
    assert len(server.refreshes) == 1


async def test_invalid_success_json_is_not_replayed(
    store: ApplicationStore, oauth: ChatGPTOAuth, server: Server
) -> None:
    server.bad_json = True
    with pytest.raises(OAuthValidationError):
        await session(store, oauth).access_token()
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()
    assert len(server.refreshes) == 1


async def test_write_failure_before_exchange(store: ApplicationStore, oauth: ChatGPTOAuth, server: Server) -> None:
    store.fail_write = store.writes + 1
    with pytest.raises(CredentialError):
        await session(store, oauth).access_token()
    assert not server.refreshes
    assert store.records["client-a"].operation is None


async def test_write_failure_after_rotation_can_retry_commit(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
) -> None:
    store.fail_write = store.writes + 2
    active = session(store, oauth)
    with pytest.raises(CredentialError):
        await active.access_token()
    assert store.records["client-a"].operation == "refresh"
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()
    assert await active.access_token() == "new-access-client-a"
    assert len(server.refreshes) == 1


async def test_restart_requires_reauth_after_interrupted_refresh(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    server: Server,
) -> None:
    interrupted = CredentialRecord(account, operation="refresh")
    store.records[account.client_id] = interrupted
    assert interrupted.public_account().tokens is None
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()
    assert not server.refreshes
    await save_account(store, account, expected=interrupted)
    assert await session(store, oauth).access_token() == "new-access-client-a"


async def test_cancelled_refresh_finishes_commit(store: ApplicationStore, oauth: ChatGPTOAuth, server: Server) -> None:
    server.delay = True
    task = asyncio.create_task(session(store, oauth).access_token())
    await asyncio.wait_for(server.started.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()  # Repeated cancellation must not release the lock early.
    server.proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await session(store, oauth).access_token() == "new-access-client-a"
    assert len(server.refreshes) == 1


async def test_logout_waits_for_refresh_then_revokes_latest(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    account: ChatGPTAccount,
) -> None:
    server.delay = True
    refresh = asyncio.create_task(session(store, oauth).access_token())
    await asyncio.wait_for(server.started.wait(), 2)
    logout = asyncio.create_task(session(store, oauth).sign_out())
    await asyncio.sleep(0)
    assert not server.revocations
    server.proceed.set()
    await refresh
    assert (await logout).revocation_confirmed
    assert server.revocations[0]["token"] == "new-refresh-client-a"
    assert store.records["client-a"].account == replace(account, tokens=None)
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()


@pytest.mark.parametrize("status", [200, 400, 503])
async def test_logout_retains_registration_even_without_confirmation(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    account: ChatGPTAccount,
    status: int,
) -> None:
    server.revoke_status = status
    result = await session(store, oauth).sign_out()
    assert result.revocation_confirmed == (status == 200)
    assert len(server.revocations) == (3 if status == 503 else 1)
    assert store.records["client-a"].account == replace(account, tokens=None)


async def test_logout_retries_temporary_failure(store: ApplicationStore, oauth: ChatGPTOAuth, server: Server) -> None:
    server.revoke_failures = 1
    assert (await session(store, oauth).sign_out()).revocation_confirmed
    assert len(server.revocations) == 2


async def test_logout_after_pending_write_does_not_resurrect_tokens(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
) -> None:
    active = session(store, oauth)
    store.fail_write = store.writes + 2
    with pytest.raises(CredentialError):
        await active.access_token()
    await session(store, oauth).sign_out()
    with pytest.raises(SignInRequired):
        await active.access_token()
    assert store.records["client-a"].account.tokens is None
    assert len(server.refreshes) == 1


async def test_logout_uses_pending_rotated_token(store: ApplicationStore, oauth: ChatGPTOAuth, server: Server) -> None:
    active = session(store, oauth)
    store.fail_write = store.writes + 2
    with pytest.raises(CredentialError):
        await active.access_token()
    assert (await active.sign_out()).revocation_confirmed
    assert server.revocations[0]["token"] == "new-refresh-client-a"


async def test_logout_write_failure_is_not_reported_as_success(store: ApplicationStore, oauth: ChatGPTOAuth) -> None:
    store.fail_write = store.writes + 2
    with pytest.raises(CredentialError):
        await session(store, oauth).sign_out()
    assert store.records["client-a"].operation == "logout"
    with pytest.raises(SignInRequired):
        await session(store, oauth).access_token()


async def test_accounts_with_same_email_are_isolated(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    server: Server,
) -> None:
    other = replace(account, client_id="client-b", subject="subject-b")
    await save_account(store, other)
    pinned = session(store, oauth)
    other_session = ChatGPTSession(store, oauth, other.client_id)
    assert await other_session.access_token() == "new-access-client-b"
    await pinned.sign_out()
    assert await other_session.access_token() == "new-access-client-b"
    assert len(server.refreshes) == 1
    assert store.records["client-b"].account.subject == "subject-b"


async def test_stale_sign_in_cannot_replace_new_tokens(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
) -> None:
    old = store.records[account.client_id]
    await session(store, oauth).access_token()
    with pytest.raises(CredentialConflict):
        await save_account(store, account, expected=old)


@pytest.mark.parametrize("changes", [{"subject": "another"}, {"host_id": "another"}, {"issuer": "https://other.test"}])
async def test_registration_cannot_change_identity(
    store: ApplicationStore,
    account: ChatGPTAccount,
    changes: dict[str, Any],
) -> None:
    with pytest.raises(CredentialConflict):
        await save_account(store, replace(account, **changes), expected=store.records[account.client_id])


async def test_background_refresh_without_foreground_calls(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    server: Server,
) -> None:
    assert account.tokens is not None
    now = time.time()
    store.records["client-a"] = CredentialRecord(
        replace(account, tokens=replace(account.tokens, received_at=now, expires_at=now + 0.3))
    )
    async with session(store, oauth) as active:
        assert not server.refreshes
        await asyncio.wait_for(server.started.wait(), 2)
        async with asyncio.timeout(2):
            while store.records["client-a"].operation is not None:
                await asyncio.sleep(0.01)
        assert active.last_error is None
    assert len(server.refreshes) == 1
    assert store.records["client-a"].account.tokens is not None
    with pytest.raises(RuntimeError, match="closed"):
        await active.access_token()


async def test_background_recovers_from_temporary_error(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    server: Server,
) -> None:
    assert account.tokens is not None
    store.records["client-a"] = CredentialRecord(
        replace(account, tokens=replace(account.tokens, expires_at=time.time() + 1))
    )
    server.status = 503
    async with session(store, oauth) as active:
        assert active.last_error is not None
        server.status = 200
        async with asyncio.timeout(2):
            while store.records["client-a"].account.tokens == account.tokens or active.last_error is not None:
                await asyncio.sleep(0.01)
        assert await active.access_token() == "new-access-client-a"


async def test_close_waits_for_background_commit(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    server: Server,
) -> None:
    assert account.tokens is not None
    now = time.time()
    store.records["client-a"] = CredentialRecord(
        replace(account, tokens=replace(account.tokens, received_at=now, expires_at=now + 0.2))
    )
    server.delay = True
    active = session(store, oauth)
    await active.start()
    await asyncio.wait_for(server.started.wait(), 2)
    closing = asyncio.create_task(active.close())
    await asyncio.sleep(0)
    assert not closing.done()
    server.proceed.set()
    await asyncio.wait_for(closing, 2)
    assert store.records["client-a"].operation is None
    assert store.records["client-a"].account.tokens is not None


async def test_missing_consent_never_refreshes(
    store: ApplicationStore, oauth: ChatGPTOAuth, account: ChatGPTAccount, server: Server
) -> None:
    assert account.tokens is not None
    store.records["client-a"] = CredentialRecord(
        replace(account, tokens=replace(account.tokens, scopes=frozenset({"openid"})))
    )
    with pytest.raises(PlanConsentRequired):
        await session(store, oauth).start()
    assert not server.refreshes


async def test_revoked_consent_is_persisted(store: ApplicationStore, oauth: ChatGPTOAuth, server: Server) -> None:
    server.changes = {"scope": "openid"}
    with pytest.raises(PlanConsentRequired):
        await session(store, oauth).access_token()
    assert store.records["client-a"].operation is None
    assert store.records["client-a"].account.tokens is not None
    assert not store.records["client-a"].account.tokens.plan_enabled


class SharedProcessStore:
    """An external test backend shared by independent interpreter processes."""

    def __init__(self, records: Any, mutex: Any) -> None:
        self.records = records
        self.mutex = mutex

    @asynccontextmanager
    async def lock(self, client_id: str) -> AsyncIterator[None]:
        while not await asyncio.to_thread(self.mutex.acquire, False):
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            await asyncio.to_thread(self.mutex.release)

    async def load(self, client_id: str) -> CredentialRecord | None:
        record: CredentialRecord | None = await asyncio.to_thread(self.records.get, client_id)
        return record

    async def save(self, record: CredentialRecord) -> None:
        await asyncio.to_thread(self.records.__setitem__, record.account.client_id, record)


def process_refresh(records: Any, mutex: Any, issuer: str) -> None:
    async def run() -> None:
        async with aiohttp.ClientSession() as http:
            oauth = ChatGPTOAuth(http, "stable-host", issuer=issuer)
            active = ChatGPTSession(SharedProcessStore(records, mutex), oauth, "client-a")
            assert await active.access_token() == "new-access-client-a"

    asyncio.run(run())


async def test_two_processes_share_rotating_credentials(account: ChatGPTAccount, server: Server) -> None:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    manager = await asyncio.to_thread(context.Manager)
    records = manager.dict({account.client_id: CredentialRecord(account)})
    mutex = manager.Lock()
    processes = [context.Process(target=process_refresh, args=(records, mutex, server.issuer)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        async with asyncio.timeout(15):
            while any(process.is_alive() for process in processes):
                await asyncio.sleep(0.05)
        for process in processes:
            process.join()
            assert process.exitcode == 0
        assert len(server.refreshes) == 1
        assert records[account.client_id].operation is None
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 2)
        await asyncio.to_thread(manager.shutdown)


async def test_concurrent_start_creates_one_worker(store: ApplicationStore, oauth: ChatGPTOAuth) -> None:
    active = session(store, oauth)
    await asyncio.gather(active.start(), active.start(), active.start())
    workers = [task for task in asyncio.all_tasks() if task.get_name() == "chatgpt-token-renewal"]
    assert len(workers) == 1
    await active.close()
    assert workers[0].done()


async def test_close_waits_for_foreground_commit(store: ApplicationStore, oauth: ChatGPTOAuth, server: Server) -> None:
    active = session(store, oauth)
    server.delay = True
    request = asyncio.create_task(active.access_token())
    await asyncio.wait_for(server.started.wait(), 2)
    closing = asyncio.create_task(active.close())
    await asyncio.sleep(0)
    assert not closing.done()
    server.proceed.set()
    await asyncio.wait_for(closing, 2)
    assert await request == "new-access-client-a"
    assert store.records["client-a"].operation is None


async def test_uncertain_rotation_cannot_confirm_remote_logout(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
) -> None:
    store.records[account.client_id] = CredentialRecord(account, operation="refresh")
    assert not (await session(store, oauth).sign_out()).revocation_confirmed
    assert store.records[account.client_id].account.tokens is None


async def test_background_terminal_error_stops_renewal(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    server: Server,
) -> None:
    assert account.tokens is not None
    now = time.time()
    store.records["client-a"] = CredentialRecord(
        replace(account, tokens=replace(account.tokens, received_at=now, expires_at=now + 0.2))
    )
    server.status, server.error = 400, "invalid_grant"
    async with session(store, oauth) as active:
        async with asyncio.timeout(2):
            while active.last_error is None:
                await asyncio.sleep(0.01)
        assert isinstance(active.last_error, SignInRequired)
        assert store.records["client-a"].account.tokens is None
    assert len(server.refreshes) == 1


async def test_network_failure_preserves_tokens(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(self: aiohttp.ClientSession, *args: Any, **kwargs: Any) -> aiohttp.ClientResponse:
        raise aiohttp.ClientConnectionError("private-secret")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", unavailable)
    with pytest.raises(OAuthError, match="Could not read") as error:
        await session(store, oauth).access_token()
    assert "private-secret" not in repr(error.value)
    assert store.records[account.client_id].account == account
    assert store.records[account.client_id].operation is None


async def test_revocation_network_failure_reports_unconfirmed(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    account: ChatGPTAccount,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(self: aiohttp.ClientSession, *args: Any, **kwargs: Any) -> aiohttp.ClientResponse:
        raise aiohttp.ClientConnectionError("private-secret")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", unavailable)
    assert not (await session(store, oauth).sign_out()).revocation_confirmed
    assert store.records[account.client_id].account == replace(account, tokens=None)


async def test_cancelled_logout_finishes_clearing_tokens(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, proceed = asyncio.Event(), asyncio.Event()
    original = ChatGPTOAuth.revoke

    async def delayed(self: ChatGPTOAuth, account: ChatGPTAccount) -> None:
        entered.set()
        await proceed.wait()
        await original(self, account)

    monkeypatch.setattr(ChatGPTOAuth, "revoke", delayed)
    task = asyncio.create_task(session(store, oauth).sign_out())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.records["client-a"].account.tokens is None


async def test_committed_write_with_error_is_reloaded_without_refresh(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = store.save

    async def committed(record: CredentialRecord) -> None:
        await original(record)
        if record.operation is None:
            raise CredentialError("Commit acknowledgement lost.")

    monkeypatch.setattr(store, "save", committed)
    active = session(store, oauth)
    with pytest.raises(CredentialError):
        await active.access_token()
    monkeypatch.setattr(store, "save", original)
    assert await active.access_token() == "new-access-client-a"
    assert len(server.refreshes) == 1


async def test_logout_keeps_pending_rotation_if_marker_save_fails(
    store: ApplicationStore,
    oauth: ChatGPTOAuth,
    server: Server,
) -> None:
    active = session(store, oauth)
    store.fail_write = store.writes + 2
    with pytest.raises(CredentialError):
        await active.access_token()
    store.fail_write = store.writes + 1
    with pytest.raises(CredentialError):
        await active.sign_out()
    assert not server.revocations
    assert (await active.sign_out()).revocation_confirmed
    assert server.revocations[0]["token"] == "new-refresh-client-a"
