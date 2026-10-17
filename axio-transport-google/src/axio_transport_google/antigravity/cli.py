"""Terminal authorization and account management for Antigravity."""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
import sys
import webbrowser
from pathlib import Path

import aiohttp
from axio.credentials import CredentialError
from axio.exceptions import StreamError

from .oauth import AntigravityAccount, AntigravityOAuth, OAuthError
from .session import AntigravityFileCredentials, AntigravitySession, list_accounts, resolve_account, save_account
from .transport import AntigravityTransport


async def read_callback() -> str:
    """Read a pasted URL without a worker thread that survives cancellation."""
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    result: asyncio.Future[str] = loop.create_future()
    buffer = bytearray()

    def receive() -> None:
        if result.done():
            return
        try:
            chunk = os.read(fd, 8192)
            buffer.extend(chunk)
            if len(buffer) > 65536:
                raise ValueError("The callback URL is too long.")
            if b"\n" in buffer or not chunk:
                value = bytes(buffer).decode().strip()
                if not value:
                    raise ValueError("No callback URL was provided.")
                result.set_result(value)
        except (OSError, ValueError) as error:
            result.set_exception(error)

    if stat.S_ISREG(os.fstat(fd).st_mode):
        while not result.done():
            receive()
        return await result
    loop.add_reader(fd, receive)
    try:
        return await result
    finally:
        loop.remove_reader(fd)


async def sign_in(
    store: AntigravityFileCredentials, *, manual: bool = False, account_id: str | None = None
) -> AntigravityAccount:
    """Authorize one identity and preserve concurrent credential updates."""
    previous = {account.account_id: account for account in await store.list_records()}
    if account_id is not None and account_id not in previous:
        raise CredentialError("The selected Antigravity account was not found.")
    manual = manual or bool(os.environ.get("SSH_CONNECTION"))
    paste = manual

    async def open_browser(url: str) -> None:
        nonlocal paste
        if not manual and await asyncio.to_thread(webbrowser.open, url):
            print("Complete Google sign-in in your browser.", flush=True)
            return
        paste = True
        print(f"Open this URL in your browser:\n{url}\nPaste the full localhost redirect URL:", flush=True)

    async def read() -> str:
        if not paste:
            await asyncio.Future[None]()
        return await read_callback()

    async with aiohttp.ClientSession() as http:
        account = await AntigravityOAuth(http).sign_in(open_browser=open_browser, read_callback=read, manual=manual)
    if account_id is not None and account.account_id != account_id:
        raise CredentialError("Sign-in returned a different Google account. The selected account was not changed.")
    await save_account(store, account, expected=previous.get(account.account_id))
    await store.select_account(account.account_id)
    return account


async def open_transport(
    store: AntigravityFileCredentials, http: aiohttp.ClientSession, account_id: str | None = None
) -> AntigravityTransport:
    selected = await resolve_account(store, account_id)
    transport = AntigravityTransport(session=http, account=AntigravitySession(store, AntigravityOAuth(http), selected))
    await transport.fetch_models()
    return transport


async def describe_accounts(store: AntigravityFileCredentials, active: str | None = None) -> str:
    selected = await store.selected_account()
    lines = []
    for account in await list_accounts(store):
        labels = [
            label
            for label, matches in (
                ("selected", account.account_id == selected),
                ("active", account.account_id == active),
            )
            if matches
        ]
        lines.append(f"{account.account_id}  {account.email or account.subject}  {', '.join(labels)}".rstrip())
    return "\n".join(lines) or "No Antigravity accounts are saved. Run axio-antigravity login."


async def run(args: argparse.Namespace) -> None:
    store = AntigravityFileCredentials(args.credentials)
    if args.command == "login":
        account = await sign_in(store, manual=args.manual, account_id=args.account)
        print(f"Signed in: {account.email or account.account_id}")
        return
    if args.command == "accounts":
        print(await describe_accounts(store))
        return
    selected = await resolve_account(store, args.account)
    if args.command == "select":
        await store.select_account(selected)
        print(f"Selected Antigravity account: {selected}")
        return
    async with aiohttp.ClientSession() as http:
        account_session = AntigravitySession(store, AntigravityOAuth(http), selected)
        if args.command == "logout":
            revoked = await account_session.logout(revoke=args.revoke)
            print("Local credentials cleared." + (" Google revocation confirmed." if revoked else ""))
            return
        if args.command == "status":
            await account_session.access_token()
            print("The selected Antigravity account has a valid OAuth token.")
            return
        transport = await open_transport(store, http, selected)
        for model in transport.models:
            print(f"{model.id}  {transport.catalog[model.id].get('displayName', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Antigravity OAuth accounts.")
    parser.add_argument("--credentials", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("login", "accounts", "select", "status", "models", "logout"):
        child = commands.add_parser(command)
        child.add_argument("--account", required=command == "select", help="Saved account ID")
        if command == "login":
            child.add_argument("--manual", action="store_true", help="Paste a redirect URL, for SSH or headless use")
        if command == "logout":
            child.add_argument("--revoke", action="store_true", help="Also revoke the Google grant")
    try:
        asyncio.run(run(parser.parse_args()))
    except TimeoutError:
        print("Antigravity sign-in timed out.", file=sys.stderr)
        raise SystemExit(1) from None
    except (CredentialError, OAuthError, StreamError, aiohttp.ClientError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
