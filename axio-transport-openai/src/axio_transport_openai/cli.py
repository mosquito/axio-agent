"""ChatGPT account commands without a Textual dependency."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from axio_responses import ResponsesAPIError

from .accounts import list_accounts, open_chatgpt, resolve_account
from .auth import sign_in
from .credential_stores import FileCredentialStore, LocalCredentialStore
from .credentials import CredentialError
from .oauth import ChatGPTOAuth, OAuthError
from .session import ChatGPTSession


async def account_status(store: LocalCredentialStore, client_id: str) -> dict[str, Any]:
    """Describe local credentials without returning token values or refreshing."""
    async with store.lock(client_id):
        record = await store.load(client_id)
    if record is None:
        raise CredentialError("The account registration was not found.")
    account = record.account
    tokens = account.tokens
    if record.operation is not None:
        state = "reauthentication_required"
    elif tokens is None:
        state = "signed_out"
    elif not tokens.plan_enabled:
        state = "consent_required"
    elif tokens.access_token is None or tokens.expires_at is None:
        state = "reauthentication_required"
    elif tokens.expires_at <= time.time():
        state = "refresh_required" if tokens.refresh_token else "reauthentication_required"
    else:
        state = "ready"
    return {
        "client_id": client_id,
        "email": account.email,
        "name": account.name,
        "state": state,
        "plan_enabled": tokens.plan_enabled if tokens else False,
        "expires_at": tokens.expires_at if tokens else None,
    }


def error_message(error: Exception) -> str:
    """Keep provider response bodies, URLs, and credentials out of diagnostics."""
    if isinstance(error, ResponsesAPIError):
        return f"ChatGPT request failed (HTTP {error.status}, recovery={error.recovery.value})."
    if isinstance(error, (OAuthError, CredentialError)):
        return str(error)
    if isinstance(error, TimeoutError):
        return "The operation timed out."
    if isinstance(error, aiohttp.ClientError):
        return "The network request failed."
    if isinstance(error, ValueError):
        return "The command configuration or model catalog is invalid."
    return "Could not access the local credential store."


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--credentials", type=Path, default=argparse.SUPPRESS, help="Override the XDG credential file."
    )
    result = argparse.ArgumentParser(description="Manage ChatGPT subscription accounts.")
    result.add_argument("--credentials", type=Path, help="Override the XDG credential file.")
    commands = result.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", parents=[common], help="Sign in and select the saved registration.")
    login.add_argument("--account", "--client-id", dest="client_id", help="Reauthorize an existing registration.")
    login.add_argument("--manual", action="store_true", help="Use a printed URL and pasted callback over SSH.")
    login.add_argument("--port", type=int, default=0)
    login.add_argument("--timeout", type=float, default=300)
    login.add_argument("--codex", action="store_true", help="Use native ChatGPT authorization with image generation.")
    accounts = commands.add_parser("accounts", parents=[common], help="List saved registrations without tokens.")
    accounts.add_argument("--json", action="store_true")
    select = commands.add_parser("select", parents=[common], help="Choose an account for new sessions.")
    select.add_argument("account")
    for name, help_text in (
        ("status", "Read local credential status without a network request."),
        ("logout", "Sign out while retaining the account registration."),
        ("models", "List the selected account's visible models."),
    ):
        command = commands.add_parser(name, parents=[common], help=help_text)
        command.add_argument("--account", help="Override the selected account for this command.")
        if name != "logout":
            command.add_argument("--json", action="store_true")
    return result


async def run(args: argparse.Namespace) -> int:
    store = FileCredentialStore(args.credentials)
    if args.command == "login":
        account = await sign_in(
            credentials=args.credentials,
            client_id=args.client_id,
            manual=args.manual,
            port=args.port,
            timeout=args.timeout,
            **({"codex": True} if args.codex else {}),
        )
        await store.select_account(account.client_id)
        print(f"Signed in and selected: {account.client_id}")
        if account.tokens is None or not account.tokens.plan_enabled:
            print("Identity sign-in succeeded. ChatGPT plan permission was not granted.", file=sys.stderr)
        return 0
    if args.command == "select":
        await store.select_account(args.account)
        print(f"Selected: {args.account}")
        return 0
    if args.command == "accounts":
        selected = await store.selected_account()
        accounts = await list_accounts(store)
        if selected is None and len(accounts) == 1:
            selected = accounts[0].client_id
        rows = []
        for account in accounts:
            row = await account_status(store, account.client_id)
            row["selected"] = account.client_id == selected
            rows.append(row)
        if args.json:
            print(json.dumps(rows))
        elif not rows:
            print("No saved accounts. Run axio-chatgpt login.")
        else:
            for row in rows:
                print(
                    f"{'*' if row['selected'] else ' '} {row['client_id']}  "
                    f"{json.dumps(row['email'] or row['name'] or '')}  {row['state']}"
                )
        return 0
    client_id = await resolve_account(store, args.account)
    if args.command == "status":
        row = await account_status(store, client_id)
        if args.json:
            print(json.dumps(row))
        else:
            print(f"Account: {client_id}\nLocal status: {row['state']}\nPlan permission: {row['plan_enabled']}")
        return 0
    async with aiohttp.ClientSession() as http:
        if args.command == "models":
            async with open_chatgpt(store, http, client_id=client_id) as transport:
                models = [
                    {
                        "id": model.id,
                        "display_name": model.display_name,
                        "capabilities": sorted(cap.value for cap in model.capabilities),
                    }
                    for model in transport.catalog
                    if model.visibility == "list"
                ]
                if args.json:
                    print(json.dumps(models))
                else:
                    for model in models:
                        print(f"{model['id']}  {json.dumps(model['display_name'])}")
        else:
            oauth = ChatGPTOAuth(http, await store.host_id())
            account_session = ChatGPTSession(store, oauth, client_id)
            try:
                result = await account_session.sign_out()
            finally:
                await account_session.close()
            print(f"Signed out locally: {client_id}")
            if not result.revocation_confirmed:
                print("Remote revocation was not confirmed. Check ChatGPT account settings.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> None:
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.command == "login" and (not 0 <= args.port <= 65535 or not 0 < args.timeout < float("inf")):
        command_parser.error("Use a port from 0 to 65535 and a finite, positive timeout.")
    try:
        code = asyncio.run(run(args))
    except (OAuthError, CredentialError, ResponsesAPIError, aiohttp.ClientError, OSError, ValueError) as error:
        print(error_message(error), file=sys.stderr)
        code = 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Operation cancelled.", file=sys.stderr)
        code = 130
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
