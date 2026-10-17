"""Sign in through a browser and verify a credential round trip.

Run from the workspace:
    uv run --directory axio-transport-google python examples/antigravity_oauth.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import webbrowser
from pathlib import Path

import aiohttp
from axio.credentials import CredentialError

from axio_transport_google.antigravity import (
    AntigravityFileCredentials,
    AntigravityOAuth,
    AntigravitySession,
    OAuthError,
    save_account,
)


async def run(credentials: Path | None, no_browser: bool, timeout: float) -> None:
    store = AntigravityFileCredentials(credentials)
    previous = {item.account_id: item for item in await store.list_records()}

    async def open_browser(url: str) -> None:
        print(f"Authorize Antigravity in your browser:\n{url}", flush=True)
        if not no_browser:
            await asyncio.to_thread(webbrowser.open, url)

    async with aiohttp.ClientSession() as http:
        oauth = AntigravityOAuth(http)
        account = await oauth.sign_in(open_browser=open_browser, timeout=timeout)
        await save_account(store, account, expected=previous.get(account.account_id))
        await store.select_account(account.account_id)
        reopened = AntigravityFileCredentials(store.path)
        selected = await reopened.selected_account()
        if selected != account.account_id:
            raise CredentialError("Saved account selection did not survive reopening.")
        token = await AntigravitySession(reopened, oauth, selected).access_token()
        if account.tokens is None or not token or account.tokens.expires_at <= time.time():
            raise OAuthError("Sign-in did not produce usable credentials.")
        print("OAUTH_OK: identity verified; refresh token saved; credentials reopened successfully.")
        print(f"Credential file: {store.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Antigravity browser OAuth.")
    parser.add_argument("--credentials", type=Path, help="Credential file; defaults to the Axio XDG location.")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds to wait for sign-in.")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.credentials, args.no_browser, args.timeout))
    except (CredentialError, OAuthError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
    except TimeoutError:
        print("Sign-in timed out.", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Sign-in cancelled.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
