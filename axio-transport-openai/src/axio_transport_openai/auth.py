"""Terminal sign-in: python -m axio_transport_openai.auth."""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
import sys
from pathlib import Path

import aiohttp

from .credential_stores import FileCredentialStore
from .credentials import CredentialError, CredentialRecord, save_account
from .oauth import ChatGPTAccount, ChatGPTOAuth, OAuthError, launch_browser, print_authorization_url


class CallbackInput:
    """Read pasted callback URLs without leaving a blocked stdin worker thread."""

    def __init__(self, fd: int | None = None) -> None:
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.buffer = bytearray()
        self.eof = False

    async def readline(self) -> str:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()

        def receive() -> None:
            if ready.done():
                return
            try:
                chunk = os.read(self.fd, 8192)
                self.buffer.extend(chunk)
                self.eof = not chunk
                if len(self.buffer) > 65536:
                    raise OAuthError("The pasted callback URL is too long.")
                if b"\n" in self.buffer or self.eof:
                    ready.set_result(None)
            except (OSError, OAuthError):
                ready.set_exception(OAuthError("Could not read the callback URL from stdin."))

        if b"\n" not in self.buffer and not self.eof:
            if stat.S_ISREG(os.fstat(self.fd).st_mode):
                # Regular input files cannot use epoll; each read is finite.
                while not ready.done():
                    receive()
            else:
                loop.add_reader(self.fd, receive)
                try:
                    await ready
                finally:
                    loop.remove_reader(self.fd)
            ready.result()
        line, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        if not line.strip():
            raise OAuthError("No callback URL was provided on stdin.")
        try:
            return line.decode("utf-8").strip()
        except UnicodeError:
            raise OAuthError("The callback URL must be UTF-8 text.") from None


class TerminalAuthorization:
    """Use a browser when available; otherwise request a pasted redirect URL."""

    def __init__(self, *, manual: bool = False) -> None:
        self.manual = manual
        self.input: CallbackInput | None = None

    async def open(self, url: str) -> None:
        if not self.manual and await launch_browser(url):
            return
        self.manual = True
        print_authorization_url(url)
        print(
            "After approval, copy the full redirect URL from the browser address bar.\n"
            "If the browser cannot reach localhost, keep that address and paste it here.\n"
            "Paste redirect URL:",
            file=sys.stderr,
            flush=True,
        )

    async def read_callback(self) -> str:
        if not self.manual:
            # Normal browser callbacks complete sign-in and cancel this waiter.
            await asyncio.Future[None]()
        if self.input is None:
            self.input = CallbackInput()
        return await self.input.readline()


async def sign_in(
    *,
    credentials: Path | None = None,
    client_id: str | None = None,
    manual: bool = False,
    port: int = 0,
    timeout: float = 300,
    codex: bool = False,
) -> ChatGPTAccount:
    """Authorize and persist an account using the default XDG credentials file."""
    store = FileCredentialStore(credentials)
    expected: CredentialRecord | None = None
    if client_id is not None:
        async with store.lock(client_id):
            expected = await store.load(client_id)
        if expected is None:
            raise CredentialError("The selected account registration was not found.")
    terminal = TerminalAuthorization(manual=manual)
    async with aiohttp.ClientSession() as http:
        oauth = ChatGPTOAuth(http, await store.host_id(), codex=codex)
        account = await oauth.sign_in(
            expected.public_account() if expected else None,
            open_browser=terminal.open,
            read_callback=terminal.read_callback,
            port=port,
            timeout=timeout,
        )
    await save_account(store, account, expected=expected)
    return account


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign in with ChatGPT and save local credentials.")
    parser.add_argument("--credentials", type=Path, help="Override the XDG credentials file.")
    parser.add_argument("--client-id", help="Reauthorize an existing registration.")
    parser.add_argument("--codex", action="store_true", help="Use native ChatGPT OAuth for images and completion.")
    parser.add_argument(
        "--manual", action="store_true", help="Print the login URL and accept the redirect URL on stdin."
    )
    parser.add_argument("--port", type=int, default=0, help="Loopback callback port; 0 selects an available port.")
    parser.add_argument("--timeout", type=float, default=300, help="Sign-in timeout in seconds.")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        account = asyncio.run(sign_in(**vars(args)))
    except (OAuthError, CredentialError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except TimeoutError:
        print("Sign-in timed out.", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Sign-in cancelled.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError):
        print("Could not start sign-in. Check the callback port and timeout.", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Signed in. Client ID: {account.client_id}")
    if account.tokens is not None and not account.tokens.plan_enabled:
        print("ChatGPT plan access was not granted.", file=sys.stderr)


if __name__ == "__main__":
    main()
