"""Self-contained browser responses for the loopback sign-in callback."""

from __future__ import annotations

from importlib.resources import files
from string import Template


def sign_in_page(*, success: bool, plan_enabled: bool = False, nonce: str) -> str:
    """Render fixed status text without account details or remote assets."""
    template = Template(files(__package__).joinpath("oauth.html").read_text(encoding="utf-8"))
    if success:
        title = "You're signed in."
        message = "You can close this tab and return to Axio."
        status = "ChatGPT plan connected" if plan_enabled else "ChatGPT plan access is not enabled"
    else:
        title = "Sign-in didn't finish."
        message = "Return to Axio and try again."
        status = "Your account was not connected"
    return template.substitute(
        title=title,
        message=message,
        status=status,
        state="success" if success else "error",
        nonce=nonce,
        button_label="Close tab (5)" if success else "Close tab",
    )
