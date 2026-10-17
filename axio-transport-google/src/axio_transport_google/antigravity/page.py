"""Browser callback pages using the existing Axio sign-in design."""

from __future__ import annotations

from importlib.resources import files
from string import Template


def sign_in_page(*, success: bool, nonce: str) -> str:
    """Render fixed text without account details, tokens, or remote assets."""
    template = Template(files(__package__).joinpath("oauth.html").read_text(encoding="utf-8"))
    return template.substitute(
        title="You're signed in." if success else "Sign-in didn't finish.",
        message="You can close this tab and return to Axio." if success else "Return to Axio and try again.",
        status="Google account connected" if success else "Your account was not connected",
        state="success" if success else "error",
        nonce=nonce,
        button_label="Close tab (5)" if success else "Close tab",
    )


def page_headers(nonce: str) -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": (
            "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
            f"script-src 'nonce-{nonce}'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        ),
        "X-Content-Type-Options": "nosniff",
    }
