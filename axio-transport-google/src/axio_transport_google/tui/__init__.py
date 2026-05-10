"""TUI screens for axio-transport-google (requires textual)."""

from __future__ import annotations

try:
    from .google import GoogleSettingsScreen, VertexSettingsScreen

    __all__ = ["GoogleSettingsScreen", "VertexSettingsScreen"]
except ImportError as _e:
    import warnings

    warnings.warn(
        f"axio-transport-google TUI screens are unavailable: {_e}. Install textual.",
        stacklevel=1,
    )
