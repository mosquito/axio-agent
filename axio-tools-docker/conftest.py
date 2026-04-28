"""Pytest configuration for axio-tools-docker."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "docker: mark test as requiring a running Docker daemon",
    )
