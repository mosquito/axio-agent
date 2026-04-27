"""Pytest configuration for gas_town tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path) -> Path:
    """Create a temporary workspace directory for tests."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace
