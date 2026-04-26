"""Test configuration for agent_swarm example tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace directory for tests."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


@pytest.fixture
def swarm_config() -> dict[str, Any]:
    """Provide a default swarm configuration for testing."""
    return {
        "roles": ["architect", "developer", "reviewer"],
        "max_iterations": 10,
    }
