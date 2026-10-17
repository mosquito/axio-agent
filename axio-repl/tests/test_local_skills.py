"""Offline local skill listing in the REPL."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import axio_repl as repl


def write_skill(root: Path, name: str = "planning", *, extra: str = "", body: str = "Instructions.") -> Path:
    source = root / name / "SKILL.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"---\nname: {name}\ndescription: Plan work\n{extra}---\n{body}\n")
    return source.resolve()


@pytest.mark.asyncio
async def test_skills_list_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_skill(tmp_path / "skills")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--isolated", "--skills-dir", "skills", "--skills-list"])

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Local skill discovery must not select a transport or load MCP configuration")

    monkeypatch.setattr(repl, "_select_transport", unexpected)
    monkeypatch.setattr("axio_repl.configuration.load_mcp_config", unexpected)
    await repl.main()
    output = capsys.readouterr().out
    assert "local:planning" in output
    assert str(source) in output
