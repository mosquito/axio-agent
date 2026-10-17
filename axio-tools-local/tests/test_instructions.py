from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from axio_tools_local.instructions import load_instructions
from axio_tools_local.paths import LocalPaths as Workspace


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    config = tmp_path / "config" / "axio"
    config.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config.parent))
    return root, config


def test_chain_preserves_order_scope_and_contents(project: tuple[Path, Path]) -> None:
    root, config = project
    api = root / "services" / "api"
    api.mkdir(parents=True)
    paths = [config / "AGENTS.md", root / "AGENTS.md", api.parent / "AGENTS.md", api / "AGENTS.md"]
    texts = ["global\n", "project\n", "service\n", "API rules\n"]
    for path, text in zip(paths, texts, strict=True):
        path.write_text(text)
    sibling = root / "frontend"
    sibling.mkdir()
    (sibling / "AGENTS.md").write_text("not applicable")
    snapshot = load_instructions(Workspace.discover(api, root))
    assert [block.source for block in snapshot.blocks] == paths
    assert [block.scope for block in snapshot.blocks] == [None, root, api.parent, api]
    assert [block.text for block in snapshot.blocks] == texts
    assert snapshot.blocks[-1].digest == hashlib.sha256(texts[-1].encode()).hexdigest()
    assert "not applicable" not in snapshot.render()
    assert "deeper scopes override" in snapshot.render()


def test_exact_name_only(project: tuple[Path, Path]) -> None:
    root, _ = project
    for name in ("agents.md", "AGENT.md", "CLAUDE.md"):
        (root / name).write_text("wrong name")
    assert not load_instructions(Workspace.discover(root, root)).blocks


def test_isolated_reads_only_explicit_files(project: tuple[Path, Path]) -> None:
    root, config = project
    (root / "AGENTS.md").write_bytes(b"\xff")
    (config / "AGENTS.md").write_bytes(b"\xff")
    explicit = root / "chosen.md"
    explicit.write_text("explicit rules")
    workspace = Workspace.discover(root, root)
    assert not load_instructions(workspace, isolated=True).blocks
    snapshot = load_instructions(workspace, isolated=True, files=[Path("chosen.md")])
    assert len(snapshot.blocks) == 1
    assert snapshot.blocks[0].source == explicit
    assert snapshot.blocks[0].scope is None


def test_multiple_roots_and_external_targets(project: tuple[Path, Path]) -> None:
    root, config = project
    other = root.parent / "other"
    other.mkdir()
    (root / "AGENTS.md").write_text("root rules")
    (other / "AGENTS.md").write_text("other rules")
    (config / "AGENTS.md").write_text("global rules")
    (root / "linked").symlink_to(other, target_is_directory=True)
    workspace = Workspace.discover(root, root, [other])
    snapshot = load_instructions(workspace, Path("linked/file.py"))
    assert [b.text for b in snapshot.blocks] == ["global rules", "other rules"]
    snapshot = load_instructions(Workspace.discover(root, root), Path("linked/file.py"))
    assert [b.text for b in snapshot.blocks] == ["global rules"]


def test_deduplication_preserves_scopes_and_byte_budget(project: tuple[Path, Path]) -> None:
    root, _ = project
    nested = root / "nested"
    nested.mkdir()
    source = root / "AGENTS.md"
    source.write_text("unique instruction")
    (nested / "AGENTS.md").symlink_to(source)
    workspace = Workspace.discover(nested, root)
    snapshot = load_instructions(workspace, files=[source, source], max_bytes=len(source.read_bytes()))
    assert [b.scope for b in snapshot.blocks] == [root, nested, None]
    assert snapshot.size == len(source.read_bytes())
    assert snapshot.render().count("unique instruction") == 1


def test_instruction_symlink_cannot_escape_root(project: tuple[Path, Path]) -> None:
    root, config = project
    target = config / "outside.md"
    target.write_text("outside rules")
    (root / "AGENTS.md").symlink_to(target)
    with pytest.raises(ValueError, match="resolves outside project root"):
        load_instructions(Workspace.discover(root, root))


@pytest.mark.parametrize("kind", ["broken_symlink", "directory", "invalid_utf8", "fifo"])
def test_invalid_instruction_source_is_an_error(project: tuple[Path, Path], kind: str) -> None:
    import os

    root, _ = project
    path = root / "AGENTS.md"
    if kind == "broken_symlink":
        path.symlink_to(root / "missing")
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="AGENTS.md"):
        load_instructions(Workspace.discover(root, root))


def test_read_and_discovery_errors_are_not_hidden(project: tuple[Path, Path]) -> None:
    root, _ = project
    (root / "AGENTS.md").write_text("rules")
    workspace = Workspace.discover(root, root)
    with patch.object(Path, "open", side_effect=PermissionError("denied")):
        with pytest.raises(ValueError, match="Cannot read instructions.*AGENTS.md"):
            load_instructions(workspace)
    with patch("axio_tools_local.instructions.os.scandir", side_effect=PermissionError("denied")):
        with pytest.raises(ValueError, match="Cannot discover instructions"):
            load_instructions(workspace)


def test_missing_explicit_source_is_an_error(project: tuple[Path, Path]) -> None:
    root, _ = project
    with pytest.raises(ValueError, match="missing.md"):
        load_instructions(Workspace.discover(root, root), files=[Path("missing.md")])


def test_byte_limit_is_cumulative_and_uses_utf8(project: tuple[Path, Path]) -> None:
    root, config = project
    (config / "AGENTS.md").write_text("é")
    (root / "AGENTS.md").write_text("é")
    workspace = Workspace.discover(root, root)
    assert load_instructions(workspace, max_bytes=4).size == 4
    with pytest.raises(ValueError, match="limit 3 bytes exceeded") as error:
        load_instructions(workspace, max_bytes=3)
    assert str(root / "AGENTS.md") in str(error.value)
    assert str(config / "AGENTS.md") in str(error.value)
    with pytest.raises(ValueError, match="greater than zero"):
        load_instructions(workspace, max_bytes=0)


def test_fresh_snapshot_replaces_changed_and_deleted_sources(project: tuple[Path, Path]) -> None:
    root, _ = project
    path = root / "AGENTS.md"
    path.write_text("old rules")
    workspace = Workspace.discover(root, root)
    before = load_instructions(workspace)
    path.write_text("new rules")
    after = load_instructions(workspace)
    assert before.blocks[0].digest != after.blocks[0].digest
    assert "old rules" not in after.render()
    path.unlink()
    assert not load_instructions(workspace).blocks
