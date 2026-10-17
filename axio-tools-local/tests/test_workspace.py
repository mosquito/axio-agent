from __future__ import annotations

from pathlib import Path

import pytest

from axio_tools_local.paths import LocalPaths as Workspace
from axio_tools_local.paths import config_directory


@pytest.mark.parametrize("worktree", [False, True])
def test_nearest_git_root_preserves_cwd(tmp_path: Path, worktree: bool) -> None:
    (tmp_path / ".git").mkdir()
    root = tmp_path / "project"
    cwd = root / "services" / "api"
    cwd.mkdir(parents=True)
    if worktree:
        (root / ".git").write_text("gitdir: /somewhere/else/worktrees/project\n")
    else:
        (root / ".git").mkdir()
    before = Path.cwd()
    workspace = Workspace.discover(cwd)
    assert workspace.project_root == root.resolve()
    assert workspace.cwd == cwd.resolve()
    assert Path.cwd() == before
    assert workspace.directories_for(Path("main.py")) == (root, root / "services", cwd)


def test_no_git_uses_cwd_and_does_not_walk_into_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".git").mkdir()
    cwd = tmp_path / "plain" / "nested"
    cwd.mkdir(parents=True)
    workspace = Workspace.discover(cwd)
    assert workspace.roots == (cwd,)


def test_explicit_root_overrides_git_and_must_contain_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "nested"
    cwd.mkdir()
    (cwd / ".git").mkdir()
    assert Workspace.discover(cwd, Path("..")).project_root == tmp_path
    with pytest.raises(ValueError, match="outside project root"):
        Workspace.discover(tmp_path, cwd)
    with pytest.raises(ValueError, match="not an existing directory"):
        Workspace.discover(cwd, Path("missing"))


def test_roots_are_independent_and_canonical(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    workspace = Workspace.discover(alias, additional_roots=[second, alias, second])
    assert workspace.roots == (first, second)
    assert workspace.root_for(second / "new.py") == second
    assert workspace.root_for(tmp_path / "outside.py") is None
    assert workspace.directories_for(tmp_path / "outside.py") == ()
    assert workspace.directories_for(second / "new.py") == (second,)


@pytest.mark.parametrize("parent_first", [True, False])
def test_nested_roots_are_rejected(tmp_path: Path, parent_first: bool) -> None:
    child = tmp_path / "child"
    child.mkdir()
    primary, additional = (tmp_path, child) if parent_first else (child, tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        Workspace.discover(primary, primary, [additional])


def test_symlink_target_uses_its_own_root(tmp_path: Path) -> None:
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (root / "linked").symlink_to(other, target_is_directory=True)
    workspace = Workspace.discover(root, root, [other])
    assert workspace.directories_for(Path("linked/new.py")) == (other,)
    assert Workspace.discover(root, root).directories_for(Path("linked/new.py")) == ()


@pytest.mark.parametrize("xdg", ["", "relative", "absolute"])
def test_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg: str) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    value = str(tmp_path / "custom") if xdg == "absolute" else xdg
    monkeypatch.setenv("XDG_CONFIG_HOME", value)
    expected = tmp_path / ("custom" if xdg == "absolute" else ".config") / "axio"
    assert config_directory() == expected
