"""Discover standard SKILL.md metadata and read local resources on demand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import axio_tools_local.skills as module
from axio_tools_local.skills import LocalSkills, discover, read_header


def write_skill(root: Path, name: str = "planning", *, extra: str = "", body: str = "Instructions.") -> Path:
    source = root / name / "SKILL.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"---\nname: {name}\ndescription: Plan work\n{extra}---\n{body}\n")
    return source.resolve()


@pytest.mark.asyncio
async def test_home_directories_and_isolated_explicit_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    config = tmp_path / "config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    locations = [
        write_skill(root, name)
        for root, name in (
            (home / ".agents/skills", "shared"),
            (home / ".claude/skills", "claude"),
            (home / ".codex/skills/.system", "system"),
            (home / ".cursor/skills", "cursor"),
            (config / "axio/skills", "axio"),
            (config / "opencode/skills", "opencode"),
        )
    ]
    write_skill(home / "unrelated", "outside")
    local = LocalSkills(tmp_path, directories=[])
    await local.reload()
    assert set(local.catalog.skills) == {str(path) for path in locations}
    isolated = LocalSkills(tmp_path, directories=[], isolated=True)
    await isolated.reload()
    assert not isolated.tools
    explicit = LocalSkills(tmp_path, directories=[Path("~/unrelated")], isolated=True)
    await explicit.reload()
    assert [skill.name for skill in explicit.catalog.skills.values()] == ["outside"]


@pytest.mark.asyncio
async def test_yaml_metadata_lazy_body_and_relative_files(tmp_path: Path) -> None:
    source = write_skill(tmp_path, extra="compatibility: Requires git\nallowed-tools: Bash\n")
    source.write_text(
        "\ufeff---\nname: planning\ndescription: >-\n  Plan work:\n  retain constraints\n"
        "compatibility: Requires git\nmetadata:\n  author: example\n---\nDo the work.\n"
    )
    reference = source.parent / "references/details.md"
    reference.parent.mkdir()
    reference.write_text("Точные требования.\n")
    local = LocalSkills(tmp_path, directories=[tmp_path], isolated=True)
    await local.reload()
    skill = local.catalog.skills[str(source)]
    assert skill.description == "Plan work: retain constraints"
    assert skill.compatibility == "Requires git"
    assert "Do the work" not in skill.description
    result = json.loads(await local.tools[0](location=str(source)))
    assert "Do the work" in result["text"]
    assert result["directory"] == str(source.parent)
    assert json.loads(await local.read(str(source), "references/details.md"))["text"] == "Точные требования.\n"


@pytest.mark.asyncio
async def test_discovery_reads_only_header(tmp_path: Path) -> None:
    source = write_skill(tmp_path)
    source.write_bytes(source.read_bytes() + b"\xff")
    local = LocalSkills(tmp_path, directories=[tmp_path], isolated=True)
    await local.reload()
    assert str(source) in local.catalog.skills
    with pytest.raises(UnicodeDecodeError):
        await local.read(str(source))


def test_symlinks_deduplicate_sources_and_terminate_cycles(tmp_path: Path) -> None:
    source = write_skill(tmp_path / "first")
    second = tmp_path / "second"
    second.mkdir()
    (second / "alias").symlink_to(source.parent, target_is_directory=True)
    (source.parent / "loop").symlink_to(tmp_path, target_is_directory=True)
    catalog = discover(((tmp_path / "first", True), (second, True)))
    assert list(catalog.skills) == [str(source)]


@pytest.mark.parametrize("body", ["Instructions.", "Different instructions."])
def test_identical_names_warn_only_for_different_content(tmp_path: Path, body: str) -> None:
    first = write_skill(tmp_path / "first")
    second = write_skill(tmp_path / "second")
    third = write_skill(tmp_path / "third", body=body)
    catalog = discover(((tmp_path, True),))
    assert set(catalog.skills) == {str(first), str(second), str(third)}
    assert bool(catalog.diagnostics) == (body != "Instructions.")


def test_duplicate_content_above_limit_keeps_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_skill(tmp_path / "first")
    write_skill(tmp_path / "second")
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 1)
    catalog = discover(((tmp_path, True),))
    assert len(catalog.skills) == 2
    assert any("Duplicate local skill name" in text for text in catalog.diagnostics)


def test_duplicate_disappearing_after_header_keeps_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_skill(tmp_path / "first")
    second = write_skill(tmp_path / "second")

    def remove_after_header(source: Path) -> tuple[dict[str, object], str]:
        header, digest = read_header(source)
        if source == second:
            source.unlink()
        return header, digest

    monkeypatch.setattr(module, "read_header", remove_after_header)
    catalog = discover(((tmp_path, True),))
    assert len(catalog.skills) == 2
    assert any("Duplicate local skill name" in text for text in catalog.diagnostics)


@pytest.mark.parametrize(
    "content",
    [
        "# No frontmatter",
        "---\nname: missing-end",
        "---\nname: plan\n---\n",
        "---\nname: plan\ndescription: ''\n---\n",
        "---\nname: [plan]\ndescription: Work\n---\n",
        "---\nname: plan\ndescription: Work\ncompatibility: [git]\n---\n",
        "---\n!!python/object/apply:os.system ['false']\n---\n",
        "---\n- not-a-mapping\n---\n",
    ],
)
def test_invalid_skill_does_not_hide_healthy_siblings(tmp_path: Path, content: str) -> None:
    good = write_skill(tmp_path, "good")
    bad = write_skill(tmp_path, "bad")
    bad.write_text(content)
    catalog = discover(((tmp_path, True),))
    assert list(catalog.skills) == [str(good)]
    assert catalog.diagnostics


def test_exact_filename_and_nonstandard_name(tmp_path: Path) -> None:
    source = write_skill(tmp_path)
    source.rename(source.with_name("skill.md"))
    assert not discover(((tmp_path, True),)).skills
    source = write_skill(tmp_path, "Nonstandard_Name")
    catalog = discover(((tmp_path, True),))
    assert str(source) in catalog.skills
    assert any("nonstandard skill name" in text for text in catalog.diagnostics)


@pytest.mark.parametrize("extra", ["disable-model-invocation: true\n", "metadata:\n  opencode/autoinvoke: 'false'\n"])
def test_disabled_automatic_invocation_is_not_exposed(tmp_path: Path, extra: str) -> None:
    write_skill(tmp_path, extra=extra)
    catalog = discover(((tmp_path, True),))
    assert not catalog.skills
    assert any("automatic invocation is disabled" in text for text in catalog.diagnostics)


@pytest.mark.asyncio
async def test_reads_reject_unknown_locations_and_escaping_paths(tmp_path: Path) -> None:
    source = write_skill(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (source.parent / "link.txt").symlink_to(outside)
    local = LocalSkills(tmp_path, directories=[tmp_path], isolated=True)
    await local.reload()
    with pytest.raises(ValueError, match="Unknown local skill"):
        await local.read(str(outside))
    for path in (str(outside), "../outside.txt", "link.txt"):
        with pytest.raises(ValueError, match="relative|outside"):
            await local.read(str(source), path)


@pytest.mark.asyncio
async def test_metadata_change_requires_reload_and_removal_drops_tool(tmp_path: Path) -> None:
    source = write_skill(tmp_path)
    local = LocalSkills(tmp_path, directories=[tmp_path], isolated=True)
    await local.reload()
    source.write_text(source.read_text().replace("Plan work", "Review work"))
    with pytest.raises(ValueError, match="metadata changed"):
        await local.read(str(source))
    await local.reload()
    assert local.catalog.skills[str(source)].description == "Review work"
    assert "Review work" in await local.read(str(source))
    source.unlink()
    await local.reload()
    assert not local.tools


@pytest.mark.asyncio
async def test_content_limit_fails_without_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = write_skill(tmp_path)
    local = LocalSkills(tmp_path, directories=[tmp_path], isolated=True)
    await local.reload()
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        await local.read(str(source))


def test_header_and_discovery_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = write_skill(tmp_path / "a/b/c")
    monkeypatch.setattr(module, "MAX_HEADER_BYTES", 20)
    with pytest.raises(ValueError, match="exceeds"):
        read_header(source)
    monkeypatch.setattr(module, "MAX_HEADER_BYTES", 65536)
    monkeypatch.setattr(module, "MAX_DEPTH", 1)
    catalog = discover(((tmp_path, True), (source.parent, True)))
    assert str(source) in catalog.skills
    assert any("depth limit" in text for text in catalog.diagnostics)
    monkeypatch.setattr(module, "MAX_DIRECTORIES", 1)
    catalog = discover(((tmp_path, True),))
    assert not catalog.skills
    assert any("directory limit" in text for text in catalog.diagnostics)
