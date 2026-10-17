from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from axio_tools_local.paths import LocalPaths as Workspace

from axio_repl.configuration import ConfigError, load_mcp_config, read_config


def write(path: Path, servers: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Workspace, Path]:
    root = tmp_path / "project"
    root.mkdir()
    config = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    return Workspace.discover(root, root), config / "axio" / "mcp.json"


def test_all_layers_replace_whole_entries(roots: tuple[Workspace, Path]) -> None:
    workspace, global_file = roots
    root = workspace.project_root
    write(global_file, {"same": {"type": "http", "url": "https://global.test/mcp", "headers": {"Old": "value"}}})
    write(root / ".mcp.json", {"same": {"type": "stdio", "command": "tool", "args": ["old"], "env": {"A": "b"}}})
    write(root / ".axio/mcp.local.json", {"same": {"type": "stdio", "command": "private"}})
    first = write(root / "first.json", {"same": {"enabled": False}})
    second = write(root / "second.json", {"same": {"type": "http", "url": "https://explicit.test/mcp"}})
    result = load_mcp_config(workspace, files=[first, second])[0]
    assert [s.layer for s in result.history("same")] == ["global", "project", "private", "explicit", "explicit"]
    effective = result.effective["same"]
    assert effective.url == "https://explicit.test/mcp"
    assert effective.headers == {}
    assert effective.args == []
    assert effective.env is None
    assert effective.command is None
    # Callers cannot mutate the stored merge through the returned mapping.
    result.effective.clear()
    assert result.effective["same"] is effective


def test_disabled_server_is_retained(roots: tuple[Workspace, Path]) -> None:
    workspace, global_file = roots
    write(global_file, {"same": {"type": "stdio", "command": "does-not-exist"}})
    write(workspace.project_root / ".mcp.json", {"same": {"enabled": False}})
    result = load_mcp_config(workspace)[0]
    assert not result.effective["same"].enabled
    with pytest.raises(ConfigError, match="disabled server"):
        result.effective["same"].resolve({})


def test_isolated_skips_invalid_automatic_files(roots: tuple[Workspace, Path]) -> None:
    workspace, global_file = roots
    global_file.parent.mkdir(parents=True)
    global_file.write_text("invalid")
    (workspace.project_root / ".mcp.json").write_text("invalid")
    explicit = write(workspace.project_root / "tools.json", {"chosen": {"type": "http", "url": "https://chosen.test"}})
    assert not load_mcp_config(workspace, isolated=True)[0].effective
    assert list(load_mcp_config(workspace, files=[explicit], isolated=True)[0].effective) == ["chosen"]
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_mcp_config(workspace, files=[explicit])


def test_independent_roots_do_not_inherit_parent_mcp(roots: tuple[Workspace, Path]) -> None:
    workspace, global_file = roots
    root = workspace.project_root
    other = root.parent / "other"
    other.mkdir()
    write(root.parent / ".mcp.json", {"unwanted": {"enabled": False}})
    write(global_file, {"global": {"enabled": False}})
    write(root / ".mcp.json", {"same": {"type": "http", "url": "https://first.test"}})
    write(other / ".mcp.json", {"same": {"type": "http", "url": "https://second.test"}})
    first, second = load_mcp_config(Workspace.discover(root, root, [other]))
    assert set(first.effective) == set(second.effective) == {"global", "same"}
    assert first.origin == str(root) and second.origin == str(other)
    assert first.effective["same"].url == "https://first.test"
    assert second.effective["same"].url == "https://second.test"


@pytest.mark.parametrize("layer", ["global", "project", "private", "explicit"])
def test_invalid_layer_is_not_hidden_by_replacement(roots: tuple[Workspace, Path], layer: str) -> None:
    workspace, global_file = roots
    files = [global_file, workspace.project_root / ".mcp.json", workspace.project_root / ".axio/mcp.local.json"]
    files.append(workspace.project_root / "explicit.json")
    for file in files:
        write(file, {"same": {"type": "http", "url": "https://valid.test"}})
    bad = files[["global", "project", "private", "explicit"].index(layer)]
    write(bad, {"same": {"type": "stdio", "command": "tool", "args": "wrong"}})
    with pytest.raises(ConfigError) as error:
        load_mcp_config(workspace, files=[files[-1]])
    assert str(bad) in str(error.value) and "same" in str(error.value) and "args" in str(error.value)


@pytest.mark.parametrize(
    "entry",
    [
        [],
        {},
        {"enabled": 0},
        {"enabled": "false"},
        {"type": "sse", "url": "https://host.test"},
        {"type": "stdio", "command": "tool", "url": "https://host.test"},
        {"type": "stdio", "command": "tool", "args": [1]},
        {"type": "stdio", "command": "tool", "env": {"A": False}},
        {"type": "stdio", "command": "tool", "env": {"A": "one"}, "envFrom": {"A": "REF"}},
        {"type": "http", "url": "file:///etc/passwd"},
        {"type": "http", "url": "https://user:secret@host.test"},
        {"type": "http", "url": "https://host.test:invalid"},
        {"type": "http", "url": "https://host.test", "headers": {"X": "a\r\nb"}},
        {"type": "http", "url": "https://host.test", "headers": {"X": "one", "x": "two"}},
        {"type": "http", "url": "https://host.test", "headers": {"X": "one"}, "headersFrom": {"x": "REF"}},
        {"type": "http", "url": "https://host.test", "timeout": True},
        {"type": "http", "url": "https://host.test", "timeout": 0},
        {"type": "http", "url": "https://host.test", "timeout": float("nan")},
        {"type": "http", "url": "https://host.test", "timeout": 10**400},
        {"type": "http", "url": "https://host.test", "surprise": "secret"},
    ],
)
def test_invalid_fields_are_rejected_without_echoing_values(tmp_path: Path, entry: Any) -> None:
    path = write(tmp_path / ".mcp.json", {"server": entry})
    with pytest.raises(ConfigError) as error:
        read_config(path, "project")
    assert str(path) in str(error.value) and "server" in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("duplicate", ["server", "field", "env"])
def test_duplicate_keys_report_server(tmp_path: Path, duplicate: str) -> None:
    records = {
        "server": '{"mcpServers":{"same":{},"same":{}}}',
        "field": '{"mcpServers":{"same":{"enabled":true,"enabled":false}}}',
        "env": '{"mcpServers":{"same":{"env":{"A":"one","A":"two"}}}}',
    }
    path = tmp_path / ".mcp.json"
    path.write_text(records[duplicate])
    with pytest.raises(ConfigError, match="server 'same'.*duplicate"):
        read_config(path, "project")


@pytest.mark.parametrize("kind", ["missing", "broken", "directory", "utf8", "malformed", "fifo"])
def test_bad_file_never_silently_disappears(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "config.json"
    if kind == "missing":
        assert read_config(path, "project") is None
    elif kind == "broken":
        path.symlink_to(tmp_path / "absent")
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(b"\xff" if kind == "utf8" else b"{secret:invalid}")
    with pytest.raises(ConfigError) as error:
        read_config(path, "explicit", required=True)
    assert "secret" not in str(error.value)


def test_relative_paths_and_literal_args(tmp_path: Path) -> None:
    path = write(
        tmp_path / "configs/mcp.json",
        {
            "s": {
                "type": "stdio",
                "command": "../bin/tool",
                "cwd": "../work",
                "args": ["$TOKEN", "~/x", "$(echo x)", ""],
            }
        },
    )
    config = read_config(path, "explicit")
    assert config is not None
    spec = config.servers[0]
    assert spec.command == str(tmp_path / "bin/tool")
    assert spec.cwd == str(tmp_path / "work")
    assert spec.args == ["$TOKEN", "~/x", "$(echo x)", ""]


def test_executable_and_secret_resolution_is_a_snapshot(tmp_path: Path) -> None:
    executable = tmp_path / "bin/tool"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 1\n")
    executable.chmod(0o755)
    path = write(
        tmp_path / ".mcp.json",
        {
            "s": {
                "type": "stdio",
                "command": "tool",
                "env": {"PATH": "bin", "LOG_LEVEL": "info"},
                "envFrom": {"TOKEN": "SOURCE_TOKEN"},
            }
        },
    )
    config = read_config(path, "project")
    assert config is not None
    env = {"HOME": str(tmp_path), "SOURCE_TOKEN": "secret-value", "UNRELATED_SECRET": "not-inherited"}
    resolved = config.servers[0].resolve(env)
    assert resolved.command == str(executable)
    assert dict(resolved.env or {}) == {
        "HOME": str(tmp_path),
        "PATH": str(tmp_path / "bin"),
        "LOG_LEVEL": "info",
        "TOKEN": "secret-value",
    }
    env["SOURCE_TOKEN"] = "changed"
    assert dict(resolved.env or {})["TOKEN"] == "secret-value"
    assert "secret-value" not in repr(resolved)
    with pytest.raises(ConfigError, match="missing environment variable"):
        config.servers[0].resolve({})


def test_http_references_and_repr_do_not_leak_values(tmp_path: Path) -> None:
    path = write(
        tmp_path / ".mcp.json",
        {
            "s": {
                "type": "http",
                "url": "https://host.test/mcp?key=secret-value",
                "headers": {"X-Key": "secret-value"},
                "headersFrom": {"Authorization": "AUTH"},
            }
        },
    )
    config = read_config(path, "project")
    assert config is not None
    resolved = config.servers[0].resolve({"AUTH": "Bearer secret-value"})
    assert dict(resolved.headers)["authorization"] == "Bearer secret-value"
    assert "secret-value" not in repr(resolved) + repr(config)
    with pytest.raises(ConfigError, match="invalid value"):
        config.servers[0].resolve({"AUTH": "bad\nheader"})


def test_discovery_never_launches_or_resolves_secrets(
    roots: tuple[Workspace, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = roots
    write(
        workspace.project_root / ".mcp.json",
        {
            "s": {
                "type": "stdio",
                "command": "missing-tool",
                "envFrom": {"TOKEN": "MISSING_SECRET"},
            }
        },
    )
    forbidden = Mock(side_effect=AssertionError("side effect during parsing"))
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    monkeypatch.setattr("shutil.which", forbidden)
    config = load_mcp_config(workspace)[0]
    assert config.effective["s"].command == "missing-tool"
    forbidden.assert_not_called()


def test_config_symlink_uses_canonical_parent(tmp_path: Path) -> None:
    target = write(tmp_path / "real/config.json", {"s": {"type": "stdio", "command": "./tool"}})
    alias = tmp_path / ".mcp.json"
    alias.symlink_to(target)
    config = read_config(alias, "project")
    assert config is not None
    assert config.source == target
    assert config.servers[0].command == str(target.parent / "tool")
    assert config.servers[0].cwd == str(target.parent)


def test_inherited_path_and_executable_symlink(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.write_text("#!/bin/sh\nexit 0\n")
    actual.chmod(0o755)
    alias = tmp_path / "tool"
    alias.symlink_to(actual)
    path = write(tmp_path / ".mcp.json", {"s": {"type": "stdio", "command": "tool"}})
    config = read_config(path, "project")
    assert config is not None
    resolved = config.servers[0].resolve({"PATH": str(tmp_path)})
    assert resolved.command == str(alias)
    assert dict(resolved.env or {})["PATH"] == str(tmp_path)
    actual.chmod(0o644)
    with pytest.raises(ConfigError, match="not found or is not executable"):
        config.servers[0].resolve({"PATH": str(tmp_path)})


def test_nonexistent_cwd_fails_only_at_resolution(tmp_path: Path) -> None:
    path = write(tmp_path / ".mcp.json", {"s": {"type": "stdio", "command": "tool", "cwd": "missing"}})
    config = read_config(path, "project")
    assert config is not None
    with pytest.raises(ConfigError, match="cwd is not an existing directory"):
        config.servers[0].resolve({})


@pytest.mark.parametrize(
    "entry,transport", [({"command": "tool"}, "stdio"), ({"url": "https://tools.test/mcp"}, "http")]
)
def test_transport_can_be_inferred(tmp_path: Path, entry: dict[str, str], transport: str) -> None:
    config = read_config(write(tmp_path / ".mcp.json", {"s": entry}), "project")
    assert config is not None and config.servers[0].transport == transport
