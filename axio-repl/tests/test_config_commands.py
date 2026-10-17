"""MCP diagnostics and explicit import preserve scope, secrets and source files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import aiohttp
import pytest
from aiohttp_tiny_mcp.testing import serving
from axio_tools_local.paths import LocalPaths as Workspace
from axio_tools_mcp.import_formats import convert
from axio_tools_mcp.provider import MCPToolProvider
from axio_tools_mcp.runtime import MCPRuntime

import axio_repl as repl
from axio_repl.configuration import ConfigError, load_mcp_config
from axio_repl.mcp_import import plan_import

from .test_mcp_runtime import registry, stdio


def write(path: Path, data: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data).encode()
    path.write_bytes(content)
    return content


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return Workspace.discover(root, root)


@pytest.mark.parametrize("options", [[], ["--why"], ["--config-explain"]])
def test_startup_preserves_mcp_config_diagnostic(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], options: list[str]
) -> None:
    source = workspace.cwd / ".mcp.json"
    source.write_text('{"mcpServers": secret-invalid-json}')
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--isolated", "--mcp-config", str(source), *options])
    with pytest.raises(SystemExit) as error:
        repl.main_sync()
    assert error.value.code == 1
    diagnostic = capsys.readouterr().err
    assert str(source) in diagnostic
    assert "JSON" in diagnostic
    assert "secret-invalid-json" not in diagnostic
    assert "model catalog is invalid" not in diagnostic


@pytest.mark.asyncio
async def test_explain_aliases_connect_without_model_and_redact_values(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = workspace.cwd
    (root / "AGENTS.md").write_text("instruction-body-secret")
    write(
        root / ".mcp.json",
        {
            "mcpServers": {
                "same": {
                    "type": "http",
                    "url": "https://example.test/private-secret?token=query-secret",
                    "headers": {"Authorization": "header-secret"},
                },
                "disabled": {"enabled": False},
                "local": {"command": "/missing/command-secret", "args": ["arg-secret"], "env": {"KEY": "env-secret"}},
            }
        },
    )
    write(root / ".axio/mcp.local.json", {"mcpServers": {"same": {"type": "http", "url": "https://other.test/mcp"}}})
    forbidden = Mock(side_effect=AssertionError("must not start a model"))
    monkeypatch.setattr(repl, "_select_transport", forbidden)
    monkeypatch.setattr(repl, "setup_history", forbidden)
    outputs: list[str] = []
    async with serving(registry()) as url:
        write(
            root / ".axio/mcp.local.json",
            {"mcpServers": {"same": {"type": "http", "url": url}, "stdio": stdio()}},
        )
        for alias in ("--config-explain", "--why"):
            runtime = MCPRuntime()
            provider = MCPToolProvider()
            provider.runtime = runtime
            monkeypatch.setattr("axio_tools_mcp.provider.MCPToolProvider", lambda: provider)
            monkeypatch.setattr(sys, "argv", ["axio-repl", alias])
            await repl.main()
            outputs.append(capsys.readouterr().out)
            sessions = [server.session for server in runtime.servers if server.session is not None]
            assert len(sessions) == 2
            assert all(not session.is_connected for session in sessions)
            assert not runtime.tools
    assert outputs[0] == outputs[1]
    output = outputs[0]
    assert "secret" not in output
    assert "shadowed (whole entry replaced)" in output
    assert 'Server "disabled": disabled' in output
    assert f"cwd: {root}" in output and "Resolution error:" in output
    assert "Endpoint origin: http://127.0.0.1:" in output
    assert "'same'  1  connected" in output
    assert "'stdio'  3  connected" in output
    assert "'local'  0  error" in output
    assert "'disabled'  0  disabled" in output
    assert "Active: 2; tools: 4" in output
    forbidden.assert_not_called()


def test_claude_user_and_project_local_scopes_are_separate(workspace: Workspace, tmp_path: Path) -> None:
    home = tmp_path / "home"
    user = {"type": "http", "url": "https://user.test/mcp"}
    local = {"type": "http", "url": "https://local.test/mcp"}
    source = home / ".claude.json"
    original = write(
        source,
        {"mcpServers": {"same": user}, "projects": {str(workspace.project_root): {"mcpServers": {"same": local}}}},
    )
    first = plan_import(workspace, "claude", "user", "private", home=home)
    second = plan_import(workspace, "claude", "local", "private", home=home)
    assert first.data["mcpServers"]["same"]["url"] == user["url"]
    assert second.data["mcpServers"]["same"]["url"] == local["url"]
    assert not first.destination.parent.exists()
    second.write()
    assert source.read_bytes() == original
    assert second.destination.stat().st_mode & 0o777 == 0o600
    assert load_mcp_config(workspace)[0].effective["same"].url == local["url"]


def test_cursor_paths_environment_refs_and_copy(workspace: Workspace, tmp_path: Path) -> None:
    source = workspace.project_root / ".cursor/mcp.json"
    original = write(
        source,
        {
            "mcpServers": {
                "local": {
                    "command": "./bin/tool",
                    "args": ["scripts/server.py"],
                    "env": {"TOKEN": "${env:SOURCE_TOKEN}"},
                }
            }
        },
    )
    with pytest.raises(ConfigError, match="server-cwd"):
        plan_import(workspace, "cursor", "project", "private")
    plan = plan_import(workspace, "cursor", "project", "private", server_cwd=Path("."))
    item = plan.data["mcpServers"]["local"]
    assert item["command"] == str(workspace.cwd / "bin/tool")
    assert item["cwd"] == str(workspace.cwd)
    assert item["args"] == ["scripts/server.py"]
    assert item["envFrom"] == {"TOKEN": "SOURCE_TOKEN"}
    plan.write()
    assert source.read_bytes() == original
    loaded = load_mcp_config(workspace)[0].effective["local"]
    assert loaded.cwd == str(workspace.cwd) and loaded.env_from == {"TOKEN": "SOURCE_TOKEN"}


def test_project_scope_is_not_guessed_from_home_path(workspace: Workspace, tmp_path: Path) -> None:
    home = workspace.cwd
    write(home / ".cursor/mcp.json", {"mcpServers": {"server": {"url": "https://example.test"}}})
    plan = plan_import(workspace, "cursor", "project", "private", home=home)
    assert plan.scope == "project"
    with pytest.raises(ValueError, match="Cursor has user and project"):
        plan_import(workspace, "cursor", "local", "private", home=home)


def test_conflicts_require_explicit_replacement_and_preserve_other_entries(workspace: Workspace) -> None:
    source = workspace.cwd / ".cursor/mcp.json"
    write(source, {"mcpServers": {"same": {"url": "https://new.test"}}})
    destination = workspace.cwd / ".axio/mcp.local.json"
    original = write(destination, {"mcpServers": {"same": {"enabled": False}, "keep": {"enabled": False}}})
    plan = plan_import(workspace, "cursor", "project", "private")
    assert plan.conflicts == ("same",) and '"same": replace' in plan.describe()
    with pytest.raises(ValueError, match="--replace"):
        plan.write()
    assert destination.read_bytes() == original
    plan.write(replace=True)
    result = json.loads(destination.read_text())
    assert result["mcpServers"]["keep"] == {"enabled": False}
    assert result["mcpServers"]["same"] == {"type": "http", "url": "https://new.test"}


def test_destination_changed_after_preview_is_not_overwritten(workspace: Workspace) -> None:
    write(workspace.cwd / ".cursor/mcp.json", {"mcpServers": {"s": {"url": "https://example.test"}}})
    plan = plan_import(workspace, "cursor", "project", "private")
    changed = write(plan.destination, {"mcpServers": {"changed": {"enabled": False}}})
    with pytest.raises(ValueError, match="changed after preview"):
        plan.write(replace=True)
    assert plan.destination.read_bytes() == changed


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "sse", "url": "https://example.test"},
        {"command": "tool", "envFile": ".env"},
        {"url": "https://example.test", "auth": {"CLIENT_SECRET": "secret"}},
        {"command": "tool", "args": ["${workspaceFolder}/secret"]},
        {"command": "tool", "env": {"TOKEN": "${VAR:-secret}"}},
        {"url": "https://example.test", "headers": {"Authorization": "Bearer ${env:TOKEN}"}},
    ],
)
def test_unsupported_features_fail_without_exposing_values(workspace: Workspace, entry: dict[str, Any]) -> None:
    source = workspace.cwd / ".cursor/mcp.json"
    write(source, {"mcpServers": {"bad": entry}})
    with pytest.raises(ConfigError) as error:
        plan_import(workspace, "cursor", "project", "private", server_cwd=workspace.cwd)
    assert "bad" in str(error.value) and "secret" not in str(error.value)


def test_claude_exact_refs_and_http_alias(workspace: Workspace) -> None:
    result = convert(
        "server",
        {
            "type": "streamable-http",
            "url": "https://example.test",
            "headers": {"Authorization": "${AUTHORIZATION}"},
        },
        "claude",
        workspace.cwd / ".claude.json",
        None,
    )
    assert result == {"type": "http", "url": "https://example.test", "headersFrom": {"authorization": "AUTHORIZATION"}}
    with pytest.raises(ConfigError, match="must specify type"):
        convert("s", {"url": "https://example.test"}, "claude", workspace.cwd / ".claude.json", None)


def test_secret_values_use_private_destination_and_stay_out_of_preview(workspace: Workspace) -> None:
    source = workspace.cwd / ".cursor/mcp.json"
    original = write(
        source,
        {
            "mcpServers": {
                "s": {
                    "url": "https://example.test/secret-path?key=secret-query",
                    "headers": {"Authorization": "secret-header"},
                }
            }
        },
    )
    with pytest.raises(ConfigError, match="--target private or user"):
        plan_import(workspace, "cursor", "project", "project")
    plan = plan_import(workspace, "cursor", "project", "private")
    assert all(value not in plan.describe() + repr(plan) for value in ("secret-path", "secret-query", "secret-header"))
    plan.write()
    assert source.read_bytes() == original
    assert json.loads(plan.destination.read_bytes())["mcpServers"]["s"]["headers"]["authorization"] == "secret-header"


def test_source_destination_alias_is_rejected(workspace: Workspace) -> None:
    write(workspace.cwd / ".mcp.json", {"mcpServers": {"s": {"type": "http", "url": "https://example.test"}}})
    with pytest.raises(ValueError, match="must differ"):
        plan_import(workspace, "claude", "project", "project")


@pytest.mark.asyncio
async def test_import_cli_preview_and_write_never_select_transport(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(workspace.cwd / ".cursor/mcp.json", {"mcpServers": {"s": {"url": "https://example.test"}}})
    forbidden = Mock(side_effect=AssertionError("must remain offline"))
    monkeypatch.setattr(repl, "_select_transport", forbidden)
    monkeypatch.setattr(aiohttp, "ClientSession", forbidden)
    arguments = ["axio-repl", "mcp", "import", "--from", "cursor", "--scope", "project", "--target", "project"]
    monkeypatch.setattr(sys, "argv", arguments)
    await repl.main()
    assert "Preview only" in capsys.readouterr().out
    assert not (workspace.cwd / ".mcp.json").exists()
    monkeypatch.setattr(sys, "argv", [*arguments, "--write"])
    await repl.main()
    assert "Copied 1" in capsys.readouterr().out
    assert (workspace.cwd / ".mcp.json").exists()
    forbidden.assert_not_called()
