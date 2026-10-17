"""Tests for MCPServerConfig."""

from __future__ import annotations

import pytest

from axio_tools_mcp.config import MCPServerConfig


def test_stdio_config() -> None:
    cfg = MCPServerConfig(name="test", command="python", args=["-m", "server"])
    assert cfg.command == "python"
    assert cfg.args == ["-m", "server"]
    assert cfg.url is None


def test_http_config() -> None:
    cfg = MCPServerConfig(name="test", url="http://localhost:8000/mcp")
    assert cfg.url == "http://localhost:8000/mcp"
    assert cfg.command is None


def test_both_set_raises() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        MCPServerConfig(name="test", command="python", url="http://localhost")


def test_neither_set_raises() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        MCPServerConfig(name="test")


def test_frozen() -> None:
    cfg = MCPServerConfig(name="test", command="python")
    with pytest.raises(AttributeError):
        cfg.name = "other"  # type: ignore[misc]


def test_to_dict_stdio() -> None:
    cfg = MCPServerConfig(
        name="fs",
        command="npx",
        args=["-y", "@mcp/filesystem"],
        env={"HOME": "/tmp"},
    )
    d = cfg.to_settings()
    assert d["command"] == "npx"
    assert '"HOME"' in d["args"] or "-y" in d["args"]
    assert "env" in d
    assert "url" not in d


def test_to_dict_http() -> None:
    cfg = MCPServerConfig(
        name="remote",
        url="http://example.com/mcp",
        headers={"Authorization": "Bearer token"},
        timeout=60.0,
    )
    d = cfg.to_settings()
    assert d["url"] == "http://example.com/mcp"
    assert "headers" in d
    assert d["timeout"] == "60.0"
    assert "command" not in d


def test_roundtrip_stdio() -> None:
    original = MCPServerConfig(
        name="fs",
        command="npx",
        args=["-y", "@mcp/filesystem"],
        env={"HOME": "/tmp"},
        timeout=45.0,
    )
    restored = MCPServerConfig.from_settings("fs", original.to_settings())
    assert restored == original


def test_roundtrip_http() -> None:
    original = MCPServerConfig(
        name="remote",
        url="http://example.com/mcp",
        headers={"Authorization": "Bearer xxx"},
    )
    restored = MCPServerConfig.from_settings("remote", original.to_settings())
    assert restored == original


def test_default_timeout() -> None:
    cfg = MCPServerConfig(name="test", command="python")
    assert cfg.timeout == 30.0


def test_to_dict_omits_defaults() -> None:
    cfg = MCPServerConfig(name="test", command="python")
    d = cfg.to_settings()
    assert "timeout" not in d
    assert "args" not in d
    assert "env" not in d
    assert "headers" not in d


def test_roundtrip_explicit_child_environment_and_cwd() -> None:
    original = MCPServerConfig(name="local", command="tool", env={"A": "b"}, cwd="/tmp/work", inherit_env=False)
    assert MCPServerConfig.from_settings("local", original.to_settings()) == original
    defaults = MCPServerConfig.from_settings("local", {"command": "tool"})
    assert defaults.inherit_env is True and defaults.cwd is None


def test_native_json_entry_roundtrip() -> None:
    data = {
        "command": "python",
        "args": ["server.py"],
        "cwd": "/work",
        "env": {"MODE": "test"},
        "envFrom": {"TOKEN": "HOST_TOKEN"},
        "timeout": 12,
    }
    config = MCPServerConfig.from_dict("example", data)
    assert config.args == ["server.py"]
    assert config.env_from == {"TOKEN": "HOST_TOKEN"}
    assert config.timeout == 12
    assert MCPServerConfig.from_dict("example", config.to_dict()) == config
    assert MCPServerConfig.from_dict("off", {"enabled": False}).enabled is False


@pytest.mark.parametrize(
    "data",
    [
        {"command": "tool", "args": "[]"},
        {"url": "https://example.test", "headers": {"Authorization": "line\nbreak"}},
        {"command": "tool", "timeout": True},
        {"command": "tool", "type": []},
        {"command": "tool", "env": {"TOKEN": "x"}, "envFrom": {"TOKEN": "HOST_TOKEN"}},
    ],
)
def test_native_json_rejects_invalid_fields(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MCPServerConfig.from_dict("invalid", data)


def test_oauth_config_roundtrips() -> None:
    entry = {
        "url": "https://example.com/mcp",
        "oauth": {
            "issuer": "https://issuer.example.com",
            "client_id": "registered",
            "scope": "read write",
            "redirect_port": 8765,
        },
    }
    config = MCPServerConfig.from_dict("docs", entry)
    assert MCPServerConfig.from_dict("docs", config.to_dict()) == config
    assert MCPServerConfig.from_settings("docs", config.to_settings()) == config
    assert config.resolve().oauth == config.oauth


@pytest.mark.parametrize(
    "options",
    [
        True,
        [],
        {"secret": "no"},
        {"redirect_port": True},
        {"redirect_port": 65536},
        {"client_id": "id"},
        {"scope": []},
    ],
)
def test_invalid_oauth_options(options: object) -> None:
    with pytest.raises(ValueError, match="oauth"):
        MCPServerConfig.from_dict("docs", {"url": "https://example.com/mcp", "oauth": options})


def test_oauth_rejects_authorization_header() -> None:
    for field in ("headers", "headersFrom"):
        with pytest.raises(ValueError, match="Authorization"):
            MCPServerConfig.from_dict(
                "docs",
                {
                    "url": "https://example.com/mcp",
                    "oauth": {},
                    field: {"Authorization": "SECRET"},
                },
            )
