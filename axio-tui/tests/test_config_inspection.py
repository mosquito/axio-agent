from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from axio_tools_mcp.runtime import MCPRuntime, Server

from axio_tui.configuration import ConfigFile, ConfiguredServer, MCPConfiguration, inspect_configurations


def test_inspection_preserves_configuration_history_and_failures(tmp_path: Path) -> None:
    earlier = ConfiguredServer(
        "docs", source=tmp_path / "global.json", layer="global", url="https://old.example/token"
    )
    effective = replace(
        earlier,
        source=tmp_path / ".mcp.json",
        layer="project",
        url="https://new.example/secret?token=secret",
        headers={"authorization": "secret"},
        headers_from={"x-key": "MY_KEY"},
    )
    project = MCPConfiguration(
        str(tmp_path),
        (
            ConfigFile(earlier.source, earlier.layer, (earlier,)),
            ConfigFile(effective.source, effective.layer, (effective,)),
        ),
    )
    runtime = MCPRuntime()
    runtime.servers = [Server(effective, status="connected", resource_error="ValueError")]
    info = inspect_configurations((project,), runtime.inspect_servers())[0]
    details = dict(info.details)
    assert info.origin == "Automatic · project"
    assert details["Configuration"] == str(tmp_path / ".mcp.json")
    assert details["Replaced configuration"] == f"global: {tmp_path / 'global.json'}"
    assert details["Endpoint origin"] == "https://new.example"
    assert details["Header references"] == "1"
    assert "secret" not in str(info.details)
    assert info.errors == ("Resource discovery failed: ValueError",)
    runtime.servers[0].status = "disabled"
    assert runtime.inspect_servers()[0].state == "disabled"
    runtime.servers[0].status = "error"
    runtime.servers[0].error = "Missing environment variable MY_KEY"
    assert "Missing environment variable MY_KEY" in runtime.inspect_servers()[0].errors
