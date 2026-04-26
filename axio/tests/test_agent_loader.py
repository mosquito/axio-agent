"""Tests for axio.agent_loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from axio.agent import Agent
from axio.agent_loader import (
    AgentSpec,
    IniAgentLoader,
    JsonAgentLoader,
    TomlAgentLoader,
    load_agents,
    make_agent_tools,
)
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_echo_tool, make_text_response
from axio.tool import Tool
from axio.transport import DummyCompletionTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toolbox() -> dict[str, Tool[Any]]:
    return {"echo": make_echo_tool()}


# ---------------------------------------------------------------------------
# AgentSpec.to_agent
# ---------------------------------------------------------------------------


class TestAgentSpec:
    def test_to_agent_returns_prototype(self) -> None:
        spec = AgentSpec(name="x", description="d", system="sys")
        agent = spec.to_agent()
        assert isinstance(agent, Agent)
        assert isinstance(agent.transport, DummyCompletionTransport)
        assert agent.system == "sys"
        assert agent.tools == []

    def test_to_agent_resolves_tools(self) -> None:
        spec = AgentSpec(name="x", description="d", system="s", tools=("echo",))
        agent = spec.to_agent(_toolbox())
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "echo"

    def test_to_agent_unknown_tool_raises(self) -> None:
        spec = AgentSpec(name="x", description="d", system="s", tools=("unknown_tool",))
        with pytest.raises(KeyError, match="unknown_tool"):
            spec.to_agent(_toolbox())

    def test_max_iterations_default(self) -> None:
        spec = AgentSpec(name="x", description="d", system="s")
        assert spec.max_iterations == 50

    def test_max_iterations_respected(self) -> None:
        spec = AgentSpec(name="x", description="d", system="s", max_iterations=99)
        assert spec.to_agent().max_iterations == 99


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------


class TestTomlAgentLoader:
    def test_full_spec(self, tmp_path: Path) -> None:
        (tmp_path / "arch.toml").write_text(
            'name = "architect"\ndescription = "Design"\nmax_iterations = 77\n'
            'tools = ["echo"]\nmodel = "big"\n\n[system]\ntext = "You are an architect."\n',
            encoding="utf-8",
        )
        spec = TomlAgentLoader().load_file(tmp_path / "arch.toml")
        assert spec.name == "architect"
        assert spec.description == "Design"
        assert spec.max_iterations == 77
        assert spec.tools == ("echo",)
        assert spec.model == "big"
        assert spec.system == "You are an architect."

    def test_name_falls_back_to_stem(self, tmp_path: Path) -> None:
        (tmp_path / "myagent.toml").write_text('system = "hello"\n', encoding="utf-8")
        spec = TomlAgentLoader().load_file(tmp_path / "myagent.toml")
        assert spec.name == "myagent"

    def test_load_from_string(self) -> None:
        spec = TomlAgentLoader().load('name = "x"\nsystem = "s"\n')
        assert spec.name == "x"
        assert spec.system == "s"

    def test_system_inline_string(self, tmp_path: Path) -> None:
        (tmp_path / "a.toml").write_text('system = "inline prompt"\n', encoding="utf-8")
        spec = TomlAgentLoader().load_file(tmp_path / "a.toml")
        assert spec.system == "inline prompt"

    def test_system_as_dict_text(self, tmp_path: Path) -> None:
        (tmp_path / "a.toml").write_text('[system]\ntext = "nested"\n', encoding="utf-8")
        spec = TomlAgentLoader().load_file(tmp_path / "a.toml")
        assert spec.system == "nested"

    def test_invalid_toml_raises_with_path(self, tmp_path: Path) -> None:
        (tmp_path / "bad.toml").write_text("not = valid = toml", encoding="utf-8")
        with pytest.raises(ValueError, match="bad.toml"):
            TomlAgentLoader().load_file(tmp_path / "bad.toml")

    def test_scan_returns_agents(self, tmp_path: Path) -> None:
        (tmp_path / "worker.toml").write_text(
            'name = "worker"\ndescription = "Works"\nsystem = "Do work."\ntools = ["echo"]\n',
            encoding="utf-8",
        )
        result = TomlAgentLoader().scan(tmp_path, _toolbox())
        assert "worker" in result
        desc, agent = result["worker"]
        assert desc == "Works"
        assert isinstance(agent, Agent)

    def test_scan_ignores_other_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "x.json").write_text('{"system": "s"}', encoding="utf-8")
        assert TomlAgentLoader().scan(tmp_path) == {}


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------


class TestJsonAgentLoader:
    def test_full_spec(self, tmp_path: Path) -> None:
        data = {
            "name": "planner",
            "description": "Plans things",
            "system": "You plan.",
            "max_iterations": 30,
            "tools": ["echo"],
            "model": "small",
        }
        (tmp_path / "planner.json").write_text(json.dumps(data), encoding="utf-8")
        spec = JsonAgentLoader().load_file(tmp_path / "planner.json")
        assert spec.name == "planner"
        assert spec.description == "Plans things"
        assert spec.system == "You plan."
        assert spec.max_iterations == 30
        assert spec.tools == ("echo",)
        assert spec.model == "small"

    def test_name_falls_back_to_stem(self, tmp_path: Path) -> None:
        (tmp_path / "myagent.json").write_text('{"system": "s"}', encoding="utf-8")
        spec = JsonAgentLoader().load_file(tmp_path / "myagent.json")
        assert spec.name == "myagent"

    def test_load_from_string(self) -> None:
        spec = JsonAgentLoader().load('{"name": "x", "system": "s"}')
        assert spec.name == "x"

    def test_system_as_dict_text(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text('{"system": {"text": "nested"}}', encoding="utf-8")
        spec = JsonAgentLoader().load_file(tmp_path / "a.json")
        assert spec.system == "nested"

    def test_invalid_json_raises_with_path(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="bad.json"):
            JsonAgentLoader().load_file(tmp_path / "bad.json")

    def test_non_object_json_raises_with_path(self, tmp_path: Path) -> None:
        (tmp_path / "arr.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="arr.json"):
            JsonAgentLoader().load_file(tmp_path / "arr.json")

    def test_scan_returns_agents(self, tmp_path: Path) -> None:
        (tmp_path / "bot.json").write_text(
            '{"name": "bot", "description": "Bots", "system": "Go.", "tools": ["echo"]}',
            encoding="utf-8",
        )
        result = JsonAgentLoader().scan(tmp_path, _toolbox())
        assert "bot" in result


# ---------------------------------------------------------------------------
# INI loader
# ---------------------------------------------------------------------------


class TestIniAgentLoader:
    def test_full_spec(self, tmp_path: Path) -> None:
        content = (
            "[agent]\n"
            "name = worker\n"
            "description = Does work\n"
            "max_iterations = 20\n"
            "tools = echo, read_file\n"
            "model = medium\n\n"
            "[system]\n"
            "text = You are a worker.\n"
        )
        (tmp_path / "worker.ini").write_text(content, encoding="utf-8")
        spec = IniAgentLoader().load_file(tmp_path / "worker.ini")
        assert spec.name == "worker"
        assert spec.description == "Does work"
        assert spec.max_iterations == 20
        assert spec.tools == ("echo", "read_file")
        assert spec.model == "medium"
        assert spec.system == "You are a worker."

    def test_name_falls_back_to_stem(self, tmp_path: Path) -> None:
        (tmp_path / "myagent.ini").write_text("[agent]\n", encoding="utf-8")
        spec = IniAgentLoader().load_file(tmp_path / "myagent.ini")
        assert spec.name == "myagent"

    def test_load_from_string(self) -> None:
        spec = IniAgentLoader().load("[agent]\nname = x\n\n[system]\ntext = hello\n")
        assert spec.name == "x"
        assert spec.system == "hello"

    def test_system_in_agent_section(self, tmp_path: Path) -> None:
        (tmp_path / "a.ini").write_text("[agent]\nsystem = inline system\n", encoding="utf-8")
        spec = IniAgentLoader().load_file(tmp_path / "a.ini")
        assert spec.system == "inline system"

    def test_empty_tools(self, tmp_path: Path) -> None:
        (tmp_path / "a.ini").write_text("[agent]\n", encoding="utf-8")
        spec = IniAgentLoader().load_file(tmp_path / "a.ini")
        assert spec.tools == ()

    def test_scan_returns_agents(self, tmp_path: Path) -> None:
        content = "[agent]\nname = bot\ndescription = A bot\ntools = echo\n\n[system]\ntext = Do it.\n"
        (tmp_path / "bot.ini").write_text(content, encoding="utf-8")
        result = IniAgentLoader().scan(tmp_path, _toolbox())
        assert "bot" in result


# ---------------------------------------------------------------------------
# load_agents
# ---------------------------------------------------------------------------


class TestLoadAgents:
    def test_mixed_directory(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.toml").write_text('name = "alpha"\ndescription = "Alpha"\nsystem = "A"\n', encoding="utf-8")
        (tmp_path / "beta.json").write_text('{"name": "beta", "description": "Beta", "system": "B"}', encoding="utf-8")
        (tmp_path / "gamma.ini").write_text(
            "[agent]\nname = gamma\ndescription = Gamma\n\n[system]\ntext = G\n",
            encoding="utf-8",
        )
        result = load_agents(tmp_path)
        assert set(result.keys()) == {"alpha", "beta", "gamma"}

    def test_result_shape(self, tmp_path: Path) -> None:
        (tmp_path / "x.toml").write_text('name = "x"\ndescription = "Desc X"\nsystem = "sys"\n', encoding="utf-8")
        desc, agent = load_agents(tmp_path)["x"]
        assert desc == "Desc X"
        assert isinstance(agent, Agent)

    def test_tools_resolved(self, tmp_path: Path) -> None:
        (tmp_path / "x.json").write_text(
            '{"name": "x", "description": "", "system": "s", "tools": ["echo"]}',
            encoding="utf-8",
        )
        _, agent = load_agents(tmp_path, _toolbox())["x"]
        assert len(agent.tools) == 1

    def test_unknown_tool_raises(self, tmp_path: Path) -> None:
        (tmp_path / "x.json").write_text(
            '{"name": "x", "description": "", "system": "s", "tools": ["nope"]}',
            encoding="utf-8",
        )
        with pytest.raises(KeyError, match="nope"):
            load_agents(tmp_path, _toolbox())

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert load_agents(tmp_path) == {}


# ---------------------------------------------------------------------------
# make_agent_tools
# ---------------------------------------------------------------------------


class TestMakeAgentTools:
    def _agents(self) -> dict[str, tuple[str, Agent]]:
        transport = StubTransport([make_text_response("done")])
        proto = Agent(system="s", transport=transport)
        return {
            "worker": ("Does the work", proto),
            "reviewer": ("Reviews the work", proto),
        }

    def test_returns_one_tool_per_agent(self) -> None:
        transport = StubTransport([make_text_response("ok")])
        tools = make_agent_tools(self._agents(), transport=transport)
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"worker", "reviewer"}

    def test_tool_description_matches_agent(self) -> None:
        transport = StubTransport([make_text_response("ok")])
        tools = make_agent_tools(self._agents(), transport=transport)
        by_name = {t.name: t for t in tools}
        assert by_name["worker"].description == "Does the work"
        assert by_name["reviewer"].description == "Reviews the work"

    async def test_tool_runs_agent(self) -> None:
        transport = StubTransport([make_text_response("result text")])
        proto = Agent(system="s", transport=DummyCompletionTransport())
        agents = {"bot": ("A bot", proto)}
        tools = make_agent_tools(agents, transport=transport)
        result = await tools[0](task="do something")
        assert result == "result text"

    async def test_tool_streams_events_to_on_event(self) -> None:
        from axio.events import TextDelta

        transport = StubTransport([make_text_response("hello")])
        proto = Agent(system="s", transport=DummyCompletionTransport())
        agents = {"bot": ("A bot", proto)}

        received: list[tuple[str, object]] = []
        tools = make_agent_tools(agents, transport=transport, on_event=lambda n, e: received.append((n, e)))
        await tools[0](task="go")

        assert any(isinstance(e, TextDelta) for _, e in received)
        assert all(n == "bot" for n, _ in received)

    async def test_custom_context_factory(self) -> None:
        calls: list[str] = []

        def factory() -> MemoryContextStore:
            calls.append("created")
            return MemoryContextStore()

        transport = StubTransport([make_text_response("ok")])
        proto = Agent(system="s", transport=DummyCompletionTransport())
        tools = make_agent_tools({"x": ("X", proto)}, transport=transport, context_factory=factory)
        await tools[0](task="t")
        assert calls == ["created"]
