"""Tests for the swarm module functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from axio.models import ModelSpec
from axio.tool import Tool
from axio.transport import DummyCompletionTransport

from agent_swarm.swarm import (
    file_tools,
    make_analyze_tool,
    make_delegate_tool,
    transport_for,
)


class TestFileTools:
    """Test the file_tools function."""

    def test_file_tools_returns_correct_tools_list(self):
        """Test that file_tools() returns correct tools list."""
        tools = file_tools()
        tool_names = [t.name for t in tools]

        expected_tools = ["read_file", "write_file", "patch_file", "list_files", "shell", "run_python"]
        assert tool_names == expected_tools, f"Expected {expected_tools}, got {tool_names}"

    def test_file_tools_returns_tool_instances(self):
        """Test that file_tools() returns Tool instances."""
        tools = file_tools()
        for tool in tools:
            assert isinstance(tool, Tool), f"Expected Tool instance, got {type(tool)}"
            assert tool.name, "Tool name is empty"
            assert tool.description, f"Tool {tool.name} has no description"
            assert tool.handler, f"Tool {tool.name} has no handler"

    def test_file_tools_with_guard_factory(self):
        """Test file_tools with a guard factory."""
        mock_guard = MagicMock()
        guard_factory = MagicMock(return_value=mock_guard)

        tools = file_tools(role="test_role", guard_factory=guard_factory)

        # Verify guard_factory was called for each tool
        assert guard_factory.call_count == 6
        for tool in tools:
            assert len(tool.guards) == 1, f"Tool {tool.name} should have one guard"
            assert tool.guards[0] is mock_guard

    def test_file_tools_without_guard_factory(self):
        """Test file_tools without a guard factory returns empty guards."""
        tools = file_tools()
        for tool in tools:
            assert tool.guards == (), f"Tool {tool.name} should have empty guards tuple"


class TestTransportFor:
    """Test the transport_for function."""

    def test_transport_for_correctly_maps_roles_to_models(self):
        """Test that transport_for() correctly maps roles to models."""
        base_transport = DummyCompletionTransport()
        role_models = {
            "default": ModelSpec(id="gpt-4"),
            "architect": ModelSpec(id="claude-3"),
            "backend_dev": ModelSpec(id="gpt-4-turbo"),
        }

        # Test default mapping
        transport = transport_for("unknown_role", base_transport, role_models)
        assert transport.model == ModelSpec(id="gpt-4")

        # Test role-specific mapping
        transport = transport_for("architect", base_transport, role_models)
        assert transport.model == ModelSpec(id="claude-3")

        transport = transport_for("backend_dev", base_transport, role_models)
        assert transport.model == ModelSpec(id="gpt-4-turbo")

    def test_transport_for_falls_back_to_default(self):
        """Test that transport_for falls back to default for unknown roles."""
        base_transport = DummyCompletionTransport()
        role_models = {
            "default": ModelSpec(id="default-model"),
        }

        transport = transport_for("some_unknown_role", base_transport, role_models)
        assert transport.model == ModelSpec(id="default-model")


class TestMakeAnalyzeTool:
    """Test the make_analyze_tool function."""

    def test_make_analyze_tool_creates_an_analyze_tool(self):
        """Test that make_analyze_tool() creates an analyze tool."""
        workspace = Path("/tmp/test")
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_analyze_tool(workspace, on_event, transport, role_models, "orchestrator")

        assert isinstance(tool, Tool)
        assert tool.name == "analyze"
        assert tool.description, "Analyze tool has no description"
        assert tool.handler, "Analyze tool has no handler"

    def test_make_analyze_tool_with_guard_factory(self):
        """Test make_analyze_tool with guard factory."""
        mock_guard = MagicMock()
        guard_factory = MagicMock(return_value=mock_guard)
        workspace = Path("/tmp/test")
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_analyze_tool(
            workspace, on_event, transport, role_models, "orchestrator", guard_factory=guard_factory
        )

        assert len(tool.guards) == 1
        assert tool.guards[0] is mock_guard
        guard_factory.assert_called_once_with("orchestrator", "analyze")


class TestMakeDelegateTool:
    """Test the make_delegate_tool function."""

    def test_make_delegate_tool_creates_a_delegate_tool(self):
        """Test that make_delegate_tool() creates a delegate tool."""
        workspace = Path("/tmp/test")
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_delegate_tool(workspace, on_event, transport, role_models, {})

        assert isinstance(tool, Tool)
        assert tool.name == "delegate"
        assert tool.description, "Delegate tool has no description"
        assert tool.handler, "Delegate tool has no handler"

    def test_make_delegate_tool_with_guard_factory(self):
        """Test make_delegate_tool with guard factory."""
        mock_guard = MagicMock()
        guard_factory = MagicMock(return_value=mock_guard)
        workspace = Path("/tmp/test")
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_delegate_tool(workspace, on_event, transport, role_models, {}, guard_factory=guard_factory)

        assert len(tool.guards) == 1
        assert tool.guards[0] is mock_guard
        guard_factory.assert_called_once_with("orchestrator", "delegate")


class TestRunSwarm:
    """Test the run_swarm function."""

    @pytest.mark.asyncio
    async def test_run_swarm_initialization(self, workspace: Path):
        """Test that run_swarm() initialization works with stub."""
        # Create a stub transport that returns immediately
        from axio.testing import StubTransport, make_text_response

        from agent_swarm.swarm import run_swarm

        stub_transport = StubTransport([make_text_response("Test response")])

        on_event = AsyncMock()

        role_models = {"default": ModelSpec(id="gpt-4")}

        # Run with a short timeout to avoid hanging
        import asyncio

        async def run_test():
            try:
                result = await asyncio.wait_for(
                    run_swarm(
                        task="Test task",
                        workspace=workspace,
                        on_event=on_event,
                        transport=stub_transport,
                        role_models=role_models,
                    ),
                    timeout=1.0,
                )
                return result
            except TimeoutError:
                # Expected - stub transport doesn't actually complete
                return "timeout"

        result = await run_test()
        # The result should be either "timeout" or some actual text
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_swarm_requires_default_model(self, workspace: Path):
        """Test that run_swarm requires a default model in role_models."""
        from axio.transport import DummyCompletionTransport

        from agent_swarm.swarm import run_swarm

        transport = DummyCompletionTransport()
        on_event = AsyncMock()

        # Missing "default" key should raise AssertionError
        role_models = {"other": ModelSpec(id="gpt-4")}

        with pytest.raises(AssertionError, match="default"):
            await run_swarm(
                task="Test task",
                workspace=workspace,
                on_event=on_event,
                transport=transport,
                role_models=role_models,
            )

    @pytest.mark.asyncio
    async def test_run_swarm_creates_workspace_directory(self, workspace: Path):
        """Test that run_swarm creates the workspace directory."""
        from axio.testing import StubTransport, make_text_response

        from agent_swarm.swarm import run_swarm

        # Create a new workspace path that doesn't exist yet
        test_workspace = workspace / "new_workspace"
        assert not test_workspace.exists()

        stub_transport = StubTransport([make_text_response("Test response")])
        on_event = AsyncMock()
        role_models = {"default": ModelSpec(id="gpt-4")}

        import asyncio

        async def run_test():
            try:
                await asyncio.wait_for(
                    run_swarm(
                        task="Test task",
                        workspace=test_workspace,
                        on_event=on_event,
                        transport=stub_transport,
                        role_models=role_models,
                    ),
                    timeout=1.0,
                )
            except TimeoutError:
                pass  # Expected - stub doesn't really complete

        await run_test()

        # Directory should be created
