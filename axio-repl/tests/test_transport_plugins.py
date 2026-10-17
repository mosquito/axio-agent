"""Installed entry points work without importing optional transport packages."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def plugin_install(tmp_path: Path) -> Path:
    """Install the independent fixture's files and metadata without a build backend."""
    source = Path(__file__).parent / "fixtures" / "transport-plugin"
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    target = tmp_path / "site-packages"
    shutil.copytree(source / "src", target)
    metadata = target / "axio_fixture_transport-0.0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {project['version']}\n"
    )
    (metadata / "entry_points.txt").write_text(
        "\n".join(
            f"[{group}]\n" + "".join(f"{name} = {value}\n" for name, value in entries.items())
            for group, entries in project["entry-points"].items()
        )
    )
    return target


_BOOTSTRAP = """
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
# Import failures simulate an environment with no installed built-in providers.
# Discovery itself uses real importlib metadata, including the fixture package.
class NoBuiltins(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('axio_transport_'):
            raise ModuleNotFoundError('optional provider is not installed')
sys.meta_path.insert(0, NoBuiltins())
sys.path.insert(0, sys.argv[1])
"""


@pytest.mark.parametrize("explicit", [False, True])
def test_external_package_runs_repl(plugin_install: Path, tmp_path: Path, explicit: bool) -> None:
    script = (
        _BOOTSTRAP
        + """
import asyncio
import axio_repl
from fixture_transport import plugin, instructions
from axio import MemoryContextStore
from axio.testing import StubTransport, make_tool_use_response, make_text_response
async def exercise(agent, context, prompt):
    assert any(i.id == 'fixture:plain' for i in agent._catalog())
    assert 'fixture_answer' not in {t.name for t in agent.get_tools()}
    agent.transport = StubTransport([
        make_tool_use_response('load_instruction', tool_input={'id': 'fixture:plain'}),
        make_tool_use_response('load_instruction', tool_input={'id': 'fixture:tools'}),
        make_tool_use_response('fixture_answer', tool_input={'message': 'fixture'}),
        make_text_response('fixture reply from fixture-second'),
    ])
    print(await agent.run(prompt, context))
    assert 'Full fixture guide' in agent._effective_system()
    assert any('fixture tool answer' in str(m.content) for m in await context.get_history())
axio_repl.run_prompt = exercise
sys.argv = ['axio-repl', '--isolated', '--model', 'fixture-second', *sys.argv[2:], 'hello']
axio_repl.setup_history = lambda: None
asyncio.run(axio_repl.main())
assert len(plugin.opened) == 1 and plugin.opened[0].closed
assert plugin.opened[0].session.closed
assert instructions.opened and instructions.opened[0].closed
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(script),
            str(plugin_install),
            *(["--transport", "fixture-plugin"] if explicit else []),
        ],
        cwd=tmp_path,
        env={**os.environ, "AXIO_FIXTURE_API_KEY": "fixture"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "fixture reply from fixture-second" in result.stdout


def test_external_package_runs_tui(plugin_install: Path, tmp_path: Path) -> None:
    script = (
        _BOOTSTRAP
        + """
import asyncio
from axio import MemoryContextStore
from axio_tui.options import StartupOptions
from axio_tools_local.paths import LocalPaths as Workspace
import axio_tui.app as ui
from fixture_transport import plugin, instructions
from axio.testing import StubTransport, make_tool_use_response, make_text_response
from axio.permission_policy import Approval
from unittest.mock import AsyncMock
ui.DB_PATH = Path.cwd() / 'tui.db'
ui.discover_tools_plugins = lambda: {}
ui.discover_guards = lambda: {}
async def run():
    app = ui.AgentApp(paths=Workspace.discover(Path.cwd()), options=StartupOptions(isolated=True))
    async with app.run_test():
        await asyncio.wait_for(app.ready.wait(), 10)
        assert app._agent is not None
        assert app._transports.available == ['fixture-plugin']
        assert app._transports.display_name('fixture-plugin') == 'Independent fixture'
        await app._reconfigure_transport('fixture-plugin')
        assert app._role_bindings[ui.ModelRole.CHAT].transport == 'fixture-plugin'
        assert await app._agent.run('hello', MemoryContextStore()) == 'fixture reply from fixture-first'
        assert any(i.id == 'fixture:plain' for i in app._agent._catalog())
        app._permissions.prompt = AsyncMock(return_value=Approval('allow'))
        app._agent.transport = StubTransport([
            make_tool_use_response('load_instruction', tool_input={'id': 'fixture:plain'}),
            make_tool_use_response('load_instruction', tool_input={'id': 'fixture:tools'}),
            make_tool_use_response('fixture_answer', tool_input={'message': 'fixture'}),
            make_text_response('loaded'),
        ])
        context = MemoryContextStore()
        assert await app._agent.run('load', context) == 'loaded'
        assert any('fixture tool answer' in str(m.content) for m in await context.get_history())
        selected = app._transports.make_transport('fixture-plugin', plugin.opened[0].models['fixture-second'])
        assert selected.session is plugin.opened[0].session
    assert plugin.opened[0].closed and plugin.opened[0].session.closed
    assert instructions.opened and instructions.opened[0].closed
asyncio.run(run())
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(plugin_install)],
        cwd=tmp_path,
        env={**os.environ, "AXIO_FIXTURE_API_KEY": "fixture", "XDG_CONFIG_HOME": str(tmp_path / "config")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_repl_import_and_help_without_any_transport(tmp_path: Path) -> None:
    script = (
        _BOOTSTRAP
        + """
import axio_repl
sys.argv = ['axio-repl', '--help']
axio_repl.main_sync()
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--transport" in result.stdout


@pytest.mark.asyncio
async def test_broken_plugin_diagnostic_and_conflict_policy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from types import SimpleNamespace

    import axio_repl

    def broken() -> object:
        raise ImportError("missing fixture dependency")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: [SimpleNamespace(name="broken", load=broken)])
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--isolated", "--transport", "broken", "hello"])
    with pytest.raises(SystemExit) as error:
        await axio_repl.main()
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "failed to load" in message and "missing fixture dependency" in message
    assert "Unknown transport" not in message


@pytest.mark.asyncio
async def test_third_party_auth_and_factory_receive_account_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from axio.models import ModelRegistry, ModelSpec
    from axio.transport import TransportCatalog, TransportEntry

    import axio_repl

    calls: list[tuple[object, ...]] = []
    model = ModelSpec("fixture")
    auth = SimpleNamespace(
        has_accounts=AsyncMock(return_value=True),
        prompt_login_if_needed=AsyncMock(),
        describe_accounts=AsyncMock(return_value="fixture accounts"),
        account_id=lambda transport: transport.account,
        sign_in=AsyncMock(return_value="second"),
        select_account=AsyncMock(),
        logout=AsyncMock(),
    )

    @asynccontextmanager
    async def open_transport(**settings: Any) -> AsyncIterator[Any]:
        calls.append(("open", settings["account"], settings["credentials"]))
        try:
            yield SimpleNamespace(
                name="Fixture", model=model, models=ModelRegistry([model]), account=settings["account"]
            )
        finally:
            calls.append(("close", settings["account"]))

    plugin = SimpleNamespace(open=open_transport, auth=auth, supports_debug=False)
    entry = TransportEntry.from_target("external-auth", plugin)
    monkeypatch.setattr(axio_repl, "discover_transports", lambda: TransportCatalog({entry.name: entry}))
    monkeypatch.setattr(axio_repl, "setup_history", lambda: None)
    inputs = iter(["/account", "/login", "/account third", "/logout", "/debug on", "/quit"])
    monkeypatch.setattr(axio_repl, "_read_input", lambda: next(inputs))
    monkeypatch.chdir(tmp_path)
    credentials = tmp_path / "auth"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axio-repl",
            "--isolated",
            "--transport",
            entry.name,
            "--account",
            "first",
            "--credentials",
            str(credentials),
        ],
    )
    await axio_repl.main()
    auth.prompt_login_if_needed.assert_awaited_once_with(credentials, "first")
    auth.sign_in.assert_awaited_once_with(credentials, "first")
    assert [call.args[1] for call in auth.select_account.await_args_list] == ["second", "third"]
    assert auth.logout.await_args.args[0].account == "third"
    assert calls == [
        ("open", "first", str(credentials)),
        ("open", "second", str(credentials)),
        ("open", "third", str(credentials)),
        ("close", "third"),
        ("close", "second"),
        ("close", "first"),
    ]
    assert "does not support debug" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cancelled_account_command_leaves_active_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from axio.models import ModelRegistry, ModelSpec
    from axio.transport import TransportCatalog, TransportEntry

    import axio_repl

    model = ModelSpec("fixture")
    closed: list[bool] = []
    auth = SimpleNamespace(
        prompt_login_if_needed=AsyncMock(),
        account_id=lambda _: "original",
        sign_in=AsyncMock(side_effect=asyncio.CancelledError),
        select_account=AsyncMock(),
    )

    @asynccontextmanager
    async def open_transport(**settings: Any) -> AsyncIterator[Any]:
        try:
            yield SimpleNamespace(name="Fixture", model=model, models=ModelRegistry([model]))
        finally:
            closed.append(True)

    entry = TransportEntry.from_target("external", SimpleNamespace(open=open_transport, auth=auth))
    monkeypatch.setattr(axio_repl, "discover_transports", lambda: TransportCatalog({entry.name: entry}))
    monkeypatch.setattr(axio_repl, "setup_history", lambda: None)
    inputs = iter(["/login", "/quit"])
    monkeypatch.setattr(axio_repl, "_read_input", lambda: next(inputs))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--isolated", "--transport", entry.name])
    await axio_repl.main()
    auth.select_account.assert_not_awaited()
    assert closed == [True]
    assert "Account command interrupted" in capsys.readouterr().out
