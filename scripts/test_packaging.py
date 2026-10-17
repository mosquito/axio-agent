"""Build local wheels and exercise independent installations outside the workspace."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "axio",
    "axio-sse",
    "axio-responses",
    "axio-context-sqlite",
    "axio-tools-local",
    "axio-tools-mcp",
    "axio-transport-google",
    "axio-transport-anthropic",
    "axio-transport-openai",
    "axio-transport-codex",
    "axio-repl",
    "axio-tui",
    "axio-tui-guards",
)
CASES = {
    "core": ["axio"],
    "local": ["axio-tools-local"],
    "mcp": ["axio-tools-mcp"],
    "google": ["axio-transport-google"],
    "anthropic": ["axio-transport-anthropic[vertexai]"],
    "guards": ["axio-tui-guards"],
    "repl": ["axio-repl"],
    "repl-local": ["axio-repl[local]"],
    "repl-mcp": ["axio-repl[mcp]"],
    "repl-transport": ["axio-repl", "fixture-transport"],
    "repl-provider": ["axio-repl", "fixture-provider"],
    "tui": ["axio-tui"],
    "tui-local": ["axio-tui[local]"],
    "tui-mcp": ["axio-tui[mcp]"],
    "tui-history": ["axio-tui[history]"],
    "tui-guards": ["axio-tui[guards]"],
    "tui-transport": ["axio-tui", "fixture-transport"],
    "tui-provider": ["axio-tui", "fixture-provider"],
}
SMOKE = r"""
import asyncio
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

case = sys.argv[1]
installed = {dist.metadata['Name'].lower().replace('_', '-') for dist in importlib.metadata.distributions()}
for package in ('axio-tools-local', 'axio-tools-mcp', 'axio-context-sqlite', 'axio-tui-guards'):
    requested = {
        'axio-tools-local': 'local', 'axio-tools-mcp': 'mcp',
        'axio-context-sqlite': 'history', 'axio-tui-guards': 'guards',
    }[package]
    assert (package in installed) == (requested in case.split('-')), (case, package, installed)
if case in ('core', 'local', 'mcp', 'google', 'anthropic', 'guards'):
    assert 'axio-tui' not in installed and 'textual' not in installed
if case == 'google':
    from axio_transport_google import GoogleTransport, VertexAITransport
    from axio_transport_google.antigravity import AntigravityTransport
    from axio.models import ModelSpec
    from axio.exceptions import StreamError
    assert 'axio-transport-anthropic' not in installed
    GoogleTransport()
    VertexAITransport()
    old = VertexAITransport(model=ModelSpec('anthropic/claude-sonnet-4-6'))
    try:
        old.stream([], [], '')
    except StreamError as error:
        assert 'anthropic-vertex' in str(error)
    else:
        raise AssertionError('legacy request was not rejected')
if case == 'anthropic':
    from axio_transport_anthropic import VertexAITransport
    assert 'axio-transport-google' not in installed
    assert '/publishers/anthropic/' in VertexAITransport(project='fixture')._build_url()

async def providers():
    from axio.plugins import discover_tool_providers
    catalog = discover_tool_providers()
    assert not catalog.errors, catalog.errors
    for provider in catalog.entries.values():
        try:
            await provider.init()
            provider.get_instructions()
        finally:
            await provider.close()

async def guards():
    from axio import Agent, MemoryContextStore, Tool
    from axio.exceptions import GuardError
    from axio.testing import StubTransport, make_text_response
    from axio_tui_guards import PathGuard, LLMGuard
    async def allow(message):
        return 'y'
    async def deny(message):
        return 'n'
    async def read_file(filename: str) -> str:
        return 'data'
    tool = Tool('read_file', read_file)
    assert await PathGuard(allow)(tool, filename='/example/a') == {'filename': '/example/a'}
    agent = Agent(system='', transport=StubTransport([make_text_response('no verdict')]))
    try:
        await LLMGuard(agent, MemoryContextStore(), deny)(tool, filename='/example/a')
    except GuardError:
        pass
    else:
        raise AssertionError('missing verdict must not allow execution')

async def tui():
    import axio_tui.app as ui
    from axio.context import MemoryContextStore
    from axio_tui.options import StartupOptions
    ui.DB_PATH = Path.cwd() / 'tui.db'
    app = ui.AgentApp(options=StartupOptions(isolated=True))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app.ready.wait(), 10)
        if 'history' in case:
            assert app._context_backend is not None
        else:
            assert isinstance(app._chat_context, MemoryContextStore)
        if 'transport' in case:
            assert app._agent is not None
            assert await app._agent.run('hello', MemoryContextStore()) == 'fixture reply from fixture-first'
        else:
            assert app._agent is None
        if 'provider' in case:
            assert any(p.get_instructions() for p in app._providers.values())
        if 'local' in case:
            assert 'read_file' in {t.name for t in app._all_tools}
        await app._show_skills()
        await pilot.pause()
        app.pop_screen()
        await pilot.pause()
        await app._show_mcp()
        await pilot.pause()
        if 'mcp' in case:
            await pilot.click('#mcp-manage')
            await pilot.pause()
            from axio_tui.mcp_manage import MCPManagementScreen
            assert isinstance(app.screen, MCPManagementScreen)
            app.pop_screen()
            await pilot.pause()
        else:
            assert not app.screen.query('#mcp-manage')
            assert 'axio_tui.mcp_manage' not in sys.modules
        app.pop_screen()
        await pilot.pause()
        await app._new_session()
        await app._fork_conversation()
        await app._replace_context([])
        if 'history' not in case:
            await app._show_session_select()
    if 'provider' in case:
        from fixture_transport import instructions
        assert instructions.opened and all(p.closed for p in instructions.opened)
    if 'transport' in case:
        from fixture_transport import plugin
        assert plugin.opened and all(p.closed and p.session.closed for p in plugin.opened)

if case.startswith('repl'):
    import axio_repl
    os.environ['AXIO_FIXTURE_API_KEY'] = 'fixture'
    sys.argv = ['axio-repl', '--isolated']
    commands = ['/skills', '/skills reload', '/mcp', 'hello', '/quit']
    axio_repl._read_input = iter(commands).__next__
    axio_repl.setup_history = lambda: None
    axio_repl.main_sync()
    if 'provider' in case:
        from fixture_transport import instructions
        assert instructions.opened and all(p.closed for p in instructions.opened)
    if 'transport' in case:
        from fixture_transport import plugin
        assert plugin.opened and all(p.closed and p.session.closed for p in plugin.opened)
elif case.startswith('tui'):
    os.environ['AXIO_FIXTURE_API_KEY'] = 'fixture'
    asyncio.run(tui())
elif case in ('local', 'mcp'):
    asyncio.run(providers())
elif case == 'guards':
    asyncio.run(guards())
else:
    import axio
print(case + ': passed')
"""


def run(arguments: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, env=env, check=True, timeout=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", choices=sorted(CASES))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="axio-packaging-") as temporary:
        work = Path(temporary)
        wheels = work / "wheels"
        for package in PACKAGES:
            run(["uv", "build", "--package", package, "--wheel", "--out-dir", str(wheels)], cwd=ROOT)
        for kind in ("transport", "provider"):
            source = work / ("fixture-" + kind)
            source.mkdir()
            shutil.copytree(ROOT / "axio-repl/tests/fixtures/transport-plugin/src", source / "src")
            group = "axio.transport" if kind == "transport" else "axio.tools.providers"
            target = (
                "fixture_transport.plugin:Plugin" if kind == "transport" else "fixture_transport.instructions:Provider"
            )
            original = tomllib.loads((ROOT / "axio-repl/tests/fixtures/transport-plugin/pyproject.toml").read_text())
            if kind == "transport":
                target = next(iter(original["project"]["entry-points"]["axio.transport"].values()))
            (source / "pyproject.toml").write_text(
                f'[project]\nname="fixture-{kind}"\nversion="0.0.1"\ndependencies=["axio"]\n'
                f'[project.entry-points."{group}"]\nfixture="{target}"\n'
                '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
                '[tool.hatch.build.targets.wheel]\npackages=["src/fixture_transport"]\n'
            )
            run(["uv", "build", str(source), "--wheel", "--out-dir", str(wheels)], cwd=work)
        constraints = work / "constraints.txt"
        constraints.write_text(
            "".join(
                f"{wheel.name.split('-')[0].replace('_', '-')} @ {wheel.as_uri()}\n"
                for wheel in sorted(wheels.glob("*.whl"))
            )
        )
        for case in args.cases or CASES:
            environment = work / case
            run(["uv", "venv", "--python", sys.executable, str(environment)], cwd=work)
            python = environment / "bin/python"
            run(
                ["uv", "pip", "install", "--python", str(python), "--constraint", str(constraints), *CASES[case]],
                cwd=work,
            )
            directory = work / (case + "-cwd")
            directory.mkdir()
            env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "VIRTUAL_ENV"}}
            env["HOME"] = str(directory)
            env["XDG_CONFIG_HOME"] = str(directory / "config")
            run([str(python), "-I", "-c", SMOKE, case], cwd=directory, env=env)


if __name__ == "__main__":
    main()
