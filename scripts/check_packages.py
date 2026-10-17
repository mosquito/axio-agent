"""Check declared package boundaries and optional integration imports."""

from __future__ import annotations

import ast
import re
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_MODULES = {"yaml": "pyyaml", "jwt": "pyjwt", "google": "google-auth"}
APPLICATIONS = {"axio-repl", "axio-tui"}
SHARED = {
    "axio-responses": {"axio-sse"},
    "axio-transport-anthropic": {"axio-sse"},
    "axio-transport-google": {"axio-sse"},
    "axio-transport-openai": {"axio-responses", "axio-sse"},
    "axio-transport-codex": {"axio-responses"},
}
# These application modules are loaded only when their corresponding extra exists.
OPTIONAL_MODULES = {
    "axio-repl": {"configuration.py", "mcp_import.py"},
    "axio-tui": {"configuration.py"},
}


def requirement_name(requirement: str) -> str:
    return re.split(r"[\[<>=!~; @]", requirement, maxsplit=1)[0].lower().replace("_", "-")


@dataclass(frozen=True)
class Import:
    module: str
    line: int
    deferred: bool
    type_only: bool


class Imports(ast.NodeVisitor):
    def __init__(self) -> None:
        self.items: list[Import] = []
        self.deferred = False
        self.type_only = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.items.append(Import(alias.name, node.lineno, self.deferred, self.type_only))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self.items.append(Import(node.module, node.lineno, self.deferred, self.type_only))

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = self.deferred
        self.deferred = True
        self.generic_visit(node)
        self.deferred = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        previous = self.type_only
        if ast.unparse(node.test) in {"TYPE_CHECKING", "typing.TYPE_CHECKING"}:
            self.type_only = True
        for child in node.body:
            self.visit(child)
        self.type_only = previous
        for child in node.orelse:
            self.visit(child)


def allowed(source: str, target: str, *, optional: bool = False) -> bool:
    if source == target:
        return True
    if source in {"axio", "axio-sse"}:
        return False
    if source in APPLICATIONS:
        return target == "axio" or optional
    return target == "axio" or target in SHARED.get(source, set())


def check(root: Path = ROOT) -> list[str]:
    projects: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.glob("axio*/pyproject.toml")):
        project = tomllib.loads(path.read_text())["project"]
        projects[project["name"]] = (path.parent, project)
    modules = {name.replace("-", "_"): name for name in projects}
    errors: list[str] = []
    for name, (directory, project) in projects.items():
        required = {requirement_name(item) for item in project.get("dependencies", [])}
        optional = {
            requirement_name(item)
            for requirements in project.get("optional-dependencies", {}).values()
            for item in requirements
        }
        for dependency in (required | optional) & projects.keys():
            if not allowed(name, dependency, optional=dependency in optional and dependency not in required):
                errors.append(f"{name}: forbidden dependency on {dependency}")
        for group, entries in project.get("entry-points", {}).items():
            for entry_target in entries.values():
                if modules.get(entry_target.split(":")[0].split(".")[0]) != name:
                    errors.append(f"{name}: {group} entry point targets another distribution: {entry_target}")
        for path in sorted((directory / "src").rglob("*.py")):
            visitor = Imports()
            visitor.visit(ast.parse(path.read_text(), filename=str(path)))
            for item in visitor.items:
                target = modules.get(item.module.split(".")[0])
                location = f"{path.relative_to(root)}:{item.line}"
                if target is None:
                    module = item.module.split(".")[0]
                    if module in sys.stdlib_module_names:
                        continue
                    dependency = EXTERNAL_MODULES.get(module, module.replace("_", "-"))
                    # google-auth explicitly installs its selected HTTP adapter.
                    supplied = dependency == "urllib3" and any(
                        item.startswith("google-auth[urllib3]") for item in project.get("dependencies", [])
                    )
                    if dependency not in required | optional and not supplied:
                        errors.append(f"{location}: undeclared external import of {dependency}")
                    continue
                if target == name:
                    continue
                if not allowed(name, target, optional=target in optional):
                    errors.append(f"{location}: forbidden import of {target}")
                elif target not in required | optional:
                    errors.append(f"{location}: undeclared import of {target}")
                elif (
                    target not in required
                    and not item.type_only
                    and not item.deferred
                    and path.name not in OPTIONAL_MODULES.get(name, set())
                ):
                    errors.append(f"{location}: eager import of optional {target}")
    return errors


def self_test() -> None:
    assert not allowed("axio", "axio-tools-local", optional=True)
    assert not allowed("axio-transport-google", "axio-transport-anthropic", optional=True)
    assert not allowed("axio-tui-guards", "axio-tui", optional=True)
    assert not allowed("axio-repl", "axio-tools-local")
    assert allowed("axio-repl", "axio-tools-local", optional=True)
    assert allowed("axio-transport-openai", "axio-responses")
    assert allowed("axio-responses", "axio-sse")
    visitor = Imports()
    visitor.visit(ast.parse("from axio import Agent\nif TYPE_CHECKING:\n import axio_tools_local\n"))
    assert not visitor.items[0].type_only and visitor.items[1].type_only
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in ("axio", "axio-tools-local"):
            directory = root / name
            directory.mkdir()
            (directory / "pyproject.toml").write_text(f'[project]\nname="{name}"\n')
        source = root / "axio/src/axio"
        source.mkdir(parents=True)
        (source / "bad.py").write_text("from axio_tools_local import read_file\n")
        assert any("forbidden import" in issue for issue in check(root))
        (source / "bad.py").unlink()
        (root / "axio/pyproject.toml").write_text('[project]\nname="axio"\ndependencies=["axio-tools-local"]\n')
        assert any("forbidden dependency" in issue for issue in check(root))


if __name__ == "__main__":
    self_test()
    problems = check()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print("Package declarations, entry points and source import boundaries passed.")
