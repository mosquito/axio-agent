# Contributing to Axio

Guidance for developers contributing to the **axio-agent** monorepo.

---

## Development Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) — workspace management, virtual environments, and fast package resolution

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Clone and Setup

```bash
git clone https://github.com/mosquito/axio-agent.git
cd axio-agent
uv sync --all-packages
```

`uv sync --all-packages` installs every workspace member and all development dependencies into a single shared virtual environment at `.venv/`. All local packages resolve directly to their workspace sources — no `pip install -e` or `PYTHONPATH` manipulation required.

---

## Workspace Structure

This repository is a **uv workspace** monorepo defined in the root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["axio", "axio-tui", "axio-transport-anthropic", ...]

[tool.uv.sources]
axio                     = { workspace = true }
axio-transport-anthropic = { workspace = true }
# ...
```

Each subdirectory in `members` is an independent Python package with its own `pyproject.toml`, `src/` layout, and `tests/`. They share a single `uv.lock` and `.venv`. Any change made in `axio/src/` is immediately reflected in every dependent package in the workspace.

To run a command inside a specific package context:

```bash
uv run --directory axio-transport-anthropic pytest
```

---

## Makefile Targets

All routine development tasks run through `make`. Do not invoke `uv run pytest` or `ruff` directly from the repo root.

```bash
make              # lint + type-check + tests for all packages + doc tests
make linter       # ruff check + ruff format --check on all packages
make typing       # mypy --strict on all packages
make pytest       # pytest on all packages
make test-docs    # markdown-pytest on docs/
make docs-html    # build Sphinx documentation
```

### Running checks on a single package

```bash
make PACKAGES=axio-transport-anthropic
```

### Running a specific test file

```bash
uv run --directory axio pytest tests/test_agent_run.py -v
```

### Running doc tests for a single guide

```bash
uv run --directory docs pytest -v guides/best-practices.md
```

---

## Code Style & Standards

- **Formatter & Linter**: [ruff](https://docs.astral.sh/ruff/), line length `119`, target `py312`.
- **Rules**: `E`, `F`, `I`, `UP`.
- **Type Checker**: [mypy](https://mypy.readthedocs.io/) in strict mode (`--strict`). All new code must pass `mypy --strict` with zero errors.
- Always include `from __future__ import annotations` at the top of every module.
- Prefer `dataclass(frozen=True, slots=True)` for value types.
- `Tool` is generic: write `list[Tool[Any]]`, never bare `list[Tool]`.

Always verify `make linter` and `make typing` pass before submitting changes.

---

## Documentation Tests

Documentation in `docs/` and package READMEs is tested with [markdown-pytest](https://github.com/mosquito/markdown-pytest). Code blocks in Markdown are annotated with HTML comments:

```markdown
<!-- name: test_my_example -->
```python
import asyncio
# ...
```
```

Hidden setup blocks (stubs required for execution that should not appear in rendered docs):

```markdown
<!--
name: test_my_example
```python
# hidden setup code - not visible in rendered docs
```
-->
```

Doc tests run offline without network access or live API keys. Use `axio.testing.StubTransport` for offline testing.

---

## Adding a New Package

1. Create directory `axio-<name>/` with `src/axio_<name>/` layout, including `py.typed`, and a `pyproject.toml` matching the style of existing packages (hatchling backend, dev dependencies).
2. Add the package to `[tool.uv.workspace] members` and `[tool.uv.sources]` in the root `pyproject.toml`.
3. Add the package to `PACKAGES` in `Makefile`.
4. Register the package in the docs build: add it to `dependencies` in `docs/pyproject.toml`, add `docs/api/<name>.md`, and list it in `docs/api/index.md`.
5. Run `uv sync --all-packages` to update `uv.lock`.

---

## Repository Layout

- **`axio/`** — Core library: `Agent`, `Tool`, `CompletionTransport`, `ContextStore`, `PermissionGuard`, events, blocks.
- **`axio-sse/`** — Sans-I/O `text/event-stream` decoder and payload reader. Zero external dependencies.
- **`axio-responses/`** — OpenAI Responses API request builders and SSE stream reader.
- **`axio-transport-*/`** — LLM transports:
  - `openai`: `/v1/responses`, `/v1/chat/completions`, ChatGPT OAuth, Nebius, OpenRouter, STT, TTS, Realtime.
  - `anthropic`: Anthropic Claude (Direct API & Google Cloud Vertex AI) with prompt caching.
  - `google`: Gemini Developer API & Vertex AI, image/video generation, Gemini Live realtime.
  - `codex`: ChatGPT desktop OAuth over Responses API.
- **`axio-tools-*/`** — Tool providers:
  - `local`: filesystem, shell, and Python execution.
  - `mcp`: Model Context Protocol bridge (stdio & HTTP).
  - `docker`: isolated Docker container sandbox.
- **`axio-context-sqlite/`** — Persistent SQLite context store with transparent compression.
- **`axio-tui/`** & **`axio-tui-guards/`** — Fullscreen Textual terminal interface and guard plugins.
- **`axio-repl/`** — Interactive CLI coding assistant that verifies its work.
- **`axio-audio/`** — Microphone, speaker, and duplex audio helpers for realtime voice sessions.
- **`docs/`** — Sphinx documentation and executable guides.
- **`examples/`** — Runnable example projects (`agent_swarm`, `gas_town`, `realtime_chat`, `realtime_smoke`).
