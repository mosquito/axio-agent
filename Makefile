PACKAGES := axio axio-audio axio-context-sqlite axio-repl axio-responses axio-sse \
            axio-tools-docker axio-tools-local axio-tools-mcp \
            axio-transport-anthropic axio-transport-codex \
            axio-transport-google axio-transport-openai \
            axio-tui axio-tui-guards examples/gas_town examples/agent_swarm \
            examples/realtime_smoke examples/realtime_chat

.PHONY: $(PACKAGES) all pytest linter typing test tests test-docs test-tutorial docs-html

all: check-packages linter typing pytest test-docs test-tutorial docs-html

linter:
	@uv run --directory axio ruff check ../scripts
	@uv run --directory axio ruff format --check ../scripts
	@for pkg in $(PACKAGES); do uv run --directory $$pkg ruff check . && uv run --directory $$pkg ruff format --check . || exit 1; done
	@uv run --directory docs ruff check ../examples/tutorial
	@uv run --directory docs ruff format --check ../examples/tutorial

typing pytest: $(PACKAGES)

$(PACKAGES):
	@uv run --directory $@ mypy .
	@uv run --directory $@ pytest -q

test-docs:
	@uv run --directory docs pytest -q .

test-tutorial:
	@uv run --directory docs pytest -q ../examples/tutorial

docs-html:
	@$(MAKE) -C docs check-html

test: pytest
tests: pytest

.PHONY: check-packages test-packaging

check-packages:
	@uv run --directory axio mypy ../scripts
	@uv run --directory axio python ../scripts/check_packages.py

test-packaging:
	@uv run --directory axio python ../scripts/test_packaging.py $(CASES)
