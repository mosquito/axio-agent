PACKAGES := axio axio-tools-docker axio-tools-local axio-tools-mcp \
            axio-transport-codex axio-transport-nebius axio-transport-openai \
            axio-tui axio-tui-guards axio-tui-rag

.PHONY: all lint pytest $(PACKAGES)

all: $(PACKAGES)

$(PACKAGES):
	@echo "==> $@"
	@uv run --directory $@ ruff check
	@uv run --directory $@ ruff format --check
	@uv run --directory $@ mypy .
	@uv run --directory $@ pytest -q

lint pytest:
	@$(MAKE) -j$(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu) $(PACKAGES)
