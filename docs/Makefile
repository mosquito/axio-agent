.PHONY: html clean cf-inspect cf-setup deploy svg record

SPHINXBUILD ?= uv run --group docs sphinx-build
SOURCEDIR = .
BUILDDIR = _build
CAST = ../examples/stream_tool_args.v2.cast
CAST_CMD = uv run --extra examples python ../examples/stream_tool_args.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf $(BUILDDIR)

cf-inspect:
	bash scripts/cf-inspect.sh

cf-setup:
	bash scripts/cf-setup.sh

record: $(CAST)

$(CAST):
	asciinema rec --cols 100 --rows 30 -f asciicast-v2 --command "echo '$$$ $(CAST_CMD)'; $(CAST_CMD); sleep 5" $@

svg: _static/stream_tool_args.svg

_static/stream_tool_args.svg: $(CAST)
	npx svg-term-cli --in $< --out $@ --window --height $$(( $$(asciinema convert -f txt $< - | wc -l) + 2 ))

deploy: html
	bash scripts/deploy.sh
