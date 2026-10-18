# axio-docs

Documentation for the Axio agent framework.

Part of the [axio-agent](https://github.com/mosquito/axio-agent) ecosystem.

## Building the docs

```bash
uv sync --group docs
uv run make html
```

The built HTML will be in `_build/html/`.

## Package documentation

`packages/index.md` lists every installable package. Each package page is a
relative symlink to its `README.md`, for example
`packages/axio.md -> ../../axio/README.md`. Edit the original README to update
both the repository and the documentation site.

When adding a package, add its README, create the corresponding symlink, and
list it in the package index. Use absolute repository URLs for README links
to source files so they also work from the documentation site.

From the repository root, run `make docs-html` for a Sphinx build with warnings
treated as errors and `make test-docs` for executable examples, including those
in the linked READMEs.

## License

MIT
