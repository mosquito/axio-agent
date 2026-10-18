# Packages

Install only the libraries and integrations your application needs. All packages
require Python 3.12 or newer. The pages below are the package READMEs, shared with
the repository so installation and usage instructions have one source.

For a ready-made assistant, start with `axio-repl` or `axio-tui`. For your own
application, start with `axio` and a transport. Core and `axio-sse` have no required
third-party dependencies. Tools, persistent history, and audio I/O are separate
packages.

```{toctree}
:maxdepth: 1
:caption: Core and context

axio
axio-context-sqlite
```

```{toctree}
:maxdepth: 1
:caption: Transports

axio-transport-openai
axio-transport-anthropic
axio-transport-google
axio-transport-codex
```

```{toctree}
:maxdepth: 1
:caption: Tools

axio-tools-local
axio-tools-mcp
axio-tools-docker
```

```{toctree}
:maxdepth: 1
:caption: Applications and audio

axio-repl
axio-tui
axio-tui-guards
axio-audio
```

```{toctree}
:maxdepth: 1
:caption: Shared protocol libraries

axio-sse
axio-responses
```

See {doc}`../api/index` for source API documentation,
{doc}`../guides/choosing-transports` to select a transport in application code,
and {doc}`../guides/plugin-discovery` for entry-point groups and plugin lifecycles.
