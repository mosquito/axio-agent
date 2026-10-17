# Choose a transport in application code

Use `axio.transport.discover_transports()` to select an installed transport by
name. This API belongs to core and works in scripts, services, and custom agent
harnesses. It does not require REPL or TUI.

Install core and the providers your application supports in the same Python
environment. For example:

```bash
uv add axio aiohttp axio-transport-openai axio-transport-anthropic
```

Core supplies discovery and the adapter. Provider packages supply the transport
implementations and register them in the `axio.transport` entry-point group.
Import these helpers from `axio.transport`; they are not top-level `axio` exports.
You can still construct a provider class directly when your application only
needs that provider.

## Discover providers and select one

<!-- name: test_transport_discovery_example -->
```python
from axio.transport import discover_transports

catalog = discover_transports()
for name, entry in catalog.entries.items():
    print(name, entry.display_name)
for name, error in catalog.errors.items():
    print(f"Cannot load {name}: {error}")
```

`catalog.entries` contains successfully loaded adapters. `catalog.errors`
contains failures to load or adapt installed entry points. `catalog.sources`
identifies each entry's distribution and target. Discovery loads plugin code;
it does not open a transport, fetch models, or validate credentials.

Select a provider explicitly with `catalog.require(name)`. The name is the entry
point ID, such as `openai`, `chatgpt`, `anthropic`, or `google`. It is separate
from the model ID. Third-party providers use the same API.

`require()` raises `TransportError` when a provider is absent or failed to load.
An empty environment produces an empty catalog. Duplicate names use the first
declaration, including when it fails; later declarations produce a warning.
Discovery does not select a default provider or switch to another provider
after an error.

## Open a transport and choose a model

`TransportEntry.open()` is an async context manager. For a class entry, it
constructs the transport and calls `fetch_models()` when available. For a
managed plugin, its factory supplies the ready transport and model catalog.
You do not need a second `fetch_models()` call after opening it.

The following functions use only core interfaces and the HTTP client. Provider
names, model IDs, and settings can come from your application's configuration.

<!-- name: test_core_transport_selection -->
```python
from __future__ import annotations

from typing import Any

import aiohttp

from axio import Agent, MemoryContextStore
from axio.transport import discover_transports


async def list_models(provider: str, **settings: Any) -> list[str]:
    entry = discover_transports().require(provider)
    async with aiohttp.ClientSession() as http:
        async with entry.open(session=http, **settings) as transport:
            return list(transport.models.keys())


async def ask(
    provider: str,
    model_id: str,
    prompt: str,
    **settings: Any,
) -> str:
    entry = discover_transports().require(provider)
    async with aiohttp.ClientSession() as http:
        async with entry.open(session=http, **settings) as transport:
            model = transport.models.get(model_id)
            if model is None:
                raise ValueError(f"Model {model_id!r} is unavailable for {provider!r}")
            selected = entry.select_model(transport, model)
            agent = Agent(system="Answer clearly.", transport=selected)
            return await agent.run(prompt, MemoryContextStore())
```

For an API-key provider, first list the models available through that transport:

```python
import asyncio
import os

print(asyncio.run(list_models("openai", api_key=os.environ["OPENAI_API_KEY"])))
```

Set `AXIO_MODEL` to one of the returned IDs, then run a request:

```python
print(
    asyncio.run(
        ask(
            "openai",
            os.environ["AXIO_MODEL"],
            "Explain what an async context manager owns.",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )
)
```

These two calls use the functions above and make real provider requests.
`AXIO_MODEL` is an application setting in this example; core does not read it.
The example rejects a missing model instead of silently selecting another one.
Catalog contents and fallback behavior belong to the provider, so opening a
transport does not guarantee that every listed model is accessible to an account.

Use `entry.select_model(transport, model)` to preserve the provider's cloning
rules. It first uses the plugin's `select_model` hook. Without that hook, it
tries the transport's `select_model`, then `with_model`, and otherwise makes a
shallow copy with the new model. Provider hooks can reset
settings that are incompatible with the selected model. If you already have a
transport without an entry, core also provides `select_transport_model()`.

Keep all requests and model copies inside the `entry.open()` context. Your code
owns the HTTP session; factory-managed account sessions close before that HTTP
session closes. A model copy shares these resources and must not outlive them.
For a long-running service, open them at service startup and close them after
active agent runs finish. Supply a separate context store for each conversation.

<!--
name: test_core_transport_selection
```python
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axio.models import Capability, ModelRegistry, ModelSpec
from axio.testing import StubTransport, make_text_response
from axio.transport import TransportError

opened = []
closed = []


class FixtureProvider:
    @asynccontextmanager
    async def open(self, *, session, **settings):
        assert settings == {"api_key": "fixture-key"}
        assert not session.closed
        models = ModelRegistry(
            ModelSpec(name, frozenset({Capability.text}))
            for name in ("first", "second")
        )
        stub = StubTransport([make_text_response("Fixture answer")])
        transport = SimpleNamespace(
            session=session, models=models, model=models["first"], stream=stub.stream
        )
        opened.append(transport)
        try:
            yield transport
        finally:
            assert not session.closed
            closed.append(transport)


async def check_examples():
    assert await list_models("fixture", api_key="fixture-key") == ["first", "second"]
    assert await ask("fixture", "second", "Hello", api_key="fixture-key") == "Fixture answer"
    assert opened[-1].model.id == "first"
    with pytest.raises(ValueError, match="unavailable"):
        await ask("fixture", "missing", "Hello", api_key="fixture-key")
    with pytest.raises(TransportError, match="Unknown transport"):
        await ask("absent", "second", "Hello")
    with patch("importlib.metadata.entry_points", return_value=[]):
        with pytest.raises(TransportError, match="Unknown transport"):
            await list_models("fixture")
    assert len(opened) == len(closed) == 3
    assert all(item.session.closed for item in closed)


entry_point = SimpleNamespace(name="fixture", load=lambda: FixtureProvider())
with patch("importlib.metadata.entry_points", return_value=[entry_point]):
    asyncio.run(check_examples())
```
-->

## Configure authentication and provider options

Pass constructor or factory settings through `entry.open()`. Their names and
accepted values belong to the provider. For example, an API-key transport can
accept `api_key` and `base_url`. The adapter passes supplied values through;
it does not translate environment-variable names into constructor arguments.

`entry.has_credentials()` checks the provider's local configuration or declared
environment variables. It does not authenticate a request. Its optional mapping
uses provider-defined keys; the default implementation checks the names in
`entry.env_vars`. This check does not configure a later `open()` call.

For account providers, `entry.auth` exposes the optional `TransportAuth` API.
Your application can call `sign_in(credentials, account)` to authorize an account
and `select_account(credentials, account)` to save its selection. The former
returns the account ID. These are explicit user actions; discovery does not
start a browser login. Pass a credential-file `Path` or `None` to these methods.

After signing in with a provider's account command, the same `ask()` function
can open the saved account. For example, after `axio-chatgpt login`:

```python
print(asyncio.run(list_models("chatgpt")))
print(asyncio.run(ask("chatgpt", os.environ["AXIO_MODEL"], "Hello")))
```

Use the common `account` and `credentials` options to select a specific account
registration and credential file for `open()`. Providers without an account
factory reject these options. See {doc}`../api/transport-openai` and
{doc}`google-transport` for provider-specific sign-in and account behavior.

The `supports_temperature`, `supports_max_tokens`, `supports_debug`, and
`supports_thinking` fields describe application controls. `open()` does not
validate options against these flags. Check them before exposing generic
controls, and follow the provider's configuration API for the actual settings.
The `auto_select` flag is also metadata: core does not implement the REPL's
automatic provider-selection policy.

Opening, model loading, and inference can raise provider-specific exceptions.
`TransportError` only covers core selection and adapter errors. Preserve the
original exception for recovery logic; `entry.format_error(error)` provides a
provider-defined display message, or `str(error)` when no formatter is supplied.

## Discover other plugins through core

`axio.plugins` also provides `discover_tools()`, `discover_tool_providers()`,
`discover_guards()`, `discover_selectors()`, and `discover_context_backends()`.
These functions return catalogs with entries, load errors, and sources.
Your application initializes the selected components and owns their lifetimes;
discovery alone does not attach tools or providers to an `Agent`.

TUI settings screens remain a separate UI integration. They are not needed to
discover or use a transport from core. See {doc}`../api/axio` for the catalogs
and lifecycle helpers, and {doc}`writing-transports` to register your own provider.
