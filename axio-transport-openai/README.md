# axio-transport-openai

[![PyPI](https://img.shields.io/pypi/v/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

OpenAI-compatible streaming transport for [axio](https://github.com/mosquito/axio-agent).

Works with any API that speaks the OpenAI chat completions format - OpenAI itself, local models via Ollama/vLLM/LM Studio, Nebius AI Studio, OpenRouter, and any other compatible provider.

## Features

- **Full SSE streaming** - parses `data:` chunks incrementally; no waiting for full responses
- **Automatic retry** - configurable backoff on 429 and 5xx responses; honours `Retry-After` header
- **Tool calling** - streams tool-use JSON fragments as `ToolInputDelta` events; parallel tool calls supported
- **Reasoning support** - `<think>...</think>` blocks emitted as `ReasoningDelta` events
- **Embeddings** - `embed()` method for models that support `/v1/embeddings`
- **Sub-transports** - `NebiusTransport`, `OpenRouterTransport`, and `OpenAICompatibleTransport` for common providers
- **aiohttp-based** - zero blocking I/O
- **Optional TUI settings screen** - install with `[tui]` extra for a Textual configuration UI

## Installation

```bash
pip install axio-transport-openai
```

With TUI settings screens:

```bash
pip install "axio-transport-openai[tui]"
```

## Usage

```python
import asyncio
import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import TextDelta
from axio_transport_openai import OpenAITransport, OPENAI_MODELS


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        transport = OpenAITransport(
            api_key="sk-...",
            model=OPENAI_MODELS["gpt-4.1-mini"],
            session=session,
        )
        agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)
        ctx = MemoryContextStore()
        async for event in agent.run_stream("What is 2 + 2?", ctx):
            if isinstance(event, TextDelta):
                print(event.delta, end="", flush=True)
        print()


asyncio.run(main())
```

The `session` parameter is **required** for streaming. Create an `aiohttp.ClientSession` in an async context and pass it to the transport.

### Sign in with ChatGPT

`axio_transport_openai.oauth` provides browser registration and returning sign-in.
It uses `aiohttp` for HTTP and PyJWT for ID-token verification. It does not require
Textual. `ChatGPTTransport` uses this account session for subscription inference.

```python
import aiohttp
from axio_transport_openai.oauth import ChatGPTAccount, ChatGPTOAuth


async def sign_in(host_id: str, account: ChatGPTAccount | None = None) -> ChatGPTAccount:
    async with aiohttp.ClientSession() as session:
        oauth = ChatGPTOAuth(session=session, host_id=host_id)
        return await oauth.sign_in(account)
```

Choose and persist a unique host ID before calling this function, for example a
`urn:uuid:` value. Reuse it for every sign-in on that host. Pass a saved account to
reauthorize its issued client ID; omit it to register another account. The caller
owns credential storage. Keep tokens in protected storage and never log them.

`sign_in()` returns an immutable `ChatGPTAccount` after verifying the ID token's
signature, issuer, audience, expiry, nonce, and subject. It does not modify the
selected account. The returned `tokens.scopes` contains the granted permissions;
`tokens.plan_enabled` reports permission to use the ChatGPT plan. Identity-only
sign-in succeeds with this property set to `False`.

The callback binds to `127.0.0.1` on an available port and uses `/auth/callback`.
Pass `port=` when a fixed port is required. The default timeout is 300 seconds and
covers the complete flow. Cancellation or timeout closes the listener. A declined
consent raises `OAuthDenied`; invalid identity or callback data raises
`OAuthValidationError`. Other OAuth failures raise `OAuthError` with safe
`status` and `code` fields. An expired authorization code triggers one fresh
browser attempt with the issued client ID.

After validation, the browser shows a local status page with the Axio owl. It
supports light and dark themes and attempts to close after five seconds. The
close button also attempts immediate closure. If the browser blocks closure, the
page asks the user to close the tab manually. The page loads no remote assets and
includes no account details or credentials.

See the [OpenAI registration and sign-in specification](https://developers.openai.com/siwc/token-sharing-open-source/sign-in).

### ChatGPT subscription transport

`ChatGPTTransport` implements completion through the public `/v1/responses` endpoint.
It uses a selected `ChatGPTSession` and the caller's `aiohttp.ClientSession`; it does
not read `OPENAI_API_KEY` or reuse Codex backend credentials.

<!-- name: test_readme_chatgpt_transport -->
```python
import aiohttp
from axio import Agent, MemoryContextStore
from axio_transport_openai import ChatGPTTransport
from axio_transport_openai.session import ChatGPTSession


async def ask(account: ChatGPTSession, http: aiohttp.ClientSession) -> str:
    transport = ChatGPTTransport(account=account, session=http)
    await transport.fetch_models()
    choices = [model for model in transport.catalog if model.visibility == "list"]
    if not choices:
        raise ValueError("This account has no visible models.")
    selected = transport.with_model(choices[0].id)
    agent = Agent(system="Be concise.", transport=selected)
    return await agent.run("Hello", MemoryContextStore())
```

The catalog retains `slug` as `id`, `display_name`, `visibility`, and server order.
`models` contains visible entries; `catalog` also retains hidden entries. Refresh
with `fetch_models()`. A failed refresh clears the catalog and raises its error.
There is no static fallback. A removed selected model requires a new selection.

`with_model()` shares the account session, connection pool, and catalog. Each new
request reads current credentials, including rotated tokens. `with_account()`
creates an independent catalog for the new account; existing transports remain
bound to their original account. Fetch the new catalog before showing its choices.
`stream()` fetches lazily if needed. Keep both sessions open until consumers finish.

Capabilities and token budgets come from the selected account's API catalog.
All catalog models are treated as supporting text and function tools, including
new model IDs. `input_modalities` enables vision when it contains `image`;
`supported_reasoning_levels` enables reasoning and restricts explicit effort values
in payloads. Unsupported efforts raise `ValueError` before inference, including
after `with_model()` changes the selected model. When the field is absent, an
explicitly supplied reasoning-capable `ModelSpec` uses the generic effort levels.
`context_window` is the configured
context budget; `max_context_window` does not replace it. Positive
`max_output_tokens` values are used when the server provides them.

Missing metadata uses explicitly supplied `model_specs=[...]`, or generic token
budgets when no override exists. The ChatGPT transport does not consult the static
`OPENAI_MODELS` registry. `ChatGPTModel.input_cost` and `output_cost` are `NaN`:
subscription pricing is unknown here, not zero. API-key prices do not describe
subscription billing.

Requests include the complete history, encrypted reasoning, and tool results.
History system messages become developer messages. Current function declarations
occupy one developer `additional_tools` item at the start of `input`, without
namespaces. Its position stays fixed across turns. A selector or schema change
replaces this declaration; old calls remain in history. This can invalidate cached
prefixes, but it does not retain removed tools or share tool state across contexts.

`reasoning_effort` and `reasoning_summary` configure known reasoning models. The
builder always sets `store=false` and `stream=true`; it exposes no arbitrary extra
parameters or unsupported sampling/output limits. Audio, video, unsupported hosted
tool history, and images without verified model support fail before inference.
Embedding and realtime interfaces are not exposed.

The subscription API also accepts inline `input_file` attachments, including PDF
and text files. The Files upload API is unavailable on this route. Axio's current
content-block API does not expose native file attachments: the REPL reads local
text files through tools and sends images as `ImageBlock`. Callers that construct
raw Responses inputs can use `ChatGPTResponses` for inline files.

See the [subscription route restrictions](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
and [account model discovery](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).

### Token renewal and application storage

`ChatGPTSession` renews tokens in the background inside an async context manager.
`access_token()` also checks expiry and renews on demand. A session stays pinned
to its issued client ID. Selecting another account in the application does not
change an existing session.

```python
from collections.abc import Awaitable, Callable

import aiohttp
from axio_transport_openai.credentials import CredentialStore
from axio_transport_openai.oauth import ChatGPTOAuth
from axio_transport_openai.session import ChatGPTSession


async def with_chatgpt(
    store: CredentialStore,
    host_id: str,
    client_id: str,
    run: Callable[[ChatGPTSession], Awaitable[None]],
) -> None:
    async with aiohttp.ClientSession() as http:
        oauth = ChatGPTOAuth(http, host_id)
        async with ChatGPTSession(store, oauth, client_id) as session:
            await run(session)
```

Inside `run`, call `await session.access_token()` before each new API request.
Keep the HTTP session open until the ChatGPT session closes. `start()` and
`close()` provide the same lifecycle without a context manager. Closing stops
background renewal and waits for active credential commits; it does not sign out.
Cancellation of an exchange waits for its result to be stored before propagating.

Use `FileCredentialStore` or `SQLiteCredentialStore` from
`axio_transport_openai.credential_stores`, or implement `CredentialStore` for
PostgreSQL, Redis, or another backend. The interface has three methods:

These classes bind the shared `axio.credentials` backends to `CredentialRecord`.
The transport supplies the dataclass and filename. Core owns serialization,
file protection, process locks, and cancellation handling.

| Method | Required behavior |
|---|---|
| `lock(client_id)` | Return an async context manager that excludes writers for this registration across all participating processes. Keep the lock through the HTTP exchange and each save. |
| `load(client_id)` | Return the latest `CredentialRecord`, or `None`. Read under the lock; do not return a stale cached record. |
| `save(record)` | Atomically commit the entire record before returning. Raise `CredentialError` with a safe message on failure. Call under the lock. |

Each `save` must be durable independently. The lock is not a database transaction
that rolls back earlier saves when an exception occurs. Distributed lock leases
must stay exclusive through the complete operation. Protect stored secrets with
file permissions, database access controls, or equivalent. Persist a stable host
ID separately and retain it after sign-out. Key registrations by issued client
ID and verified identity; use email only as a display label.

Store every `CredentialRecord` field: `account`, `revision`, and `operation`.
The account contains the complete token set, scopes, expiry, and verified identity.
A recovery marker is saved before refreshing. If saving rotated tokens fails,
the same live session can retry the save without another token exchange. After
a process crash with that marker present, reauthorize the same registration;
the old refresh token may already have been spent. `record.public_account()`
omits tokens when such an operation was interrupted.

Use `save_account(store, account)` from `axio_transport_openai.credentials` to
save an initial validated sign-in. For returning sign-in, load the existing record
under its lock, pass `record.public_account()` to `oauth.sign_in()`, then call
`save_account(store, result, expected=record)`. `CredentialConflict` means another
operation changed the record during sign-in; it prevents overwriting newer tokens.

Renewal normally starts 60 seconds before expiry, or halfway through a shorter
token lifetime. Failures before the refresh request and explicit temporary HTTP
errors retain credentials and retry with bounded backoff. A still-valid access
token remains usable on those paths. A lost response or timeout after sending
retains the recovery marker: the old refresh token must not be replayed.
Reauthorize the same registration to recover from that uncertain outcome.
`last_error` exposes background
failures without logging secrets. Terminal refresh errors clear tokens and raise
`SignInRequired`; missing plan consent raises `PlanConsentRequired`. Neither case
switches accounts or billing modes.

`await session.sign_out()` stops renewal, revokes the latest refresh token, and
clears local tokens while retaining the registration. Its `SignOutResult` contains
`revocation_confirmed`. If false, local tokens are cleared but remote revocation
could not be confirmed. Storage failures raise `CredentialError` instead of
reporting successful local sign-out.

See the [OpenAI account lifecycle specification](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions).

### Subscription request execution

`ChatGPTResponses` in `axio_transport_openai.subscription` sends a prepared subscription
request through a `ChatGPTSession` and an `aiohttp.ClientSession`. It provides HTTP/SSE
execution for transport integrations; it does not build agent requests or discover models.
The caller must supply a subscription-compatible body and keep both sessions open.

`stream(payload)` checks plan consent before sending and reads through response completion.
HTTP and SSE failures raise `axio_responses.ResponsesAPIError`, which retains the provider code,
parameter, HTTP status, request ID, and diagnostic body. `ResponsesIncompleteError` and
`ResponsesInterruptedError` distinguish incomplete responses from disconnected streams.
`SignInRequired` and `PlanConsentRequired` also remain catchable as `StreamError`.

Temporary failures receive at most `max_attempts` attempts, including the initial request.
Backoff uses `Retry-After` or `retry_base_delay` and is capped at 300 seconds. No retry occurs
after an event reaches the caller. Authentication and eligibility failures retain credentials;
they do not trigger OAuth loops or a switch to API-key billing.

A quota error sets `session.plan_usage_error` and pauses new requests through that session,
including requests from another executor sharing it. Direct the user to ChatGPT settings →
Usage. After resolving the limit, the caller can explicitly call `session.resume_plan_usage()`.
The session does not infer a reset time. This pause is local to the session, not persisted
across processes. Token renewal remains independent of the inference pause.

See the [OpenAI recovery rules](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery).

### File and SQLite credentials

```python
from pathlib import Path
from axio_transport_openai.credential_stores import FileCredentialStore, SQLiteCredentialStore

file_store = FileCredentialStore()
sqlite_store = SQLiteCredentialStore(Path.home() / ".config" / "axio" / "chatgpt.db")
```

`FileCredentialStore()` defaults to `$XDG_CONFIG_HOME/axio/chatgpt.auth`, or
`~/.config/axio/chatgpt.auth` when the XDG variable is unset, empty, or relative.
Pass a path to override it. The file holds all registrations and the host ID.
Writes replace the complete file atomically. A short file-update lock prevents
concurrent writes to different registrations from overwriting each other.

`SQLiteCredentialStore(path)` uses a caller-selected database. Each load opens
a fresh `mode=ro` connection, reads, and closes within one worker-thread call.
Each write opens its own connection, commits, and closes in one worker thread.
Reads retain SQLite locking and change detection. `immutable=1` is unsuitable for
a database that other connections modify; SQLite documents possible incorrect
results and `SQLITE_CORRUPT` errors. See [SQLite URI parameters](https://www.sqlite.org/uri.html).

Both stores provide `await store.host_id()` to create or retrieve the stable host
identity. Each `save` commits separately, including recovery markers. An adjacent
`<filename>.locks` directory holds per-registration process locks across OAuth
requests. Keep that directory intact while applications use the store. Neither
backend needs a `close()` call.

The built-in stores require a local POSIX filesystem with `flock`, atomic rename,
and durable writes. They create directories with mode `0700` and files with mode
`0600`. Existing directories must belong to the current user and must not be
writable by other users. Insecure files, symlinks, and hard-linked files are
rejected. Credentials use filesystem access controls; they are not encrypted.
Custom backends can support other platforms or shared database services.

`LocalCredentialStore.read_value()` and `write_value()` are public extension
methods inherited from `axio.credentials.Credentials`. They exchange dictionaries
and run in worker threads under the corresponding registration lock. JSON is
encoded only at the file or SQLite boundary.
`SQLiteCredentialStore.connect()` can also be overridden; its caller closes the
connection. Shared file helpers are public in `axio.credentials.files`, and
`CredentialCodec` provides typed serialization in `axio.credentials`.
The previous `axio_transport_openai.credential_files` imports remain available.

Use `accounts.list_accounts(store)` for metadata without tokens.
The generic `store.list_records()` returns complete records and must not be used
directly as CLI output. Existing credential files and SQLite databases retain
their records, selection, host identity, and recovery markers.

### Terminal and SSH sign-in

Install this package to get the `axio-chatgpt` command. It does not require Textual.

```bash
axio-chatgpt login
axio-chatgpt accounts
axio-chatgpt select CLIENT_ID
axio-chatgpt status
axio-chatgpt models
```

`login` saves and selects a registration only after successful authorization.
Use `login --account CLIENT_ID` to reauthorize it; plain `login` registers another
account. Account references are issued client IDs, so accounts with the same email
remain distinct. Identity-only authorization succeeds, but `status` reports
`consent_required` until the user grants ChatGPT plan access.

`accounts`, `status`, and `models` support `--json`. Status is local and offline;
`ready` describes saved credentials, not remaining plan quota. Expired credentials
with a refresh token report `refresh_required`; opening a session renews them.
`models` lists the selected account's visible catalog and fails without a fallback
if discovery fails. Commands do not print token values or provider error bodies.

Pass `--credentials PATH` to any command to override the default XDG file.
`status`, `models`, and `logout` accept `--account CLIENT_ID` without changing the
saved selection. Otherwise, commands use the saved selection, or the sole stored
registration. Multiple registrations without a selection require an explicit choice.
Selection affects new sessions; running sessions keep their original account.

`axio-chatgpt logout` clears local tokens and attempts server revocation. It keeps
the registration and selection for returning sign-in. If revocation fails, it
prints a warning after clearing local credentials. Command failures exit with
status 1, invalid arguments with 2, and cancellation with 130.

For the terminal agent, install `axio-repl` and run:

```bash
axio-repl --transport chatgpt --model gpt-6-luna
```

On first use with an empty credential store, the REPL offers sign-in and continues
after browser authorization. This requires terminal input; piped input does not
prompt. The separate `axio-chatgpt login` command is optional for this workflow.

The [standalone example](examples/chatgpt_agent.py) runs an agent with a local
multiplication tool and the selected saved account:

```bash
python axio-transport-openai/examples/chatgpt_agent.py --model gpt-6-luna
```

Run this example from the repository root. It uses `accounts.open_chatgpt()` to
resolve the account, fetch its models, and close background renewal on exit.
The caller owns the HTTP session. Credentials remain saved for later runs.

Use `axio-chatgpt login --manual` over SSH. The standalone sign-in module remains
available for callers that manage selection themselves:

```bash
python -m axio_transport_openai.auth
```

This command signs in and saves credentials in the default XDG file. It prints
the issued client ID after saving. Use `--client-id ID` for returning sign-in,
`--credentials PATH` to override the file, and `--timeout SECONDS` to change the
default five-minute timeout.

When no browser is available, or when Axio runs over SSH, the command prints the
authorization URL to stdout and accepts the final redirect URL on stdin. Use
`--manual` to select this behavior explicitly:

```bash
ssh user@remote
python -m axio_transport_openai.auth --manual
```

Open the printed URL in your local browser and approve sign-in. The browser may
then report that it cannot connect to `127.0.0.1`. Copy the complete URL from its
address bar and paste it into the remote terminal. No SSH port forwarding is
required. The pasted callback uses the same state and identity checks as a
browser callback. The command does not print access, refresh, or ID tokens.

A browser callback can still complete the flow while stdin is waiting. For SSH
port forwarding instead of pasting, start SSH with
`-L 1455:127.0.0.1:1455` and run the command with `--port 1455`.

Library callers can supply an asynchronous `read_callback` function to
`ChatGPTOAuth.sign_in()`. It must return the complete redirect URL. The callback
listener remains active while the function waits; completion, cancellation, or
timeout cancels the pending input reader.

### Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Default API key if not passed to the constructor |
| `OPENAI_BASE_URL` | Default base URL (falls back to `https://api.openai.com/v1`) |

### Local models (Ollama, vLLM, LM Studio)

```python
from axio.models import ModelSpec, Capability
from axio_transport_openai import OpenAITransport

transport = OpenAITransport(
    api_key="ollama",  # any non-empty string
    model=ModelSpec(id="llama3.2", capabilities=frozenset({Capability.text})),
    base_url="http://localhost:11434/v1",
)
```

### Streaming events

```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio.events import TextDelta, SessionEndEvent

agent = Agent(
    system="",
    tools=[],
    transport=StubTransport([make_text_response("Why did the chicken cross the road?")]),
)


async def main() -> None:
    ctx = MemoryContextStore()
    async for event in agent.run_stream("Tell me a joke", ctx):
        match event:
            case TextDelta(delta=text):
                print(text, end="", flush=True)
            case SessionEndEvent(total_usage=usage):
                print(f"\n[{usage.input_tokens}in / {usage.output_tokens}out tokens]")


asyncio.run(main())
```

## Configuration reference

`OpenAITransport` is a dataclass with the following fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"OpenAI"` | Display name (used by TUI) |
| `api_key` | `str` | `$OPENAI_API_KEY` | API key |
| `base_url` | `str` | `$OPENAI_BASE_URL` or `https://api.openai.com/v1` | API base URL |
| `model` | `ModelSpec` | `OPENAI_MODELS["gpt-4.1-mini"]` | Active model |
| `models` | `ModelRegistry` | all `OPENAI_MODELS` | Available models |
| `session` | `aiohttp.ClientSession \| None` | `None` | HTTP session (required for streaming) |
| `max_retries` | `int` | `10` | Maximum retry attempts on 429 / 5xx |
| `retry_base_delay` | `float` | `5.0` | Base delay in seconds for exponential backoff |

## Models

| Model ID | Context | Max output | Capabilities | Price (in/out per M tokens) |
|---|---|---|---|---|
| `gpt-5.4` | 1,050,000 | 128,000 | text, vision, tool use | $10 / $40 |
| `gpt-5.4-mini` | 400,000 | 128,000 | text, vision, tool use | $1.50 / $6 |
| `gpt-5.4-nano` | 400,000 | 128,000 | text, tool use | $0.30 / $1.20 |
| `gpt-5.1` | 400,000 | 128,000 | text, vision, tool use | $5 / $20 |
| `gpt-5` | 400,000 | 128,000 | text, vision, tool use | $5 / $20 |
| `gpt-5-mini` | 400,000 | 128,000 | text, vision, tool use | $1.25 / $5 |
| `gpt-5-nano` | 400,000 | 128,000 | text, tool use | $0.25 / $1 |
| `o4-mini` | 200,000 | 100,000 | text, reasoning, tool use | $1.10 / $4.40 |
| `o3` | 200,000 | 100,000 | text, reasoning, tool use | $10 / $40 |
| `o3-mini` | 200,000 | 100,000 | text, reasoning, tool use | $1.10 / $4.40 |
| `gpt-4.1` | 1,047,576 | 32,768 | text, vision, tool use | $2 / $8 |
| `gpt-4.1-mini` | 1,047,576 | 32,768 | text, vision, tool use | $0.40 / $1.60 |
| `gpt-4.1-nano` | 1,047,576 | 32,768 | text, tool use | $0.10 / $0.40 |
| `gpt-4o` | 128,000 | 16,384 | text, vision, tool use | $2.50 / $10 |
| `gpt-4o-mini` | 128,000 | 16,384 | text, vision, tool use | $0.15 / $0.60 |

The default model is `gpt-4.1-mini`.

## `fetch_models()`

`await transport.fetch_models()` resets `transport.models` to the built-in `OPENAI_MODELS` registry. It does not make a network request. Override `model` directly to switch the active model.

## Serialisation

`OpenAITransport` supports JSON round-trip for storing and restoring configuration:

```python
# Serialise
data = transport.to_dict()  # -> {"name": ..., "base_url": ..., "api_key": ..., "models": [...]}

# Restore
transport = OpenAITransport.from_dict(data, session=session)
```

`from_dict` falls back to `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables if the stored values are empty.

## Sub-transports

### NebiusTransport

`NebiusTransport` connects to [Nebius AI Studio](https://studio.nebius.com/) (`https://api.tokenfactory.nebius.com/v1`). It inherits all retry and streaming behaviour from `OpenAITransport`.

```python
from axio_transport_openai.nebius import NebiusTransport

transport = NebiusTransport(
    api_key="...",  # or set NEBIUS_API_KEY
    session=session,
)
```

| Field | Default |
|---|---|
| `name` | `"Nebius AI Studio"` |
| `api_key` | `$NEBIUS_API_KEY` |
| `base_url` | `https://api.tokenfactory.nebius.com/v1` |
| `model` | `deepseek-ai/DeepSeek-V3-0324` |

`fetch_models()` queries `/v1/models?verbose=true` and populates `transport.models` with all models returned by the API, including their context windows, output limits, capabilities (text, vision, tool use, embedding), and pricing.

### OpenRouterTransport

`OpenRouterTransport` connects to [OpenRouter](https://openrouter.ai/) (`https://openrouter.ai/api/v1`), which provides a unified API over hundreds of models from many providers.

```python
from axio_transport_openai.openrouter import OpenRouterTransport

transport = OpenRouterTransport(
    api_key="...",  # or set OPENROUTER_API_KEY
    session=session,
)
```

| Field | Default |
|---|---|
| `name` | `"OpenRouter"` |
| `api_key` | `$OPENROUTER_API_KEY` |
| `base_url` | `https://openrouter.ai/api/v1` |
| `model` | `google/gemini-2.5-pro-preview` |

`fetch_models()` queries `/v1/models` and populates `transport.models` with all models returned by the API, including their context windows, output limits, capabilities (text, vision, tool use, embedding), and pricing.

### OpenAICompatibleTransport

`OpenAICompatibleTransport` is a thin subclass of `OpenAITransport` for user-defined custom providers. Instances are created by the TUI hub screen and persisted to `~/.local/share/axio/openai-custom.json`. You can also instantiate them directly:

```python
from axio.models import ModelSpec, ModelRegistry, Capability
from axio_transport_openai.custom import OpenAICompatibleTransport

transport = OpenAICompatibleTransport(
    name="localai",
    base_url="http://localhost:8080/v1",
    api_key="",
    models=ModelRegistry(
        [
            ModelSpec(
                id="llama3.2",
                context_window=131_072,
                max_output_tokens=4_096,
                capabilities=frozenset({Capability.text, Capability.tool_use}),
            )
        ]
    ),
    session=session,
)
```

`fetch_models()` is a no-op for this transport - the model list is provided at construction time.

The JSON configuration format used by the TUI is:

```json
[
  {
    "name": "localai",
    "base_url": "http://localhost:8080/v1",
    "api_key": "",
    "models": [
      {
        "id": "llama3.2",
        "context_window": 131072,
        "max_output_tokens": 4096,
        "capabilities": ["text", "tool_use"],
        "input_cost": 0.0,
        "output_cost": 0.0
      }
    ]
  }
]
```

## Plugin registration

When installed, this package registers these transports via entry points. ChatGPT
requires an authenticated account session supplied by its caller. TUI opens this session
through `open_for_app()` and uses `with_model()` to retain shared credentials.
Its **Configure ChatGPT** screen provides login, account selection, permission status,
sign-out, and a [ChatGPT Usage link](https://chatgpt.com/settings/usage), as described in
the [official recovery guidance](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery).
The screen stores only the credential-file path and registration reference in TUI settings.

```toml
[project.entry-points."axio.transport"]
chatgpt        = "axio_transport_openai:ChatGPTTransport"
openai         = "axio_transport_openai:OpenAITransport"
nebius         = "axio_transport_openai.nebius:NebiusTransport"
openrouter     = "axio_transport_openai.openrouter:OpenRouterTransport"
openai-custom  = "axio_transport_openai.custom:OpenAICompatibleTransport"
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-transport-codex](https://github.com/mosquito/axio-agent) · [axio-transport-anthropic](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
