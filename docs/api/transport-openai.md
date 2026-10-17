# `axio-transport-openai`

OpenAI and OpenAI-compatible completion transports.

`api` selects the endpoint. `OpenAITransport` defaults to `"responses"` and posts to
`/v1/responses`, reading it through {doc}`responses`. `NebiusTransport`,
`OpenRouterTransport` and `OpenAICompatibleTransport` set `"chat"` and post to
`/v1/chat/completions`. That endpoint refuses function tools beside any reasoning effort
other than `"none"` — see {ref}`the troubleshooting entry <tools-and-reasoning-400>`.

`extra_params` is folded into the request. Its `tools` are merged with the agent's
function declarations rather than replacing them, so adding a hosted tool does not take
away the calls the agent is there to dispatch.

```{eval-rst}
.. autoclass:: axio_transport_openai.OpenAITransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.nebius.NebiusTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.openrouter.OpenRouterTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.custom.OpenAICompatibleTransport
   :members:
```

## ChatGPT subscription

`ChatGPTTransport` uses a pinned `ChatGPTSession` and the public Responses endpoint.
It discovers visible models for that account, builds subscription-compatible requests,
and delegates HTTP/SSE recovery to `ChatGPTResponses`. Model copies share the account
session and catalog; account changes create a separate catalog. Subscription prices
are unknown (`NaN`). All catalog models receive function tools. Vision, reasoning,
and context limits come from the API metadata instead of the API-key registry.

Install `axio-transport-openai` and run `axio-chatgpt login`
to save and select a registration in the default XDG credentials file. Use `--manual`
for sign-in over SSH. The command prints an issued client ID after successful sign-in.
Keep this ID as the account reference; keep tokens in the credential store.

Use `axio-chatgpt accounts`, `select CLIENT_ID`, `status`, `models`, and `logout`
to manage saved accounts. `status` is offline; `models` fetches the account catalog.
These commands accept `--credentials PATH`. Status and model listing accept
`--json` and `--account CLIENT_ID`; the latter does not change the saved selection.
Reauthorize a registration with `axio-chatgpt login --account CLIENT_ID`.

<!-- name: test_saved_chatgpt_session -->
```python
import aiohttp
from axio import Agent, MemoryContextStore
from axio_transport_openai import ChatGPTTransport
from axio_transport_openai.credential_stores import FileCredentialStore
from axio_transport_openai.oauth import ChatGPTOAuth
from axio_transport_openai.session import ChatGPTSession


async def ask_saved_account(client_id: str, model_id: str) -> str:
    store = FileCredentialStore()
    async with aiohttp.ClientSession() as http:
        oauth = ChatGPTOAuth(http, await store.host_id())
        async with ChatGPTSession(store, oauth, client_id) as account:
            transport = ChatGPTTransport(account=account, session=http)
            await transport.fetch_models()
            selected = transport.with_model(model_id)
            agent = Agent(system="Be concise.", transport=selected)
            return await agent.run("Hello", MemoryContextStore())
```

Choose `model_id` from that account's visible catalog. Image input is enabled by
the catalog's `input_modalities`. Explicit `model_specs` supply fallback metadata
for missing fields; a new model ID does not disable tools. Session closure stops
background renewal and retains credentials for the next process. Conversation persistence is separate;
use a persistent context store or serialize `Message` objects to resume history.

`PlanConsentRequired` means identity sign-in succeeded without subscription
permission. `SignInRequired` requires reauthorization of the selected registration.
`ResponsesAPIError` preserves the HTTP status, provider code, and request ID.
Quota exhaustion pauses inference for the shared account session. Call
`resume_plan_usage()` only after resolving the limit. Errors never select another
account or switch to API-key billing.

Run `axio-repl --transport chatgpt --model gpt-6-luna` to use the selected account
in the CLI. Scripts can use `accounts.open_chatgpt(store, http, model=...)` to
resolve the selected registration, fetch its catalog, and manage session lifetime.
It also accepts `client_id=` for an explicit account. Selection changes affect
future sessions only. TUI **Configure ChatGPT** provides sign-in, registration selection,
permission status and sign-out. The registry opens managed account sessions and calls
`with_model()` for model copies; generic config stores only account references.
Codex OAuth uses a separate transport and credential flow.

```{eval-rst}
.. autoclass:: axio_transport_openai.ChatGPTTransport
   :members:

.. autoclass:: axio_transport_openai.chatgpt_models.ChatGPTModel
   :members:

.. autofunction:: axio_transport_openai.chatgpt.build_chatgpt_payload

.. autoclass:: axio_transport_openai.session.ChatGPTSession
   :members:

.. autoclass:: axio_transport_openai.credential_stores.FileCredentialStore
   :members:

.. autoclass:: axio_transport_openai.credential_stores.SQLiteCredentialStore
   :members:
```

## Realtime

```{eval-rst}
.. autoclass:: axio_transport_openai.OpenAIRealtimeTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.OpenAIRealtimeSession
   :members:
```
