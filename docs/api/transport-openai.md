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

`ChatGPTTransport` uses a pinned `ChatGPTSession`. Ordinary sign-in uses the public
Responses endpoint; `axio-chatgpt login --codex` uses native ChatGPT OAuth and the
Codex backend for completion and image generation.
It discovers visible models for that account, builds subscription-compatible requests,
and delegates HTTP/SSE recovery to `ChatGPTResponses`. Model copies share the account
session and catalog; account changes create a separate catalog. Subscription prices
are unknown (`NaN`). All server completion models receive function tools. Vision, reasoning,
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
The native sign-in mode saves credentials in the same Axio store. It does not
import the separate `axio-transport-codex` transport's credentials.

### Images

Authorize with `axio-chatgpt login --codex` before generating images. Native login
uses port 1455 by default; `--manual` supports SSH. The ordinary public plan-sharing
token is not accepted by the native image endpoint. The account's authorization
mode is persisted and determines routing after restart.

`ChatGPTImageTransport` implements `ImageGenTransport`. Its `models` registry is
`CHATGPT_IMAGE_MODELS`, currently containing `gpt-image-2` as a `ChatGPTImageModel`
with `Capability.image_generation` and `Capability.image_editing`.
Native `ChatGPTTransport.fetch_models()` also
appends these image presets to the completion catalog. They describe implemented
image support, not guaranteed account entitlement. Filter completion choices by
`Capability.text` and image choices by `Capability.image_generation`.

<!-- name: test_chatgpt_image_generation -->
```python
import aiohttp
from axio_transport_openai import ChatGPTImageTransport
from axio_transport_openai.chatgpt_images import image_media_type
from axio_transport_openai.session import ChatGPTSession


async def draw(account: ChatGPTSession, http: aiohttp.ClientSession) -> tuple[bytes, str]:
    generator = ChatGPTImageTransport(account=account, session=http)
    images = await generator.generate_images("An owl reading a book", quality="low")
    return images[0], image_media_type(images[0])
```

For an open completion transport, `chatgpt.image_transport()` returns this class
with the same sessions. `chatgpt.with_model("gpt-image-2").generate_images(prompt)`
also returns image bytes. Image models reject `stream()` calls.

One image per request (`n=1`) is supported. The standalone generator accepts
`size` (`auto`, `1024x1024`, `1536x1024`, `1024x1536`), `quality` (`auto`, `low`,
`medium`, `high`), and `transparent_background`. Native output is PNG.
Output format selection, partial images, and seeds are not exposed.

The protocol follows [Codex CLI rust-v0.163.0](https://github.com/openai/codex/tree/rust-v0.163.0)
and posts to `/backend-api/codex/images/generations` on ChatGPT. Backend access
and quotas depend on the account. `ChatGPTImageError` preserves API diagnostics;
invalid results raise an error. Quota errors pause the shared account session.
Generation POSTs are never automatically replayed, including after server errors
or cancellation. A connection failure can leave the server outcome unknown.
The caller owns both sessions, and closing them retains credentials.

`ChatGPTImageTransport` also implements `ImageEditTransport` from `axio.transport`.
Use `edit_images(prompt, images=[ImageBlock(...)])` for changes to existing images.
Sources are required and retain their order; PNG/JPEG/WebP bytes are sent as inline
image references to `/backend-api/codex/images/edits`. `ChatGPTTransport.edit_images()`
delegates to the same implementation. The standalone class accepts the generation
size/quality/background settings. The native protocol has no mask field;
`mask` must remain None. Editing shares the account's quota gate and never retries
requests automatically. `image_editing` does not imply `vision`.

`ChatGPTImageTools(chatgpt.image_transport())` creates two `Tool` instances in
`.tools`: `generate_image(prompt)` and `edit_image(prompt, images)`. Attach these
to `Agent.tools`. The model supplies image IDs to the edit tool; the host registers
the actual attachments with `add_image(ImageBlock(...))` and includes their IDs in
the conversation. Results contain new IDs and `ImageBlock`s for subsequent edits.
Use one tool set per conversation. Its read-only `.images` catalog can be persisted
by the host and restored with the constructor's `images=` argument. No filesystem
paths, URLs, or raw base64 are requested from the model. Both tools declare external
access, so standard tool guards apply.

```{eval-rst}
.. autoclass:: axio_transport_openai.ChatGPTTransport
   :members:

.. autoclass:: axio_transport_openai.chatgpt_models.ChatGPTModel
   :members:

.. autoclass:: axio_transport_openai.ChatGPTImageModel
   :members:

.. autoclass:: axio_transport_openai.ChatGPTImageTransport
   :members:

.. autoclass:: axio_transport_openai.ChatGPTImageTools
   :members:

.. autofunction:: axio_transport_openai.chatgpt_images.image_media_type

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

## Standalone API speech

These API-key transports are independent of completion and ChatGPT subscription
credentials. See the [package speech examples](https://github.com/mosquito/axio-agent/tree/master/axio-transport-openai#standalone-speech-api-key-separately-billed)
for billing boundaries, encoding and streaming cleanup requirements.

```{eval-rst}
.. autoclass:: axio_transport_openai.OpenAISTTTransport
   :members:

.. autoclass:: axio_transport_openai.OpenAITTSTransport
   :members:
```
