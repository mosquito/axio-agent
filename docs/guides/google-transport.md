# Google Transport

`axio-transport-google` provides a Gemini transport for both standard
completion and realtime (Gemini Live) sessions. It supports the Google
GenAI Developer API and Vertex AI.

## Install

```bash
pip install axio-transport-google
```

For Vertex AI with application-default credentials:

```bash
pip install "axio-transport-google[vertexai]"
```

## Quick start

Set the API key and create a transport:

```bash
export GEMINI_API_KEY="..."
```

```python
from axio_transport_google import GoogleTransport

transport = GoogleTransport()
```

The transport starts on `gemini-3.1-flash-lite-preview`, the cheapest of the three chat
models. Switch it before the first call if you want another one.

## Antigravity subscription

Select `antigravity` explicitly to use Google OAuth and Antigravity account access:

```bash
uv run axio-antigravity login
uv run axio-repl --transport antigravity
```

`AntigravityTransport` lives in `axio_transport_google.antigravity`. It implements
completion, streaming, reasoning, and tools. Its catalogue comes from the selected
account, so use `/model` or `axio-antigravity models` to find model IDs. Developer
API model IDs are not interchangeable with those IDs. Image/video generation,
Claude, and Gemini Live are outside this transport's scope.

<!-- name: test_antigravity_public_api -->
```python
import aiohttp
from axio import Agent, MemoryContextStore
from axio_transport_google.antigravity import AntigravityTransport


async def ask_antigravity(prompt: str) -> str:
    async with aiohttp.ClientSession() as http:
        transport = AntigravityTransport(session=http)
        await transport.fetch_models()
        agent = Agent(system="Answer the question.", transport=transport)
        return await agent.run(prompt, MemoryContextStore())
```

The caller owns the HTTP session. `with_model(transport.models[model_id])` returns
a transport that shares the pinned account session. Token refresh uses the core
credential store's lock, including across processes. The default file is
`${XDG_CONFIG_HOME:-$HOME/.config}/axio/antigravity.auth`, with mode `0600`.
The file is not encrypted. `AntigravitySQLiteCredentials` provides the SQLite
backend; applications can pass their own compatible store to `AntigravitySession`.

For SSH, `axio-antigravity login --manual` accepts the full redirect URL copied
from the browser. `accounts`, `select --account ID`, `status`, and `logout` manage
saved accounts. `logout --revoke` also requests revocation from Google. These
commands remain usable when project or catalogue discovery fails.

The transport never switches accounts, models, or payment methods to recover from
quota or permission errors. Missing project discovery requires setup in the
Antigravity application. Axio does not create Cloud projects. HTTP 429 and server
failures have bounded retries that respect `Retry-After`. A stream is not retried
after its first payload.

This is an unofficial internal API. [Google's Antigravity terms](https://antigravity.google/terms)
restrict third-party OAuth use and permit suspension. API stability is not guaranteed.

### Antigravity in the TUI

Install `axio-tui[google]`, or run `uv run axio` in this workspace. Open
**Configure Antigravity** in the command palette to sign in or select an account.
**Use account** reloads the available models. The screen uses the same credential
store as the REPL and remains available when discovery fails. It also accepts
pasted browser redirect URLs for SSH sessions. Closing it cancels pending sign-in.
**Sign out** clears local tokens; Google grant revocation remains an explicit CLI
operation with `axio-antigravity logout --revoke`.

## Models

| Model ID | Capabilities | Context | Notes |
|---|---|---|---|
| `gemini-3.1-pro-preview` | text, vision, audio, video, tools, reasoning | 1M tokens | Flagship |
| `gemini-3-flash-preview` | text, vision, audio, video, tools, reasoning | 1M tokens | Fast/cheap |
| `gemini-3.1-flash-lite-preview` | text, vision, audio, video, tools, reasoning | 1M tokens | Lightest; the default |
| `gemini-3.1-flash-image-preview` | text, vision, image generation | 1M tokens | Nano Banana |
| `gemini-3-pro-image-preview` | text, vision, image generation | 1M tokens | Image gen |

## Switching models

```python
from axio_transport_google import GoogleTransport
from axio.models import Capability

transport = GoogleTransport()

# Switch to a specific model
transport.model = transport.models["gemini-3-flash-preview"]

# Find the cheapest reasoning model
transport.model = transport.models.by_capability(Capability.reasoning).by_cost().first()
```

## Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `GEMINI_API_KEY` env var | API key for the Developer API |
| `model` | `gemini-3.1-flash-lite-preview` | Active `ModelSpec` |
| `temperature` | `None` | Sampling temperature (uses model default if unset) |
| `top_p` | `None` | Nucleus sampling probability |
| `top_k` | `None` | Top-k sampling |
| `seed` | `None` | Random seed for deterministic outputs |
| `thinking_budget` | `None` | Token budget for chain-of-thought reasoning |
| `thinking_level` | `None` | Gemini 3+ thinking level (see below) |
| `max_output_tokens` | `None` | Override the model's default max output |
| `max_retries` | `5` | Retries on 429/500/503 with exponential backoff |
| `safety_settings` | `None` | List of `SafetySettingDict` (see below) |
| `debug` | `False` | Log raw request/response bodies |
| `service_tier` | `None` | Forwarded as `generationConfig.serviceTier` |
| `media_resolution` | `None` | Forwarded as `generationConfig.mediaResolution`, upper-cased |
| `nudge_on_media_tool_result` | `True` | Append a short user message after a tool returns media (see below) |
| `vertexai` | `GOOGLE_GENAI_USE_VERTEXAI` env var | Route through Vertex AI instead of the Developer API |
| `project` | `GOOGLE_CLOUD_PROJECT` env var | Vertex AI project |
| `location` | `GOOGLE_CLOUD_LOCATION` env var | Vertex AI location |

### Thinking level

`thinking_level` applies to Gemini 3+ models, which take a level rather than a budget.
What is valid depends on the model family:

| Model | Valid levels |
|---|---|
| `-pro` | `LOW`, `MEDIUM`, `HIGH` |
| `-pro-image` | `HIGH` |
| `-flash-image` | `MINIMAL`, `HIGH` |
| Flash, Flash-Lite | `MINIMAL`, `LOW`, `MEDIUM`, `HIGH` |

There is no `NONE`. A value the family does not support is silently replaced by its highest
level, so a misspelling buys the most expensive setting rather than failing. A reasoning model
left unset also gets `HIGH`. `thinking_budget` is the Gemini 2.5 form. It is not sent for a
Gemini 3+ model.

### Media tool results

Gemini stops generating after about twenty tokens when media arrives as sibling `inlineData`
parts beside a `functionResponse`. With `nudge_on_media_tool_result` left on, the agent appends
a short "Proceed." message so the model actually looks at the content. `Agent` reads the flag
off the transport, so this Google-specific behaviour stays in a Google-specific field.

### Reasoning signatures

Gemini signs the reasoning it produces. The signature has to come back unaltered on the next
request. The transport emits it as `ReasoningSignature`. The agent stores it on the `ReasoningBlock`
in the turn. The transport puts it back on the part Gemini signed: a thought part, a function-call part, or a
plain text part. A proof on answer text travels as `TextSignature` and is stored on the
`TextBlock`; the other two travel as `ReasoningSignature` and `ToolUseStart.signature`. A signature that is missing, altered or attached to the
wrong part comes back as the finish reason `MISSING_THOUGHT_SIGNATURE`, which maps to
`StopReason.error`.

The consequence for a context store: `ReasoningBlock.signature`, `ToolUseBlock.signature` and
`TextBlock.signature` must all survive the round trip through `to_dict`/`from_dict`. Drop it and the *next* turn fails, not the one that dropped it. See
{doc}`writing-transports` for the three providers' replay shapes.

### Grounding and citations

Gemini's `citationMetadata` and `groundingMetadata` do not map onto axio's `Citation`, because the
shapes do not line up. The transport therefore forwards them whole as
`ProviderEvent(provider="google")`, with `kind` set to the metadata's own name. Anything else the
API sends that axio has no type for (`executableCode`, `codeExecutionResult`, `fileData`) arrives
the same way under `kind="part"`.

## Vertex AI

Use `VertexAITransport` to route through Google Cloud Vertex AI instead of
the Developer API. It reads credentials from application-default credentials
(`gcloud auth application-default login`).

```python
from axio_transport_google import VertexAITransport

transport = VertexAITransport(
    project="my-gcp-project",
    location="us-central1",
)
```

Or set environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="1"
```

On Vertex AI you can also use Anthropic models with a `anthropic/` prefix:

```python
transport.model = transport.models["anthropic/claude-opus-4-6"]
```

## Safety settings

Override the default safety thresholds:

```python
from axio_transport_google import GoogleTransport
from axio_transport_google._generated_types import SafetySetting

transport = GoogleTransport(
    safety_settings=[
        SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
        ),
    ]
)
```

## Image generation

When the selected model supports `Capability.image_generation`, the transport
exposes `generate_images`:

```python
import asyncio
from axio_transport_google import GoogleTransport


async def main() -> None:
    transport = GoogleTransport()
    images: list[bytes] = await transport.generate_images(
        "A photorealistic owl sitting on a branch",
        model="gemini-3.1-flash-image-preview",
        n=1,
    )
    with open("owl.png", "wb") as f:
        f.write(images[0])


asyncio.run(main())
```

## Video generation

```python
import asyncio
from axio_transport_google import GoogleTransport


async def main() -> None:
    transport = GoogleTransport()
    videos: list[bytes] = await transport.generate_videos(
        "Time-lapse of clouds moving over mountains",
        model="veo-3.1-fast-generate-001",
        duration_seconds=6,
        aspect_ratio="16:9",
    )
    with open("timelapse.mp4", "wb") as f:
        f.write(videos[0])


asyncio.run(main())
```

Video generation runs an async polling loop until the job completes.

## Tools registered as entry points

When installed, `axio-transport-google` registers two tools under `axio.tools`:

| Entry point | Tool | Description |
|---|---|---|
| `generate_image` | `generate_image` | Generate images via Gemini Nano Banana |
| `generate_video` | `generate_video` | Generate videos via Veo |

Pass these handlers to `Tool` explicitly, or use them through `axio-repl`.

## Realtime (Gemini Live)

For low-latency voice conversations, use `GeminiLiveTransport` with
`RealtimeAgent`. See the {doc}`realtime-audio` guide for the full setup.

```python
from axio_transport_google.realtime import GeminiLiveTransport
from axio.realtime import RealtimeAgent

transport = GeminiLiveTransport()

async with RealtimeAgent(
    system="You are a helpful assistant.", transport=transport
) as agent:
    ...
```

For Vertex AI realtime, use `VertexLiveTransport`. If you have multiple Vertex
regions available, the transport can auto-select the nearest one:

```python
from axio_transport_google.realtime import (
    VertexLiveTransport,
    probe_nearest_live_region,
)

region = await probe_nearest_live_region()
transport = VertexLiveTransport(location=region)
```
