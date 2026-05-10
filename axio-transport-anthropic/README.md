# axio-transport-anthropic

[![PyPI](https://img.shields.io/pypi/v/axio-transport-anthropic)](https://pypi.org/project/axio-transport-anthropic/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-anthropic)](https://pypi.org/project/axio-transport-anthropic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Anthropic Claude transport for [axio](https://github.com/mosquito/axio-agent).

Streams Claude responses over the Anthropic Messages API using `aiohttp` and
SSE parsing. Supports direct API, Vertex AI, prompt caching, extended thinking,
and automatic retry on rate-limit and overload errors.

## Features

- **All Claude models** — Opus, Sonnet, Haiku; configurable via `ANTHROPIC_MODELS`
- **Vertex AI** — use Claude via Google Cloud with ADC authentication
- **Prompt caching** — `cache_control: ephemeral` applied automatically to the
  system prompt and the last tool definition
- **Extended thinking** — `ReasoningDelta` events emitted for thinking blocks
- **Retry logic** — automatic backoff on 429 / 529; honours `Retry-After` header
- **TUI integration** — settings screen for API key and model selection

## Installation

```bash
pip install axio-transport-anthropic
```

With the TUI settings screen:

```bash
pip install "axio-transport-anthropic[tui]"
```

With Vertex AI support:

```bash
pip install "axio-transport-anthropic[vertexai]"
```

## Usage

```python
import asyncio
import aiohttp
from axio import Agent
from axio.context import MemoryContextStore
from axio_transport_anthropic import AnthropicTransport, ANTHROPIC_MODELS

async def main() -> None:
    async with aiohttp.ClientSession() as session:
        transport = AnthropicTransport(
            api_key="sk-ant-...",
            model=ANTHROPIC_MODELS["claude-sonnet-4-6"],
            session=session,
        )
        agent = Agent(system="You are helpful.", tools=[], transport=transport)
        ctx = MemoryContextStore()
        print(await agent.run("Hello!", ctx))

asyncio.run(main())
```

Set the API key via environment variable instead of passing it directly:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Vertex AI

```python
transport = AnthropicTransport(
    vertexai=True,
    project="my-gcp-project",
    location="us-east5",
    session=session,
)
# Uses Application Default Credentials (gcloud auth application-default login)
```

## Models

| Model ID | Context | Max output | Notes |
|---|---|---|---|
| `claude-opus-4-6` | 1 M | 128 k | Most capable |
| `claude-sonnet-4-6` | 1 M | 64 k | Balanced (default) |
| `claude-haiku-4-5-20251001` | 200 k | 64 k | Fastest / cheapest |
| `claude-opus-4-5` | 200 k | 64 k | — |
| `claude-sonnet-4-5` | 200 k | 64 k | — |

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `""` | Anthropic API key |
| `model` | `claude-sonnet-4-6` | Active model |
| `base_url` | `https://api.anthropic.com/v1` | API base URL |
| `vertexai` | `False` | Use Vertex AI backend |
| `project` | `""` | GCP project ID (Vertex AI) |
| `location` | `""` | GCP region, e.g. `us-east5` (Vertex AI) |
| `max_retries` | `10` | Max retry attempts on 429/529 |
| `retry_base_delay` | `5.0` | Base delay (seconds) for exponential backoff |

## License

MIT
