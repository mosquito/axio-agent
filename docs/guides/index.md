# How-To Guides

Practical guides for composing and extending Axio in your applications.

Start with {doc}`choosing-transports` to discover installed providers, select a
model, and run an agent through core without importing REPL or TUI.

Extending Axio means writing one of four things: a tool, a guard, a context store, or a transport.
{doc}`writing-transports` is the longest of them, because a transport is the only place that knows a
provider exists. It reads the provider's stream through [`axio-sse`](../api/sse.md), converts its
token counts, maps its stop reasons, and replays its reasoning.

```{toctree}
:maxdepth: 1

axio-repl
choosing-transports
google-transport
realtime-audio
mcp-tools
multimodal
writing-tools
writing-transports
writing-guards
writing-context-stores
docker-sandbox
testing
cookbook
best-practices
agent-swarm
gas-town
```
