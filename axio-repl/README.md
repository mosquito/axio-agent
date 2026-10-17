# axio-repl

Interactive REPL coding assistant powered by the [axio](../axio) agent framework.
Uses pluggable transports with API keys or a saved ChatGPT subscription account.

## Philosophy

axio-repl is an opinionated terminal agent that **actually verifies its work**.
The system prompt encodes hard-won lessons from watching models cut corners:

- **Stream everything, hide nothing.** Every piece of information shown to the
  user — tool arguments, stdout/stderr, images, exit codes — must also be
  faithfully presented to the model. The model should see exactly what the user
  sees, so it can reason about the same reality. No summarizing, no truncating,
  no dropping context between what's displayed and what's sent back.
- **Not tested — not done.** The agent must run tests, re-read edited files,
  and observe actual results instead of assuming success from exit codes.
- **Iterative UI review.** When building or modifying UI, the agent captures
  real screenshots at multiple viewport sizes (desktop, tablet, mobile) via
  Playwright/Puppeteer, reads every screenshot through the model's vision, and
  critically lists visual defects. It repeats the screenshot → fix → re-screenshot
  loop until zero defects — no premature "looks good".
- **Ground everything in project context.** Read before editing. List the
  directory before guessing. Never refuse a safe request.
- **Minimal edits.** Don't reformat surrounding code, don't narrate tool calls.
  The user sees the full tool output in the terminal.

## Features

- **Pluggable transports** — auto-detected from API keys via
  `axio.transport` entry points. Ships with support for OpenAI, Anthropic,
  Google (Gemini API & Vertex AI), Nebius, OpenRouter, and Codex.
- **Runtime model switching** — `/model <query>` to switch models mid-session
  without restarting. Capabilities (vision, reasoning, image generation) are
  re-evaluated and the system prompt adapts automatically.
- **Streaming tool arguments** — tool call fields appear incrementally as the
  model generates them, so you see what's happening before execution starts.
- **Streaming tool output** — shell command stdout/stderr streams line-by-line
  in real time instead of buffering until completion.
- **Vision** — `read_file` on images (PNG, JPG, GIF, WebP) and videos returns
  multimodal content blocks. The model sees the actual pixels, not a description.
- **Image & video generation** — when the Google transport is installed,
  `generate_image` and `generate_video` tools are available for Gemini Nano
  Banana / Veo models.
- **AGENTS.md** — global and project instructions loaded at startup, with source
  paths and directory scopes. Offline diagnostics explain which files apply.
- **Multiline paste** — pasting multi-line text into the prompt is handled
  gracefully with continuation markers (`...`).
- **Graceful interruption** — Ctrl-C cancels the running agent loop, preserving
  partial tool output in conversation context so the model knows what happened.
- **Readline history** — persisted across sessions in `~/.axio_repl_history`.
- **Single-prompt mode** — pass a prompt as argument for scripting and non-interactive use.

## Install

```bash
uv tool install "axio-repl[local,mcp,openai]"
```

The base REPL does not install any provider. Choose `openai`, `google`,
`anthropic`, `codex`, or `all` extras, or install a provider package explicitly:

```bash
uv tool install axio-repl --with axio-transport-anthropic
uv tool install axio-repl --with axio-transport-google
```

Or within the monorepo workspace:

```bash
uv run axio-repl
```

The base installation starts without tools or transports. It accepts commands;
requests explain how to configure a transport. Single-prompt mode exits with an
error when no transport is configured. Install `local` for local tools,
AGENTS.md, and local skills; install `mcp` for MCP connections and configuration
commands. Either extra works independently. `all` includes both tool extras and
all transport extras.

Installed third-party tools and instruction providers are discovered through
`axio.tools` and `axio.tools.providers`. Providers can implement
`refresh(*, config=None)` to refresh their tools and instructions from the REPL.
Both `/skills reload` and `/mcp reload` refresh providers that support this
capability; `/mcp reload` also rereads MCP configuration. Startup failures are
reported per provider, and independent providers remain available.

## Usage

```bash
# Interactive REPL (auto-detects transport from API keys)
axio-repl

# Single prompt (non-interactive)
axio-repl "list the files in this project"

# Explicit transport and model
axio-repl --transport anthropic --model claude-sonnet-4-20250514

# Google Gemini
axio-repl --transport google --model gemini-3.1-pro-preview

# Custom temperature and iteration limit
axio-repl --temperature 0.5 --max-iterations 100
```

`--model` requires an exact model ID from the selected transport's catalog.
An unknown ID exits with an error and lists the available models. This applies
to every transport. An unsuccessful `/model <query>` keeps the current model.

## Project instructions

The REPL preserves its starting working directory. It selects the project root
from `--project-root`, the nearest ancestor with a `.git` file or directory,
or the working directory when no Git marker is found. Automatic Git discovery
stops before the home directory or filesystem root. Use `--project-root` to
select either explicitly; the selected root must contain the working directory.

At startup, instructions load in this order:

1. `$XDG_CONFIG_HOME/axio/AGENTS.md`, or `~/.config/axio/AGENTS.md` when
   `XDG_CONFIG_HOME` is empty, unset, or relative.
2. `<project-root>/AGENTS.md` and each exact `AGENTS.md` down to the working directory.
3. Files selected with repeated `--instructions-file PATH`, in argument order.

Each block identifies its source and scope. Project instructions apply to their
directory and descendants. Deeper scopes override conflicting broader instructions;
other rules supplement them. Global and explicit files apply throughout the session.
Within the same scope, later entries take precedence. Explicit user requests take
precedence over file instructions. `/model` preserves the current scoped snapshot.

```bash
axio-repl --project-root /work/repo
axio-repl --workspace-root /work/another-repo
axio-repl --instructions-explain services/api/main.py
axio-repl --isolated --instructions-file ./session-rules.md "review this code"
```

Additional workspace roots must be independent: nested roots are rejected and
canonical duplicates are removed. Startup loads only the chain to the working
directory. `--instructions-explain PATH` inspects another file's chain without
starting a model, opening network sessions, or changing REPL history. A nonexistent
target is treated as a file. Sibling instructions do not enter that chain.

Paths are resolved through symlinks. An external target uses its registered root,
if any, and does not inherit the original project's instructions. A project
`AGENTS.md` symlink outside its root is an error. Explicit instruction files may
reside outside the project. Each canonical source is read and counted once, while
all applicable scopes are retained.

The default total limit is 131072 UTF-8 bytes. Set `--instructions-max-bytes N`
to change it. Invalid UTF-8, unreadable files, missing explicit files, and size
overflow stop startup with a diagnostic. Text is never silently truncated.
`--isolated` disables automatic instruction loading but retains explicit files.

The REPL refreshes known instructions before each model request. Before file
operations, it loads instructions for their paths. Recursive search checks all
candidate paths before reading contents. New or changed instructions defer the
whole tool batch so the model can review them before deciding which calls to retry.
Instruction-file edits run separately from other calls in their batch; the next
model request receives the updated rules. Each subtree retains its own scope.
Shell checks cover only its declared `cwd`; shell commands and MCP servers can
access other paths internally. These checks do not provide a filesystem sandbox.

## MCP servers

The REPL automatically connects to enabled servers from the project's `.mcp.json`.
It does not ask for confirmation. `/mcp` shows each server's state and available
tool count. `/mcp reload` rereads configuration and reconnects the servers.
A failed server has zero tools; other servers can still connect.
Each server has one initialization deadline for the connection, handshake, and
catalog discovery. The default is 30 seconds; set `"timeout": 60` in that server's
configuration to change it. Progress and timeout messages identify the server and
the limit. After a timeout, the connection closes and `/mcp reload` retries it.
The same setting bounds tool and resource requests. An interrupted request marks
the connection unavailable until `/mcp reload` reconnects it.
Stopping a stdio process can take up to five additional seconds before it is killed.

```json
{
  "mcpServers": {
    "local-tools": {
      "type": "stdio",
      "command": "my-mcp-server",
      "args": ["--read-only"]
    },
    "company-tools": {
      "type": "http",
      "url": "https://tools.example.org/mcp"
    }
  }
}
```

`type` can be omitted: `command` selects stdio, and `url` selects HTTP.

Configuration priority, from lowest to highest:

1. `${XDG_CONFIG_HOME:-$HOME/.config}/axio/mcp.json`.
2. `<project-root>/.mcp.json`.
3. `<project-root>/.axio/mcp.local.json`.
4. Each `--mcp-config PATH`, in command-line order.

Entries replace earlier entries with the same server name in full.
`{"enabled": false}` disables an inherited server. Invalid configuration reports
its source and server; it does not fall back to an earlier entry.
The REPL uses the selected project root, including when started in a subdirectory.
It does not search each ancestor for MCP files. Additional workspace roots have
independent configurations; their tool names use `w1__`, `w2__`, and so on.
Within one root, tool names use `<server>__<tool>`.

`--isolated` disables automatic MCP and instruction loading. Explicit
`--mcp-config` files still apply. Configured servers connect in both interactive
and single-prompt mode. Connections close on exit; `/model` preserves MCP tools.

Stdio commands run without a shell. Relative command paths and `cwd` use the
configuration file's directory; that directory is also the default `cwd`.
A command without `/` uses `PATH`. Arguments remain literal.
Children inherit only `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`,
`LC_CTYPE`, `TMPDIR`, `TMP`, and `TEMP`. `env` adds or overrides values.
`envFrom` maps child variable names to host variable names; `headersFrom` does
the same for HTTP header names. Missing variables produce an error for that server.
For example, `"headersFrom": {"Authorization": "MCP_AUTHORIZATION"}` reads the
complete authorization header from the environment. HTTP `headers` accepts
literal values. Keep credentials out of version control.

## Explain configuration and check MCP

```bash
axio-repl --why
# Equivalent long form:
axio-repl --config-explain
```

Both commands show instruction sources and scopes, MCP configuration layers,
whole-entry overrides, and effective working directories. They then connect to
each enabled MCP server and print its state and tool count, as `/mcp` does.
Disabled servers stay disabled. A failed connection does not stop other servers
from connecting. All connections close before the command exits.

These commands run before model selection and do not require LLM credentials.
They perform MCP initialization and discovery, including the supported skill
catalog, without calling tools. Stdio servers start as subprocesses; HTTP servers
receive network requests. `--isolated` and `--mcp-config` select configurations
as they do for a normal session. Values in commands, arguments, environment
entries, headers, and URL paths and queries are redacted. Source paths, server
names, and endpoint origins remain visible.

## Import MCP configuration

Import is explicit. The REPL does not automatically read another client's files.
Preview the selected source scope and destination first:

```bash
axio-repl mcp import --from cursor --scope project --target private
axio-repl mcp import --from cursor --scope project --target private --write
```

`--write` copies entries and preserves the source file. Conflicting server names
also require `--replace`; replacement applies to the whole entry. Other destination
entries remain unchanged. `--server NAME` selects one server and can be repeated.
Import preview and writing do not connect to servers or start a model.

| Source | `--scope` | Location |
|---|---|---|
| Claude Code | `user` | `~/.claude.json`, top-level `mcpServers` |
| Claude Code | `local` | `~/.claude.json`, `projects[canonical project root].mcpServers` |
| Claude Code | `project` | `<project-root>/.mcp.json` |
| Cursor | `user` | `~/.cursor/mcp.json` |
| Cursor | `project` | `<project-root>/.cursor/mcp.json` |

See the [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp)
and [Cursor MCP documentation](https://cursor.com/docs/mcp) for source formats.
Source scope is explicit; a path under the home directory does not imply user scope.

`--target user` writes the global Axio `mcp.json`; `project` writes `.mcp.json`;
`private` writes `.axio/mcp.local.json`. Add the private path to `.gitignore`.
Destination files use mode `0600`. The destination must differ from the source.
If a stdio entry has no `cwd`, supply `--server-cwd PATH`. Relative executable
paths use that working directory; arguments remain literal after copying.

The importer supports stdio and HTTP. Exact Claude `${VAR}` and Cursor
`${env:VAR}` values in `env` or `headers` become `envFrom` or `headersFrom`
references. It rejects embedded interpolation, fallback values, other templates,
SSE, `envFile`, and client-specific authentication settings with a diagnostic.
Literal environment values, headers, and URL queries require a private or user
destination. Arbitrary secrets inside arguments or paths cannot be identified
automatically; inspect the source before copying it into a shared project file.

## MCP resources

The model sees each server's resources and URI templates, including resources
that duplicate skills. `<server>__list_resources` refreshes this catalog;
`<server>__read_resource` loads an exact URI from that server. Contents return as
MCP JSON with text or base64 blobs and MIME types. Template parameters let the
model construct additional resource URIs.

Axio sends URIs unchanged. It does not replace `skill://` with
`mcp-extensions://` or resolve these URIs as local paths. `/model` preserves
resource access, and `/mcp reload` refreshes catalogs and connections.

## MCP skills

`/skills` lists the skills published by connected MCP servers, including each
server name, skill URI, and description. Ask the agent to use a listed skill,
for example: `Use workflows:planning to plan this change`.

The model receives the catalog at startup. It reads `SKILL.md` and supporting
text files only when needed, through `<server>__read_skill`. Each read checks
the file's UTF-8 size and SHA-256 digest against the published manifest.
Files stay on the MCP server; relative references do not use local paths.
Identical skill names remain separate by server and URI.

`/model` preserves the catalog. `/mcp reload` refreshes it, including removed
skills. Single-prompt mode uses the same discovery and reading tools.
An invalid catalog reports an error without removing the server's ordinary tools.
An interrupted connection requires `/mcp reload`.

This implementation supports the Skills compatibility resources under
`mcp-extensions://io.modelcontextprotocol/skills`, over stdio and HTTP.
It uses the published `skills/list` resource, with pagination, or the static
`skills` resource. Native extension RPC methods, unlisted skills, binary files,
and manifests marked `dynamic` are not supported. Skills must publish a complete
manifest with at most 512 files and 16 MiB of total content.

Skill reads use the exact manifest URI. A rejected URI produces an error without
trying another scheme. Other advertised resource URIs remain available through
`<server>__read_resource`.

## Local skills

The REPL also discovers [Agent Skills](https://agentskills.io/specification):
directories with a `SKILL.md` file containing YAML `name` and `description` fields.
It searches these user locations automatically:

- `${XDG_CONFIG_HOME:-$HOME/.config}/axio/skills`
- `~/.agents/skills`
- `~/.claude/skills`
- `~/.codex/skills`
- `~/.cursor/skills`
- `${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills`

The format does not prescribe an installation path. `.agents/skills` is a
[cross-client convention](https://agentskills.io/integrate-skills).
Search stays within these directories. Add another directory with repeated
`--skills-dir PATH`; relative paths use the working directory, and `~` is expanded.
Project skill directories are read only when selected explicitly.

```bash
axio-repl --skills-list
axio-repl --skills-dir ./team-skills
axio-repl --isolated --skills-dir ~/.agents/skills --skills-list
```

`--skills-list` lists local skills without starting a model or connecting MCP.
Inside the REPL, `/skills` lists both local and MCP skills. `/skills reload`
refreshes available providers; `/mcp reload` also rereads MCP configuration. `--isolated`
disables automatic local discovery while retaining explicit `--skills-dir` paths.

The model receives names, descriptions, locations, and optional `compatibility`
requirements. `read_local_skill` loads instructions and supporting UTF-8 files
on demand. Relative file paths stay inside the skill's directory. Script and
asset paths use that directory; discovery does not execute scripts.
Changed frontmatter requires `/skills reload` before another read.

Canonical paths deduplicate symlinked installations. Distinct files with the
same name remain visible with their full paths. Identical `SKILL.md` content
does not produce a duplicate warning. Different content, or content that cannot
be compared, produces a warning. Invalid metadata produces a
diagnostic without hiding other skills. Nonstandard names produce a warning.
Skills that disable automatic invocation through `disable-model-invocation`
or `metadata.opencode/autoinvoke` are excluded. Other client-specific fields
do not configure Axio tools or permissions. `compatibility` describes requirements;
it does not verify installed programs or services.

Discovery reads frontmatter, limited to 64 KiB per file. For duplicate names,
it also compares full `SKILL.md` content up to 16 MiB per file. It visits at most
2000 directories, to a depth of six below each selected directory. Individual
text reads are limited to 16 MiB; oversized content fails without truncation.

## ChatGPT account

```bash
axio-repl --transport chatgpt --model gpt-6-luna
```

When the credential store is empty, the REPL asks `Sign in now? [y/N]`.
Answer `y` to authorize in the browser and continue in the same REPL process.
Over SSH, the sign-in flow prints a URL and accepts the browser redirect URL.
Declining or reaching end of input cancels startup. Piped input does not trigger
an interactive login; sign in from a terminal first.

`axio-chatgpt` remains available for account management and scripted setup:

```bash
axio-chatgpt accounts
axio-chatgpt models
```

Choose a visible model from `axio-chatgpt models`. ChatGPT requires explicit
`--transport chatgpt`; API-key auto-detection remains available for other providers.
The REPL uses the saved selection, or the only saved registration. Use
`axio-chatgpt select CLIENT_ID` to change the default for future sessions.
`--account CLIENT_ID` overrides it for one REPL session. `--credentials PATH`
overrides the default `~/.config/axio/chatgpt.auth` file, or its XDG equivalent.

The selected account stays fixed while `/model` switches among its visible models.
Use `/account` to list registrations and `/account ID` to select one and clear
history. `/login` reauthorizes the active account; `/logout` clears its tokens and history.
ChatGPT starts with `medium` reasoning effort when the model supports that level;
otherwise it leaves the effort unset.
Override it with `--thinking <effort>` or `/thinking <effort>`.
`/thinking` lists the selected model's supported levels. Unsupported values are
rejected locally. A model switch clears an effort that the new model does not support.
Use `/thinking default` to clear an explicit ChatGPT reasoning effort. Switching
to a model without declared reasoning support also clears this setting.
Background token renewal stops when the REPL closes; saved credentials remain.
Use `axio-chatgpt login --account CLIENT_ID` when reauthorization is required.
Identity-only sign-in needs ChatGPT plan consent before inference is available.
`axio-chatgpt status` reports local credential state; `axio-chatgpt logout` signs out.
For SSH sign-in, use `axio-chatgpt login --manual` and paste the browser redirect URL.

ChatGPT accepts `/thinking low` and other supported reasoning effort values.
It does not support `--temperature`, `--max-tokens`, or `--debug`.
Its account flow does not require Textual.

## REPL Commands

| Command              | Description                                    |
|----------------------|------------------------------------------------|
| `/mcp` | Show MCP server states and tool counts |
| `/mcp reload` | Reload configuration and reconnect MCP servers |
| `/skills` | List local and MCP skills with their sources and descriptions |
| `/skills reload` | Refresh available tool and instruction providers |
| `/help`              | Show available tools and commands               |
| `/model`             | Show current model and list available models    |
| `/model <query>`     | Switch to a model matching the query            |
| `/quit` `/exit` `/q` | Exit the REPL                                   |

## Tools

| Tool            | Description                                                    |
|-----------------|----------------------------------------------------------------|
| `read_file`     | Read file contents; images and videos returned as vision blocks |
| `write_file`    | Create or overwrite files                                       |
| `patch_file`    | Replace line ranges in files (1-indexed, inclusive)              |
| `list_files`    | List directory contents                                         |
| `search_files`  | Text/regex search across files                                  |
| `shell`         | Run shell commands with streaming output and process-group cleanup |
| `generate_image` | Generate images via Gemini Nano Banana (Google transport only) |
| `generate_video` | Generate videos via Veo (Google transport only)                |

## Antigravity subscription

```bash
axio-repl --transport antigravity
```

For local development, run `uv sync --all-packages`, then
`uv run axio-repl --transport antigravity`. A separately installed `axio-repl`
uses its own package registry and can still report an unknown transport.

The first terminal run starts Google browser sign-in. Saved sessions use
`${XDG_CONFIG_HOME:-$HOME/.config}/axio/antigravity.auth`. `--account ID` pins an
account; `--credentials PATH` selects another credential file. For SSH, use
`axio-antigravity login --manual` and paste the full browser redirect URL.

`/model` lists the account's live Gemini catalogue. `/account` lists saved accounts
and identifies the active account. `/account ID` opens that account and clears
conversation history. `/login` reauthorizes the active account; `/logout` clears
its local tokens and conversation history. Account changes outside the REPL affect
future sessions; the active session remains pinned. `/model` preserves its account.

`axio-antigravity accounts`, `select --account ID`, `status`, `models`, and `logout`
work outside the REPL. Use these commands to recover from startup discovery errors.
`logout --revoke` also requests Google grant revocation. `status` checks the OAuth
token, not subscription entitlement. `--debug` is unavailable for OAuth sessions.

Antigravity supports Gemini completion, reasoning, and function tools. It does not
expose Claude, media generation, or realtime sessions. Quota errors never trigger
an automatic account or payment change. See the Google transport guide for the
internal API limitations and credential storage details.

## Transports

Transports are discovered via the `axio.transport` entry point group.
The REPL checks installed provider metadata and picks the first eligible transport
with credentials. API-key providers declare these environment variables:

| Transport        | Env Variable                   | Package                    |
|------------------|--------------------------------|----------------------------|
| `google`         | `GEMINI_API_KEY`               | `axio-transport-google`    |
| `google-vertex`  | `GOOGLE_GENAI_USE_VERTEXAI`    | `axio-transport-google`    |
| `anthropic`      | `ANTHROPIC_API_KEY`            | `axio-transport-anthropic` |
| `openai`         | `OPENAI_API_KEY`               | `axio-transport-openai`    |
| `nebius`         | `NEBIUS_API_KEY`               | `axio-transport-openai`    |
| `openrouter`     | `OPENROUTER_API_KEY`           | `axio-transport-openai`    |
| `codex`          | *(API key varies)*             | `axio-transport-codex`     |

Use `--transport <name>` to force a specific transport regardless of env vars.

## Capability-Aware System Prompt

The system prompt adapts based on the selected model's declared capabilities:

- **Vision** — unlocks instructions to `read_file` images and do screenshot-based
  UI review.
- **Reasoning** — notes that extended thinking is available.
- **Image generation** — enables inline image generation guidance.
- **Video** — enables `read_file` for video content.
- **Tool use** — gates all tool-related rules (edit workflow, testing, verification).

This means switching from a text-only model to a vision model mid-session
(via `/model`) automatically updates what the agent is instructed to do.

## Architecture

```
┌─────────────┐     ┌───────────┐     ┌──────────────────┐
│  axio-repl  │────▶│   axio    │────▶│    transport      │
│  (terminal  │     │  (agent   │     │  (anthropic /     │
│   UI, I/O)  │     │   loop)   │     │   google / openai │
└─────────────┘     └───────────┘     │   / nebius / ...)  │
                          │           └──────────────────┘
                    ┌─────┴─────┐
                    │ tools     │
                    │ (local fs │
                    │  + shell) │
                    └───────────┘
```

- **axio-repl** owns the terminal UI: readline, event rendering, REPL commands.
- **axio** runs the agent loop: dispatch tools, manage conversation context,
  handle cancellation.
- **transports** handle LLM communication — message conversion, streaming,
  model registries.
- **axio-tools-local** provides the file and shell tools.
