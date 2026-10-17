# axio-repl

`axio-repl` is a terminal coding assistant that runs an Axio agent with file and
shell tools. It auto-detects your LLM backend from environment variables and
streams every token and tool call directly to the terminal.

## Install

```bash
uv tool install "axio-repl[local,mcp,openai]"
```

Add optional transports:

```bash
# Anthropic Claude
uv tool install "axio-repl[local]" --with axio-transport-anthropic

# Google Gemini
uv tool install "axio-repl[local]" --with axio-transport-google
```

Within the monorepo workspace:

```bash
uv run axio-repl
```

The base REPL has no mandatory provider dependencies. Install the desired
provider in the same environment, or use the `openai`, `google`, `anthropic`,
`codex`, or `all` extra. Install `local` for file tools, scoped instructions,
and local skills; install `mcp` for remote tools and skills. These extras are
independent. Without a transport, the REPL accepts commands and explains why
requests cannot run. Single-prompt mode returns an error. Third-party providers use the same
[entry-point contract](writing-transports.md).

## Start the REPL

```bash
axio-repl
```

The REPL checks installed provider metadata and picks the first eligible transport
with credentials. API-key providers declare these environment variables:

| Transport | Env Variable | Package |
|---|---|---|
| `google` | `GEMINI_API_KEY` | `axio-transport-google` |
| `google-vertex` | `GOOGLE_GENAI_USE_VERTEXAI` | `axio-transport-google` |
| `anthropic` | `ANTHROPIC_API_KEY` | `axio-transport-anthropic` |
| `openai` | `OPENAI_API_KEY` | `axio-transport-openai` |
| `nebius` | `NEBIUS_API_KEY` | `axio-transport-openai` |
| `openrouter` | `OPENROUTER_API_KEY` | `axio-transport-openai` |

Override with `--transport <name>`. `--model` is looked up in that transport's registry by
exact id. An unknown id exits with an error and lists available models for that transport:

```bash
axio-repl --transport anthropic --model claude-sonnet-4-6
axio-repl --transport google --model gemini-3.1-pro-preview
```

## ChatGPT subscription

Start the REPL with a ChatGPT model:

```bash
axio-repl --transport chatgpt --model gpt-6-luna
```

If the credential store is empty, the REPL asks whether to sign in. Answer `y`
to open browser authorization and continue in the same process. Over SSH, the
sign-in flow prints a URL and accepts the pasted redirect URL. Declining or EOF
cancels startup. Piped input does not prompt; sign in from a terminal first.
Existing registrations and explicit `--account` choices use their saved state.

The account commands come with `axio-transport-openai`, an optional provider package.
Use `axio-chatgpt accounts` and `axio-chatgpt select CLIENT_ID` to change the
default account. `--account CLIENT_ID` overrides it for one REPL session.
`--credentials PATH` overrides the default XDG credential file. Reauthorize with
`axio-chatgpt login --account CLIENT_ID`; use `login --manual` for SSH sign-in.

ChatGPT requires explicit transport selection and plan consent. `/account` lists
registrations; `/account ID` selects one and clears conversation history. `/login`
reauthorizes the active registration; `/logout` signs it out and clears history.
`/model` changes the model within the active account's visible catalog. Closing the REPL stops renewal and retains credentials.
`axio-chatgpt status` checks local state without network access.
`axio-chatgpt logout` clears credentials and attempts server revocation.

ChatGPT supports reasoning effort through `--thinking low` or `/thinking low`.
It rejects `--temperature`, `--max-tokens`, and `--debug`.

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

## Single-prompt mode

Pass a prompt as an argument for non-interactive use:

```bash
axio-repl "list the files in this project"
axio-repl --transport openai "write tests for src/auth.py"
```

## MCP servers

The REPL automatically connects to enabled servers from the project's `.mcp.json`.
It does not ask for confirmation. `/mcp` shows each server's state and available
tool count. `/mcp reload` rereads configuration and reconnects the servers.
A failed server has zero tools; other servers can still connect.

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

The model receives the resource and URI template catalogs from each connected
MCP server. It can refresh them with `<server>__list_resources` and read a
resource with `<server>__read_resource`. Resource contents load only when requested.
Reads return MCP contents as JSON, preserving text, base64 blobs, and MIME types.

Resource URIs go to their originating server unchanged. Axio does not replace
`skill://` with `mcp-extensions://` or treat either scheme as a local path.
Resources remain visible even when they duplicate skills. A resource template
lets the model construct a URI with the required parameters before reading it.
`/model` preserves access; `/mcp reload` refreshes the catalogs and connections.

## MCP skills

Run `/skills` to list skills from connected MCP servers. Each entry shows its
server, URI, and description. Ask the agent to use a listed skill, for example:
`Use workflows:planning to plan this change`.

The model receives only the catalog at startup. It uses `<server>__read_skill`
to read `SKILL.md` and supporting text files when needed. Relative paths refer
to the skill directory on that server. Each read verifies the UTF-8 byte count
and SHA-256 digest from the manifest. Skill names remain scoped to their server;
the URI distinguishes duplicate names within one server.

Skills work in single-prompt mode and survive `/model` changes.
`/mcp reload` refreshes the catalog. Invalid catalogs produce diagnostics while
ordinary tools remain available. Interrupted connections require a reload.

Discovery uses the Skills compatibility resource bridge under
`mcp-extensions://io.modelcontextprotocol/skills`, over stdio or HTTP.
The client reads the paginated `skills/list` resource or the static `skills`
resource. Native extension RPC methods, unlisted skills, binary files, and
`dynamic` manifests are not supported. A manifest can contain up to 512 files
with a combined size of 16 MiB.

Skill reads also preserve the manifest's URI. If the server rejects that URI,
Axio reports the error without trying a different scheme. The model can still
read other URIs advertised in the independent resource catalog.

## Local skills

The REPL reads the [Agent Skills format](https://agentskills.io/specification):
`SKILL.md` with YAML `name` and `description` fields, plus optional supporting files.
It searches these directories in the user's home and XDG configuration:

| Directory | Source |
|---|---|
| `${XDG_CONFIG_HOME:-$HOME/.config}/axio/skills` | Axio |
| `~/.agents/skills` | Shared skills |
| `~/.claude/skills` | Claude-compatible skills |
| `~/.codex/skills` | Codex-compatible skills |
| `~/.cursor/skills` | Cursor-compatible skills |
| `${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills` | OpenCode-compatible skills |

Installation paths are conventions, not part of the file format.
Use `--skills-dir PATH` to add directories, including project skills.
Repeat the flag for multiple locations. Relative paths use the working directory;
`~` expands to the user's home. `--isolated` retains only explicit locations.

```bash
axio-repl --skills-list
axio-repl --skills-dir ./team-skills
axio-repl --isolated --skills-dir ~/.agents/skills --skills-list
```

`--skills-list` works without a model or MCP connection. `/skills` combines local
and remote catalogs. `/skills reload` refreshes local skills; `/mcp reload`
refreshes remote skills. Both catalogs survive model changes.

Discovery reads metadata and compares full `SKILL.md` content for duplicate
names, up to 16 MiB per file. The model uses `read_local_skill` to read the
instructions and supporting UTF-8 files when needed. Relative references resolve
inside the skill directory. Scripts are not executed during discovery.
Changes to frontmatter require `/skills reload` before another read.

Symlink aliases of one canonical file appear once. Same-named skills at distinct
paths remain separate. Identical `SKILL.md` content produces no duplicate warning.
Different content, or content that cannot be compared, produces a warning.
Invalid files are skipped with a
diagnostic; nonstandard names produce a warning. `compatibility` is shown as
environment requirements, not a guarantee that those requirements are installed.
Client-specific fields do not grant permissions or change execution settings.
Skills with `disable-model-invocation: true` or `metadata.opencode/autoinvoke: false`
are excluded from automatic discovery.

Search stays within selected directories, with a depth limit of six and a total
limit of 2000 visited directories. Frontmatter is limited to 64 KiB.
Text reads above 16 MiB fail without truncating content.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--transport` | auto | Transport name (see table above) |
| `--model` | transport default | Model name |
| `--account` | saved selection | ChatGPT registration for this session |
| `--credentials` | XDG default | ChatGPT credential file |
| `--project-root` | Git root or cwd | Explicit project root containing cwd |
| `--workspace-root` | none | Additional independent root; repeatable |
| `--instructions-file` | none | Explicit session instructions; repeatable |
| `--instructions-max-bytes` | 131072 | Total UTF-8 bytes across unique instruction sources |
| `--instructions-explain PATH` | none | Print applicable sources and scopes, then exit without a model |
| `--isolated` | off | Disable automatic instructions, MCP, and local skills; explicit paths still apply |
| `--mcp-config` | none | Explicit MCP configuration; repeatable |
| `--config-explain`, `--why` | off | Explain configuration, check MCP connections, and exit without a model |
| `--skills-dir` | none | Additional local skills directory; repeatable |
| `--skills-list` | off | List local skills without a model or MCP connection |
| `--temperature` | transport default | Sampling temperature |
| `--thinking` | `medium` for ChatGPT models that support it; otherwise provider default | Thinking level, reasoning effort, or a token budget |
| `--max-tokens` | transport default | Max output tokens |
| `--max-iterations` | 30 | Max agent iterations |
| `--debug` | off | Log raw request/response bodies |

## REPL commands

| Command | Description |
|---|---|
| `/model` | Show current model and list available models |
| `/model <query>` | Switch to a model matching the query |
| `/thinking [level]` | Show or set thinking level/budget; `default` clears ChatGPT reasoning effort |
| `/temperature [val]` | Show or set sampling temperature |
| `/max-tokens [val]` | Show or set max output tokens |
| `/iterations [val]` | Show or set max agent iterations |
| `/debug [on\|off]` | Toggle request/response debug logging |
| `/mcp` | Show MCP server states and tool counts |
| `/mcp reload` | Reload configuration and reconnect MCP servers |
| `/skills` | List local and MCP skills with their sources and descriptions |
| `/skills reload` | Refresh local skill discovery |
| `/help` | List all tools and commands |
| `/quit`, `/exit`, `/q` | Exit the REPL |

ChatGPT starts with `medium` reasoning effort when the model supports that level;
otherwise it leaves effort unset. Use `--thinking <effort>` to override it at startup.
`/thinking` shows the levels supported by the selected model. Unsupported values
are rejected locally, including startup arguments. Switching models clears a
saved effort if the new model does not support it. `/thinking default` also clears it manually.
`/thinking none` is a provider effort value, not a reset.

## Tools

| Tool | Description |
|---|---|
| `read_file` | Read file contents; images and videos are returned as vision blocks |
| `write_file` | Create or overwrite files |
| `patch_file` | Replace line ranges (1-indexed, inclusive) |
| `list_files` | List directory contents |
| `search_files` | Text or regex search across files |
| `shell` | Run shell commands with streaming stdout/stderr |
| `generate_image` | Generate images via Gemini (Google transport only) |
| `generate_video` | Generate videos via Veo (Google transport only) |

## AGENTS.md

The REPL loads global and project instructions at startup. It finds the project
root from `--project-root`, the nearest ancestor with a `.git` file or directory
(including Git worktrees), or the working directory if neither applies. The
working directory does not change. Automatic Git discovery stops before the
home directory and filesystem root; either can be selected explicitly.

The instruction chain starts at `$XDG_CONFIG_HOME/axio/AGENTS.md`. If
`XDG_CONFIG_HOME` is empty, unset, or relative, it uses `~/.config/axio/AGENTS.md`.
Next come `<root>/AGENTS.md` and each exact `AGENTS.md` down to the working
directory. For example:

```markdown
# My project

- Always run `make test` after editing Python files
- The main entry point is `src/app.py`
- Use the `dev` branch for all changes
```

Each block carries its source path and scope. Deeper directory scopes override
conflicting broader rules; other rules supplement them. Explicit user requests
take precedence. `/model` preserves the current scoped snapshot.

```bash
# Explain another file's chain without a model, API key, or network connection.
axio-repl --instructions-explain services/api/main.py

# Add an independent workspace. This does not load its instructions at startup.
axio-repl --workspace-root /work/other-repo

# Load only explicitly selected session instructions.
axio-repl --isolated --instructions-file ./session-rules.md
```

`--project-root` must contain the working directory. Additional roots must not
overlap. Canonical duplicates are removed. Relative arguments resolve from the
starting working directory. A nonexistent explain target is treated as a file.
Explicit instruction files can have any name and apply throughout the session.
Repeated `--instructions-file` arguments preserve their order; later entries
override earlier entries within the same scope.

Symlinks resolve before choosing a target's project. Targets outside registered
roots receive only global and explicit instructions. A project `AGENTS.md` that
resolves outside its root is an error. Canonical sources are read once, with all
applicable scopes retained. The loader does not read sibling instruction files.

Instructions have a total limit of 131072 UTF-8 bytes, configurable through
`--instructions-max-bytes N`. Unreadable files, invalid UTF-8, missing explicit
files, and overflow stop startup with a source-specific error. No text is silently
truncated. Missing automatic instruction files are allowed.

The REPL refreshes known instruction files before each model request and checks
all declared paths before dispatching a tool batch. If it finds new or changed
instructions after the model's decision, it defers the entire batch. Each call
receives a result that asks the model to review the current scoped instructions
and decide which operations to retry. No handler in that batch starts first.

This check covers `read_file`, `write_file`, `patch_file`, `list_files`, and
`search_files`. Recursive search loads instructions for all candidate files
before reading their contents. Listing a directory does not load instructions
from its child directories. Files returned through symlinks use their canonical
scope. Loaded scopes remain separate across directories and workspace roots.

An instruction-file edit runs separately from other operations in its batch.
Only its first instruction edit runs; other calls are deferred. The next model
request receives the updated instructions. Changed and deleted files replace
the previous snapshot. `/model` preserves the current snapshot.

For `shell`, the REPL checks only the declared `cwd` chain. Shell commands and
MCP servers can access other paths internally. This mechanism does not sandbox
filesystem access or prevent concurrent external file changes.

## Capability-aware system prompt

The system prompt adapts to the selected model's capabilities. Switching models
with `/model` recalculates capabilities and rewrites the prompt automatically:

- **Vision** - `read_file` on images (PNG, JPG, GIF, WebP) returns pixel data.
  Screenshot-based UI review loops are unlocked.
- **Reasoning** - Extended thinking (chain-of-thought) is available.
- **Image generation** - Inline image generation via `generate_image`.
- **Video** - `read_file` on video files returns vision blocks.

## Multiline input

Paste multi-line text directly into the prompt. The REPL detects continuation
lines and joins them before sending:

```
You> Refactor this function:
...   def old(x):
...     return x+1
```

## Interrupting the agent

Press **Ctrl-C** to cancel the running agent loop. Partial tool output (stdout
already captured, files already written) is preserved in conversation context
so the model sees what happened and can resume cleanly.
