# axio-repl

Interactive REPL coding assistant powered by the [axio](../axio) agent framework.
Works with any LLM backend via pluggable transports — bring your own API key.

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
- **AGENTS.md** — workspace-level instructions loaded into the system prompt from
  an `AGENTS.md` file in the working directory.
- **Multiline paste** — pasting multi-line text into the prompt is handled
  gracefully with continuation markers (`...`).
- **Graceful interruption** — Ctrl-C cancels the running agent loop, preserving
  partial tool output in conversation context so the model knows what happened.
- **Readline history** — persisted across sessions in `~/.axio_repl_history`.
- **Single-prompt mode** — pass a prompt as argument for scripting and non-interactive use.

## Install

```bash
uv tool install axio-repl
```

To add optional transports:

```bash
uv tool install axio-repl --with axio-transport-anthropic
uv tool install axio-repl --with axio-transport-google
```

Or within the monorepo workspace:

```bash
uv run axio-repl
```

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

## REPL Commands

| Command              | Description                                    |
|----------------------|------------------------------------------------|
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

## Transports

Transports are discovered via the `axio.transport` entry point group.
The REPL picks the first transport whose required environment variable is set:

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
