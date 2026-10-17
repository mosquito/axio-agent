# axio-transport-google

Google GenAI (Gemini) transport for Axio with image and video support.

## API reference

Request/response types are generated from the Vertex AI discovery documents:

- **v1:** https://aiplatform.googleapis.com/$discovery/rest?version=v1
- **v1beta1:** https://aiplatform.googleapis.com/$discovery/rest?version=v1beta1

To regenerate TypedDict definitions after an API update:

```bash
python scripts/generate_types.py [--version v1beta1]
```

This produces `src/axio_transport_google/_generated_types.py`, which the transport
uses for type annotations. Conformance tests in `tests/test_generated_types.py`
validate that payload builders match the discovery schema.

## Antigravity OAuth

`AntigravityTransport` uses a Google account's Antigravity access. It is a separate
transport from `google` (Developer API key) and `google-vertex` (Cloud credentials).
It supports Gemini completion, streaming, reasoning, and function tools. It does
not expose Claude, image/video generation, or Gemini Live.

```bash
axio-antigravity login
axio-antigravity models
axio-repl --transport antigravity
```

From this workspace, use `uv run axio-repl --transport antigravity`. Run
`uv sync --all-packages` after adding or changing package entry points. An older
`uv tool install` environment does not use this workspace's transport registry.

The REPL opens Google sign-in when no account is saved and stdin is a terminal.
For SSH, run `axio-antigravity login --manual`, open the printed URL, and paste
the full localhost redirect URL. The callback page uses the shared Axio sign-in
design and reports success after token exchange and identity verification.

Credentials use the core file store at `${XDG_CONFIG_HOME:-$HOME/.config}/axio/antigravity.auth`.
Files have mode `0600`; they contain refresh tokens and are not encrypted.
`axio-antigravity --credentials PATH …` overrides the file. Python applications
can also use `AntigravitySQLiteCredentials` or a compatible core credential store.

Account commands remain available if catalogue discovery fails:

```bash
axio-antigravity accounts
axio-antigravity select --account ACCOUNT_ID
axio-antigravity status
axio-antigravity logout
axio-antigravity logout --revoke
```

`status` checks and, if necessary, refreshes the OAuth token. It does not establish
subscription entitlement. `logout` removes local tokens; `--revoke` also requests
Google grant revocation. Active transports keep their selected account. Changing
the saved selection affects future sessions. Revocation and local logout affect
future requests from existing transports too.

Models and limits come from the account's live catalogue. `/model` uses those IDs,
which can differ from Developer API IDs and display names. Quota and permission
errors never switch to another account, model, or payment method. No Cloud project
is created automatically. If discovery reports no companion project, complete
setup in the Antigravity application first.

See `examples/antigravity_chat.py` for a Python tool loop. This uses an unofficial
internal API, which can change without notice. [Google's Antigravity terms](https://antigravity.google/terms)
restrict third-party OAuth use and permit account suspension.

### TUI

Install `axio-tui[google]` to include the Google transport and its Textual settings.
From this workspace, run `uv run axio`, then open **Configure Antigravity** in the
command palette. Saved REPL accounts are available in the same credential store.
The screen supports browser sign-in, pasted redirect URLs, account selection, and
local sign-out. **Use account** reloads the model catalogue. Settings remain
available when sign-in or model discovery fails.

TUI config stores only the credential file path and account ID. Model copies
share the selected OAuth session. Changing accounts affects new transports;
existing copies stay pinned to their original account.


## Moving Claude Vertex settings

Google transports now serve Gemini only. Claude models with an `anthropic/`
prefix no longer appear in the Google Vertex catalog. A request with an old
model ID fails with migration instructions before network access.

Install `axio-transport-anthropic[vertexai]` and choose `anthropic-vertex` in
REPL or TUI. Use the Claude model ID without `anthropic/`. Copy the previous
Vertex project and location to the new transport's settings, or set
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. Keep the existing ADC setup.
For scripts, use `axio_transport_anthropic.VertexAITransport` with those values.

TUI preserves old model and transport settings. An unavailable saved chat model
requires an explicit selection; it does not silently switch to Gemini. Google
settings remain available for Gemini. Sampling and thinking settings must be
set explicitly on the chosen Claude transport.
