# Claude Code through Codex with a ChatGPT subscription

This configuration runs Claude Code against agent-relay's Anthropic-compatible endpoint, then executes each request with the Codex CLI using an existing ChatGPT subscription login.

It completes the repository's previously reserved `--backend codex` path. The original backend remains unchanged: API-compatible clients can still route requests through `claude -p`. This note covers the inverse composition where Claude Code is the client and `codex exec` is the backend.

```text
Claude Code
  -> ANTHROPIC_BASE_URL=http://127.0.0.1:18082
  -> agent-relay /v1/messages
  -> codex exec --json --ephemeral
  -> ChatGPT subscription
  -> gpt-5.6-sol
```

The Codex backend removes `OPENAI_API_KEY` and `CODEX_API_KEY` from the child process environment. This is intentional: authentication comes from `codex login`, not usage-billed API credentials.

## Prerequisites

- Python 3.10 or newer and `uv`
- Claude Code available as `claude`
- A current Codex CLI available as `codex`
- A ChatGPT account with access to the configured model
- Linux with systemd user services, or macOS with launchd

The configuration was verified with Claude Code 2.1.207 and Codex CLI 0.144.1. Update Codex before troubleshooting model availability:

```bash
npm install -g @openai/codex@latest
codex --version
```

## Authenticate Codex with ChatGPT

Sign in interactively and verify the login:

```bash
codex login
codex login status
```

Do not run `codex login --with-api-key` for this configuration. A shell-level `OPENAI_API_KEY` is safe to leave defined because agent-relay removes it from Codex subprocesses, but the systemd or launchd service must not persist that key.

Verify the model directly before introducing the relay:

```bash
codex exec --json --ephemeral \
  --model gpt-5.6-sol \
  --sandbox read-only \
  --skip-git-repo-check \
  'Reply with exactly: codex-subscription-ok'
```

The output should contain an `item.completed` event whose `agent_message` text is `codex-subscription-ok`.

## Install the relay service

From the repository that should be the Codex working directory:

```bash
uv sync
uv run agent-relay service install \
  --host 127.0.0.1 \
  --port 18082 \
  --backend codex \
  --model gpt-5.6-sol
```

The installer pins the service to the exact relay executable used during installation. On Linux it creates `~/.config/systemd/user/agent-relay.service` and uses the installation directory as `WorkingDirectory`. The default Codex sandbox is `workspace-write`, so install from the project directory Codex should be allowed to modify. Override it with `AGENT_RELAY_CODEX_SANDBOX=read-only` when file writes are not required.

Check the service:

```bash
systemctl --user status agent-relay.service
curl -sS http://127.0.0.1:18082/health
curl -sS http://127.0.0.1:18082/v1/models
```

Expected health fields include:

```json
{
  "status": "ok",
  "backend": "codex",
  "cli": "codex",
  "routing_default_model": "gpt-5.6-sol"
}
```

Confirm the unit does not contain an API key:

```bash
systemctl --user show agent-relay.service -p ExecStart -p Environment
```

## Launch Claude Code

Use the included launcher from any working directory. The launcher enables Claude Code's `--dangerously-skip-permissions` mode by default and sends its current directory in the `X-Agent-Relay-Cwd` header:

```bash
scripts/claude-codex
```

To install the launcher on `PATH`:

```bash
ln -s "$PWD/scripts/claude-codex" "$HOME/.local/bin/claude-codex"
claude-codex
```

The relay validates that directory and uses it as the cwd for the request's Codex subprocess. The service `WorkingDirectory` and `CLAUDE_RELAY_CWD` remain fallbacks for clients that do not send the header.

Codex emits completed assistant messages rather than token-sized text events. The relay splits these into 256-character SSE deltas so terminal clients can render and wrap long responses normally. Set `AGENT_RELAY_STREAM_CHUNK_SIZE` before starting the service to tune that limit.

The equivalent environment is:

```bash
unset ANTHROPIC_API_KEY
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_BASE_URL="http://127.0.0.1:18082"
export ANTHROPIC_MODEL="gpt-5.6-sol"
export ANTHROPIC_CUSTOM_MODEL_OPTION="gpt-5.6-sol"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="GPT-5.6 SOL"
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="1"
export ANTHROPIC_CUSTOM_HEADERS="X-Agent-Relay-Cwd: $PWD"
claude --dangerously-skip-permissions
```

Claude Code uses its existing Claude login for client authentication, while `ANTHROPIC_BASE_URL` routes inference through agent-relay. The relay does not validate or forward the Claude authorization header to OpenAI. Keeping `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` unset also avoids Claude Code's custom-auth connector warning. Claude model aliases such as `sonnet` and full `claude-*` IDs are mapped to the configured Codex default model.

## End-to-end verification

Test the Anthropic-compatible endpoint directly:

```bash
curl -sS http://127.0.0.1:18082/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer unused' \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 64,
    "messages": [{
      "role": "user",
      "content": "Reply with exactly: relay-ok"
    }]
  }'
```

The response's `model` should be `gpt-5.6-sol` and its text should be `relay-ok`.

Then verify the complete Claude Code path:

```bash
claude-codex -p --model gpt-5.6-sol \
  --output-format text \
  'Reply with exactly: claude-codex-ok'
```

## Troubleshooting

### Model requires a newer Codex version

Upgrade Codex, then restart the relay:

```bash
npm install -g @openai/codex@latest
agent-relay service restart
```

### Service repeatedly exits with unrecognized arguments

The unit may point at an older global `agent-relay`. Re-run installation with the intended executable and inspect `ExecStart`:

```bash
uv run agent-relay service install --backend codex --model gpt-5.6-sol
systemctl --user show agent-relay.service -p ExecStart
```

### Claude Code warns that connectors are disabled

Unset both custom Anthropic credential variables and start a new session:

```bash
unset ANTHROPIC_API_KEY
unset ANTHROPIC_AUTH_TOKEN
claude-codex
```

The launcher now does this automatically. The relay supports text requests, but it does not proxy Claude.ai connectors or Anthropic tool-use blocks.

### Inspect relay failures

```bash
journalctl --user -u agent-relay.service -n 100 --no-pager
curl -sS http://127.0.0.1:18082/health
```

The relay binds to `127.0.0.1` in this setup. Do not expose it to an untrusted network: the endpoint itself has no authentication, the launcher bypasses Claude Code permission prompts, and the backend can run Codex tools inside its configured working directory and sandbox.
