# agent-relay

[![CI](https://github.com/npow/claude-relay/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/claude-relay/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentrelay-cli)](https://pypi.org/project/agentrelay-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Docs](https://img.shields.io/badge/docs-mintlify-18a34a?style=flat-square)](https://mintlify.com/npow/claude-relay)

Drop-in OpenAI **and Anthropic** API server that routes through agent CLIs (currently [Claude Code](https://docs.anthropic.com/en/docs/claude-code)).

> Compatibility note: `claude-relay` remains available as a compatibility package/command alias.

## Why

You have tools that speak the OpenAI or Anthropic API. You have Claude Code with its tools, MCP servers, and agentic capabilities. **agent-relay** bridges the two — point any compatible client at it and every request flows through `claude -p` under the hood.

- **Use Claude Code from any OpenAI or Anthropic client** — Cursor, Continue, aider, LangChain, custom scripts
- **Keep Claude Code's superpowers** — tool use, MCP servers, file access, shell execution
- **Zero config** — if `claude` works on your machine, so does this
- **Real token usage** — reports actual token counts from Claude (not zeros)
- **Token-level streaming** — uses `--include-partial-messages` for true real-time deltas

## Install

```bash
# With uv (recommended)
uvx agent-relay serve

# Or install globally
uv tool install agentrelay-cli
agent-relay serve

# Or from source
git clone https://github.com/npow/claude-relay.git
cd claude-relay
uv sync
uv run agent-relay serve
```

## Quick start

```bash
agent-relay serve
# Server starts on http://localhost:18082
```

### Run as background service (macOS)

```bash
# Install and auto-start on login
agent-relay service install
```

The installer will offer to add these to your `~/.zshrc` (or `~/.bashrc`) so every SDK and agent picks up the relay automatically:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:18082"
export OPENAI_BASE_URL="http://127.0.0.1:18082/v1"
```

```bash
# Check status
agent-relay service status

# Update
uv tool upgrade agentrelay-cli
agent-relay service restart

# Stop and remove
agent-relay service uninstall
```

Point any OpenAI-compatible client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18082/v1", api_key="unused")

# Streaming
for chunk in client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")

# Non-streaming
resp = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### Anthropic SDK

```python
import anthropic

# Just set the base URL — the SDK reads ANTHROPIC_BASE_URL automatically
# export ANTHROPIC_BASE_URL=http://localhost:18082
client = anthropic.Anthropic(base_url="http://localhost:18082")

# Streaming
with client.messages.stream(
    model="sonnet",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="")

# Non-streaming
resp = client.messages.create(
    model="sonnet",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.content[0].text)
```

### LangChain

```python
from langchain_anthropic import ChatAnthropic

# export ANTHROPIC_BASE_URL=http://localhost:18082
llm = ChatAnthropic(model="sonnet")
print(llm.invoke("Hello!").content)
```

### curl

```bash
# OpenAI format
curl http://localhost:18082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# OpenAI Responses format
curl http://localhost:18082/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","input":"Hello"}'

# Anthropic format
curl http://localhost:18082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}'
```

## Configuration

```
agent-relay serve [--host HOST] [--port PORT]
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `18082` | Bind port |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |
| `/v1/responses` | POST | Responses API (OpenAI-compatible) |
| `/v1/messages` | POST | Messages (Anthropic-compatible) |
| `/v1/models` | GET | List available models |
| `/v1/metrics` | GET | Relay counters and uptime |
| `/health` | GET | Server and CLI status |

All endpoints also work without the `/v1` prefix. CORS is enabled for all origins.

## Observability

Request logging is enabled by default with:

- request method, path, status, and duration
- per-request `X-Request-Id` header (accepts incoming `x-request-id` too)
- in-process counters exposed at `/metrics` (and `/v1/metrics`)

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `AGENT_RELAY_LOG_LEVEL` | `INFO` | Python log level |
| `SENTRY_DSN` | unset | Enable Sentry error reporting |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.0` | Sentry tracing sample rate |
| `SENTRY_ENVIRONMENT` | unset | Sentry environment tag |
| `SENTRY_LOG_LEVEL` | `INFO` | Breadcrumb/log capture threshold sent to Sentry |
| `SENTRY_EVENT_LEVEL` | `ERROR` | Log level that becomes a Sentry event |
| `BETTERSTACK_SOURCE_TOKEN` | unset | Send logs to Better Stack |
| `BETTERSTACK_INGESTING_HOST` | `https://in.logs.betterstack.com` | Better Stack ingest host |
| `BETTERSTACK_LOG_LEVEL` | `INFO` | Better Stack handler level |
| `AGENT_RELAY_SMART_ROUTING` | `false` | Route default/auto requests to `haiku`/`sonnet`/`opus` |
| `AGENT_RELAY_SMART_ROUTING_FORCE` | `false` | Route even when a specific model is requested |
| `AGENT_RELAY_ROUTING_DEFAULT_MODEL` | `sonnet` | Base model used when model is omitted |
| `AGENT_RELAY_ROUTING_CLASSIFIER_BACKEND` | `ollama` | Router classifier backend |
| `AGENT_RELAY_ROUTING_CLASSIFIER_MODEL` | `qwen2.5:3b` | Local classifier model |
| `AGENT_RELAY_ROUTING_CLASSIFIER_URL` | `http://127.0.0.1:11434/api/chat` | Local classifier endpoint |
| `AGENT_RELAY_ROUTING_CLASSIFIER_TIMEOUT` | `2.0` | Classifier timeout in seconds |
| `AGENT_RELAY_ROUTING_CLASSIFIER_CACHE_TTL` | `600` | Classifier cache TTL in seconds |
| `AGENT_RELAY_ROUTE_MODEL_SIMPLE` | `haiku` | Model for SIMPLE tier |
| `AGENT_RELAY_ROUTE_MODEL_MEDIUM` | `sonnet` | Model for MEDIUM tier |
| `AGENT_RELAY_ROUTE_MODEL_COMPLEX` | `opus` | Model for COMPLEX tier |
| `AGENT_RELAY_ROUTE_MODEL_REASONING` | `opus` | Model for REASONING tier |

Install Sentry support:

```bash
uv pip install 'agentrelay-cli[observability]'
```

Crash detection tip: monitor `/health` fields `pid` and `started_at`; if either changes unexpectedly, the process restarted.

Minimal setup example:

```bash
export SENTRY_DSN="https://<key>@o<org>.ingest.sentry.io/<project>"
export SENTRY_ENVIRONMENT="prod"
export SENTRY_TRACES_SAMPLE_RATE="0.1"
export SENTRY_LOG_LEVEL="INFO"
export SENTRY_EVENT_LEVEL="ERROR"
agent-relay serve
```

### Better Stack logs

Install observability extras:

```bash
uv pip install 'agentrelay-cli[observability]'
```

Set env vars and run:

```bash
export BETTERSTACK_SOURCE_TOKEN="<your-source-token>"
export BETTERSTACK_INGESTING_HOST="https://in.logs.betterstack.com"
export BETTERSTACK_LOG_LEVEL="INFO"
agent-relay serve
```

Then open Better Stack Logs and filter by `service=claude-relay` or `request_id`.

### Smart model routing

Use `model: "auto"` (or enable global routing) to let relay choose a model by prompt complexity:

```bash
export AGENT_RELAY_SMART_ROUTING=true
export AGENT_RELAY_ROUTING_DEFAULT_MODEL=sonnet
agent-relay serve
```

Behavior:

- routing is LLM-first (local classifier), no rule fallback path
- explicit model choice is preserved unless `AGENT_RELAY_SMART_ROUTING_FORCE=true`
- if classifier fails/times out, request returns `routing_error` (HTTP 503)

Strategy notes:

- tier classifier returns exactly one tier: SIMPLE/MEDIUM/COMPLEX/REASONING
- tier maps to configurable models via env vars

### Ship local logs to remote Grafana/Loki (with reconnect)

If `agent-relay` runs on your laptop and Grafana/Loki is on another host (for example `hetzner-recon`), use the included scripts:

Terminal 1:

```bash
./scripts/run-relay-with-logfile.sh
```

Terminal 2:

```bash
./scripts/run-promtail-shipper.sh hetzner-recon
```

What this does:

- keeps a local durable log file at `~/.claude-relay-observability/logs/relay.log`
- starts an SSH tunnel to remote Loki (`127.0.0.1:3100` on the remote host)
- runs Promtail in Docker to ship logs to Loki through the tunnel
- auto-reconnects the tunnel if network drops, then Promtail resumes sending

Grafana Explore query:

```logql
{job="claude-relay", service="claude-relay"}
```

### Supported features

| Feature | Status |
|---|---|
| Streaming (SSE) | Yes |
| System messages | Yes (via `--system-prompt`) |
| Multi-turn conversations | Yes |
| Multimodal (text parts) | Yes |
| Model selection | Yes |
| Token usage reporting | Yes |
| CORS | Yes |

### Models

Pass any model name — it goes directly to `claude --model`:

| Model | Description |
|---|---|
| `opus` | Most capable |
| `sonnet` | Balanced (default) |
| `haiku` | Fastest |

## Limitations

- `temperature`, `max_tokens`, `top_p`, and other sampling parameters are ignored (Claude Code CLI does not expose them)
- No tool/function calling passthrough (Claude Code uses its own tools internally, but they aren't exposed via the OpenAI tool-calling protocol)
- Each request spawns a new `claude` process (~2-3s overhead on top of API latency)
- No image/audio content forwarding — only text parts of multimodal messages are extracted

## How it works

```
OpenAI client     ─┐
                    ├→  claude-relay  →  claude -p  →  Anthropic API
Anthropic client  ─┘     (FastAPI)      (stream-json)
```

Each request spawns a `claude -p` process with `--output-format stream-json --include-partial-messages`. The proxy translates between the OpenAI or Anthropic wire format and Claude Code's streaming JSON protocol. Requests are stateless — no conversation history bleeds between calls.

## Development

```bash
uv sync
uv run pytest tests/ -v
```

## License

[MIT](LICENSE)
