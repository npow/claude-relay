# claude-relay

[![CI](https://github.com/npow/claude-relay/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/claude-relay/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/claude-relay)](https://pypi.org/project/claude-relay/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Drop-in OpenAI **and Anthropic** API server that routes through [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## Why

You have tools that speak the OpenAI or Anthropic API. You have Claude Code with its tools, MCP servers, and agentic capabilities. **claude-relay** bridges the two — point any compatible client at it and every request flows through `claude -p` under the hood.

- **Use Claude Code from any OpenAI or Anthropic client** — Cursor, Continue, aider, LangChain, custom scripts
- **Keep Claude Code's superpowers** — tool use, MCP servers, file access, shell execution
- **Zero config** — if `claude` works on your machine, so does this
- **Real token usage** — reports actual token counts from Claude (not zeros)
- **Token-level streaming** — uses `--include-partial-messages` for true real-time deltas

## Install

```bash
# With uv (recommended)
uvx claude-relay serve

# Or install globally
uv tool install claude-relay
claude-relay serve

# Or from source
git clone https://github.com/npow/claude-relay.git
cd claude-relay
uv sync
uv run claude-relay serve
```

## Quick start

```bash
claude-relay serve
# Server starts on http://localhost:18082
```

### Run as background service (macOS)

```bash
# Install and auto-start on login
claude-relay service install
```

The installer will offer to add these to your `~/.zshrc` (or `~/.bashrc`) so every SDK and agent picks up the relay automatically:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:18082"
export OPENAI_BASE_URL="http://127.0.0.1:18082/v1"
```

```bash
# Check status
claude-relay service status

# Update
uv tool upgrade claude-relay
claude-relay service restart

# Stop and remove
claude-relay service uninstall
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
claude-relay serve [--host HOST] [--port PORT]
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
| `/health` | GET | Server and CLI status |

All endpoints also work without the `/v1` prefix. CORS is enabled for all origins.

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
