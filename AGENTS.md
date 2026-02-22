# AGENTS.md — Claude-Relay

## What this is

Drop-in OpenAI & Anthropic API-compatible server that routes requests through `claude -p` CLI.

## Quick setup

**Prerequisites:** `claude` CLI installed and on PATH, Python 3.10+, `uv`

```bash
# Install and run
uvx claude-relay serve

# Or from source
git clone https://github.com/npow/claude-relay.git
cd claude-relay && uv sync && uv run claude-relay serve
```

Default: `http://0.0.0.0:8082`. Override with `--host` / `--port`.

## Configuring clients

**OpenAI SDK (Python):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8082/v1", api_key="unused")
response = client.chat.completions.create(
    model="sonnet", messages=[{"role": "user", "content": "Hello"}], stream=True
)
```

**Anthropic SDK (Python):**
```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8082", api_key="unused")
response = client.messages.create(
    model="sonnet", max_tokens=1024, messages=[{"role": "user", "content": "Hello"}]
)
```

**LangChain:**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8082/v1", api_key="unused", model="sonnet")
```

**curl:**
```bash
curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"Hello"}]}'
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server + CLI status |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| POST | `/v1/messages` | Anthropic-compatible messages |

All endpoints also available without `/v1` prefix (e.g. `/models`, `/chat/completions`, `/messages`).

## Models

| Model | Notes |
|-------|-------|
| `opus` | Most capable |
| `sonnet` | **Default** if omitted |
| `haiku` | Fastest |

Passed directly to `claude --model`.

## Ignored parameters

These are silently discarded — Claude Code CLI does not expose them:

`temperature`, `max_tokens`, `top_p`, `top_k`, `n`, `tools`, `tool_choice`, `functions`, `function_call`, `response_format`

Image/audio content blocks are stripped to text-only.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Health check fails | `curl http://localhost:8082/health` — verify `claude` is on PATH |
| `claude` not found | Ensure CLI installed: `which claude`. Restart shell if just installed |
| Slow responses (~2-3s overhead) | Expected — each request spawns a `claude -p` subprocess |
| Images/audio ignored | Only text content is extracted from multimodal messages |
| No tool calling | Claude Code uses its own tools internally; tool parameters are ignored |
| CORS errors | CORS is enabled for all origins by default |

## Development

**Source layout:**
```
src/claude_relay/
├── __init__.py        # Version
├── __main__.py        # CLI entry point (argparse)
└── server.py          # FastAPI app, all endpoints and logic
tests/
└── test_server.py     # Unit + integration tests
```

**Run tests:**
```bash
uv sync
uv run pytest tests/ -v
```
