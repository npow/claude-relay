"""OpenAI-compatible API proxy that routes requests through Claude Code CLI."""

import asyncio
import json
import shutil
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__

app = FastAPI(title="claude-relay", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = [
    {"id": "opus", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
    {"id": "sonnet", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
    {"id": "haiku", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(messages: list[dict]) -> tuple[Optional[str], str]:
    """Convert OpenAI messages array to (system_prompt, user_prompt)."""
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(c["text"] for c in content if c.get("type") == "text")

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            conversation_parts.append(f"User: {content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    user_msgs = [m for m in messages if m.get("role") == "user"]
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]

    if len(user_msgs) == 1 and not asst_msgs:
        content = user_msgs[0].get("content", "")
        if isinstance(content, list):
            content = "\n".join(c["text"] for c in content if c.get("type") == "text")
        prompt = content
    else:
        prompt = "\n\n".join(conversation_parts)

    return system_prompt, prompt


# ---------------------------------------------------------------------------
# Claude CLI command
# ---------------------------------------------------------------------------


def build_claude_cmd(
    prompt: str,
    system_prompt: Optional[str],
    model: Optional[str],
) -> list[str]:
    cmd = ["claude", "-p", "--output-format", "stream-json", "--include-partial-messages"]
    if model:
        cmd.extend(["--model", model])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    cmd.append(prompt)
    return cmd


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def make_chat_response(text: str, model: str, usage: dict | None = None) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": (usage or {}).get("input_tokens", 0),
            "completion_tokens": (usage or {}).get("output_tokens", 0),
            "total_tokens": (
                (usage or {}).get("input_tokens", 0)
                + (usage or {}).get("output_tokens", 0)
            ),
        },
    }


def make_stream_chunk(
    text: str,
    model: str,
    chunk_id: str,
    finish_reason: str | None = None,
) -> dict:
    delta = {"content": text} if text else {}
    if finish_reason:
        delta = {}
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    cli_found = shutil.which("claude") is not None
    return {
        "status": "ok" if cli_found else "degraded",
        "version": __version__,
        "claude_cli": cli_found,
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "sonnet")
    stream = body.get("stream", False)

    system_prompt, prompt = build_prompt(messages)
    cmd = build_claude_cmd(prompt, system_prompt, model)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        initial = make_stream_chunk("", model, chunk_id)
        initial["choices"][0]["delta"] = {"role": "assistant", "content": ""}

        async def generate():
            yield f"data: {json.dumps(initial)}\n\n"
            async for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "stream_event":
                    continue
                inner = event.get("event", {})
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield f"data: {json.dumps(make_stream_chunk(text, model, chunk_id))}\n\n"
                elif inner.get("type") == "message_delta":
                    if inner.get("delta", {}).get("stop_reason"):
                        yield f"data: {json.dumps(make_stream_chunk('', model, chunk_id, finish_reason='stop'))}\n\n"
            yield "data: [DONE]\n\n"
            await proc.wait()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    result_text = ""
    usage: dict = {}
    async for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result_text = event.get("result", "")
            usage = event.get("usage", {})

    await proc.wait()

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Claude CLI error: {stderr.decode()}", "type": "server_error"}},
        )

    return JSONResponse(content=make_chat_response(result_text, model, usage))
