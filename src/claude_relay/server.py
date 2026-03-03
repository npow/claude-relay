"""OpenAI- and Anthropic-compatible API proxy that routes requests through Claude Code CLI."""

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager as _acm
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
except ImportError:  # pragma: no cover - optional dependency
    sentry_sdk = None
    LoggingIntegration = None

try:
    from logtail import LogtailHandler
except ImportError:  # pragma: no cover - optional dependency
    LogtailHandler = None

# ---------------------------------------------------------------------------
# Concurrency & lifecycle state
# ---------------------------------------------------------------------------

_max_concurrent: int = int(os.environ.get("CLAUDE_RELAY_MAX_CONCURRENT", "10"))
_request_timeout: float = float(os.environ.get("CLAUDE_RELAY_REQUEST_TIMEOUT", "300"))
_active_processes: set = set()
# Environment for child claude processes — strip relay URL vars to prevent recursive loops
_subprocess_env: dict = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "CLAUDECODE")}
_active_count: int = 0
_semaphore: asyncio.Semaphore = asyncio.Semaphore(_max_concurrent)
_started_at: float = time.time()
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-",
)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        record.request_id = _request_id_var.get("-")
        return True


def _configure_logging() -> None:
    level_name = os.environ.get("AGENT_RELAY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format=(
                "%(asctime)s %(levelname)s %(name)s "
                "request_id=%(request_id)s %(message)s"
            ),
        )
    root.setLevel(level)
    request_filter = _RequestIdFilter()
    for handler in root.handlers:
        handler.addFilter(request_filter)


_configure_logging()
logger = logging.getLogger("claude_relay.server")


def _init_betterstack() -> None:
    source_token = os.environ.get("BETTERSTACK_SOURCE_TOKEN")
    if not source_token:
        return
    if LogtailHandler is None:
        logger.warning("BETTERSTACK_SOURCE_TOKEN is set but logtail-python is not installed")
        return
    host = os.environ.get("BETTERSTACK_INGESTING_HOST", "https://in.logs.betterstack.com")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"https://{host}"
    handler = LogtailHandler(source_token=source_token, host=host)
    handler.setLevel(getattr(logging, os.environ.get("BETTERSTACK_LOG_LEVEL", "INFO").upper(), logging.INFO))
    logging.getLogger().addHandler(handler)
    logger.info("Better Stack logging initialized host=%s", host)


def _init_sentry() -> None:
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return
    if sentry_sdk is None:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed")
        return
    sentry_log_level_name = os.environ.get("SENTRY_LOG_LEVEL", "INFO").upper()
    sentry_event_level_name = os.environ.get("SENTRY_EVENT_LEVEL", "ERROR").upper()
    sentry_log_level = getattr(logging, sentry_log_level_name, logging.INFO)
    sentry_event_level = getattr(logging, sentry_event_level_name, logging.ERROR)
    integrations = []
    if LoggingIntegration is not None:
        integrations.append(
            LoggingIntegration(level=sentry_log_level, event_level=sentry_event_level),
        )
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        environment=os.environ.get("SENTRY_ENVIRONMENT"),
        release=f"agent-relay@{__version__}",
        integrations=integrations,
    )
    logger.info(
        "Sentry initialized log_level=%s event_level=%s",
        sentry_log_level_name,
        sentry_event_level_name,
    )


_init_sentry()
_init_betterstack()

_stats: dict[str, int] = {
    "requests_total": 0,
    "timeouts_total": 0,
    "capacity_rejections_total": 0,
    "subprocess_start_failures_total": 0,
    "subprocess_errors_total": 0,
    "uncaught_exceptions_total": 0,
}
_smart_routing_enabled: bool = os.environ.get("AGENT_RELAY_SMART_ROUTING", "false").lower() in {"1", "true", "yes", "on"}
_smart_routing_force: bool = os.environ.get("AGENT_RELAY_SMART_ROUTING_FORCE", "false").lower() in {"1", "true", "yes", "on"}
_routing_default_model: str = os.environ.get("AGENT_RELAY_ROUTING_DEFAULT_MODEL", "sonnet")
_routing_tier_models: dict[str, str] = {
    "SIMPLE": os.environ.get("AGENT_RELAY_ROUTE_MODEL_SIMPLE", "haiku"),
    "MEDIUM": os.environ.get("AGENT_RELAY_ROUTE_MODEL_MEDIUM", "sonnet"),
    "COMPLEX": os.environ.get("AGENT_RELAY_ROUTE_MODEL_COMPLEX", "opus"),
    "REASONING": os.environ.get("AGENT_RELAY_ROUTE_MODEL_REASONING", "opus"),
}
_routing_classifier_backend: str = os.environ.get("AGENT_RELAY_ROUTING_CLASSIFIER_BACKEND", "ollama").lower()
_routing_classifier_model: str = os.environ.get("AGENT_RELAY_ROUTING_CLASSIFIER_MODEL", "qwen2.5:3b")
_routing_classifier_url: str = os.environ.get("AGENT_RELAY_ROUTING_CLASSIFIER_URL", "http://127.0.0.1:11434/api/chat")
_routing_classifier_timeout: float = float(os.environ.get("AGENT_RELAY_ROUTING_CLASSIFIER_TIMEOUT", "2.0"))
_routing_classifier_cache_ttl: int = int(os.environ.get("AGENT_RELAY_ROUTING_CLASSIFIER_CACHE_TTL", "600"))
_routing_classifier_cache: dict[str, tuple[float, dict]] = {}


def configure(max_concurrent: int = 10, request_timeout: float = 300.0) -> None:
    """Set concurrency and timeout limits.  Call before starting the server."""
    global _semaphore, _max_concurrent, _request_timeout, _active_count
    _max_concurrent = max_concurrent
    _request_timeout = request_timeout
    _active_count = 0
    _semaphore = asyncio.Semaphore(max_concurrent)
    _stats.update({
        "requests_total": 0,
        "timeouts_total": 0,
        "capacity_rejections_total": 0,
        "subprocess_start_failures_total": 0,
        "subprocess_errors_total": 0,
        "uncaught_exceptions_total": 0,
        "routing_failures_total": 0,
    })


def _extract_tier(text: str) -> str:
    m = re.search(r"\b(SIMPLE|MEDIUM|COMPLEX|REASONING)\b", text.upper())
    if not m:
        raise ValueError(f"Classifier response missing tier label: {text[:120]}")
    return m.group(1)


def _classify_tier_ollama(prompt: str, message_count: int, structured_output: bool) -> dict:
    cache_key = hashlib.sha256(
        f"{prompt}|{message_count}|{int(structured_output)}|{_routing_classifier_model}".encode(),
    ).hexdigest()
    now = time.time()
    cached = _routing_classifier_cache.get(cache_key)
    if cached and (now - cached[0]) <= _routing_classifier_cache_ttl:
        return cached[1]

    classifier_prompt = (
        "Classify the user request into exactly one tier for model routing.\n"
        "Return only one word: SIMPLE, MEDIUM, COMPLEX, or REASONING.\n"
        "Use REASONING for formal proofs/multi-step logical derivations.\n"
        "Use COMPLEX for hard coding/system design/debug tasks.\n"
        "Use MEDIUM for normal coding and analytical tasks.\n"
        "Use SIMPLE for short factual/basic chat tasks.\n"
        f"Message count: {message_count}\n"
        f"Structured output required: {'yes' if structured_output else 'no'}\n\n"
        f"User prompt:\n{prompt[:6000]}"
    )
    payload = json.dumps(
        {
            "model": _routing_classifier_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a strict router classifier."},
                {"role": "user", "content": classifier_prompt},
            ],
            "options": {"temperature": 0},
        },
    ).encode()
    req = urllib.request.Request(
        _routing_classifier_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_routing_classifier_timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Routing classifier unavailable: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Routing classifier timed out") from exc

    text = (
        (body.get("message") or {}).get("content")
        or body.get("response")
        or ""
    )
    tier = _extract_tier(text)
    result = {
        "tier": tier,
        "model": _routing_tier_models.get(tier, "sonnet"),
        "confidence": None,
        "signals": ["llm_classifier"],
        "backend": _routing_classifier_backend,
    }
    _routing_classifier_cache[cache_key] = (now, result)
    return result


def _select_model(
    requested_model: str | None,
    prompt: str,
    message_count: int = 1,
    structured_output: bool = False,
) -> tuple[str, bool]:
    """Select model based on request + routing config. Returns (model, routed)."""
    requested = (requested_model or "").strip()
    if not requested:
        requested = _routing_default_model
    should_route = (
        requested == "auto"
        or (_smart_routing_enabled and requested == _routing_default_model)
        or _smart_routing_force
    )
    if not should_route:
        return requested, False
    if _routing_classifier_backend != "ollama":
        raise RuntimeError(
            f"Unsupported routing classifier backend: {_routing_classifier_backend}",
        )
    reasons = _classify_tier_ollama(
        prompt=prompt,
        message_count=message_count,
        structured_output=structured_output,
    )
    model = reasons["model"]
    logger.info(
        "model_routed requested_model=%s selected_model=%s tier=%s backend=%s message_count=%s structured_output=%s signals=%s",
        requested,
        model,
        reasons["tier"],
        reasons["backend"],
        message_count,
        structured_output,
        ",".join(reasons["signals"]),
    )
    return model, True


async def _acquire_slot() -> bool:
    """Non-blocking concurrency-slot acquire.  Returns *True* on success."""
    global _active_count
    if _semaphore.locked():
        return False
    await _semaphore.acquire()
    _active_count += 1
    return True


def _release_slot() -> None:
    global _active_count
    _active_count -= 1
    _semaphore.release()


async def _cleanup_process(proc: asyncio.subprocess.Process) -> None:
    """Kill *proc* if still running, wait for it, and deregister."""
    _active_processes.discard(proc)
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await proc.wait()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lifespan (graceful shutdown)
# ---------------------------------------------------------------------------


@_acm
async def _lifespan(_app: FastAPI):
    yield
    # Shutdown: kill all tracked subprocesses
    procs = list(_active_processes)
    for p in procs:
        try:
            p.kill()
        except ProcessLookupError:
            pass
    for p in procs:
        try:
            await p.wait()
        except Exception:
            pass
    _active_processes.clear()


app = FastAPI(title="agent-relay", version=__version__, lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = _request_id_var.set(request_id)
    _stats["requests_total"] += 1
    start = time.monotonic()
    try:
        response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "request_completed method=%s path=%s status=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    except Exception as exc:
        _stats["uncaught_exceptions_total"] += 1
        logger.exception(
            "request_failed method=%s path=%s",
            request.method,
            request.url.path,
        )
        if sentry_sdk is not None:
            sentry_sdk.capture_exception(exc)
        raise
    finally:
        _request_id_var.reset(token)


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


def build_prompt_anthropic(
    messages: list[dict],
    system: str | list[dict] | None = None,
) -> tuple[Optional[str], str, bool]:
    """Convert Anthropic message format to (system_prompt, user_prompt, force_json).

    Anthropic passes ``system`` as a top-level field (string or list of
    content blocks), not inside the messages array.  Messages use the same
    role/content structure but content can be ``str`` or
    ``list[{type, text}]``.

    When the last message is an assistant prefill (short content like ``{``),
    it is stripped from the conversation and ``force_json=True`` is returned
    so the caller can pass ``--json-schema`` to ``claude -p``.
    """
    # Build system prompt from the top-level system field.
    if isinstance(system, list):
        system_prompt = "\n\n".join(
            b["text"] for b in system if b.get("type") == "text"
        )
    elif isinstance(system, str):
        system_prompt = system
    else:
        system_prompt = None

    # Normalise content blocks to plain strings.
    def _text(content) -> str:
        if isinstance(content, list):
            return "\n".join(c["text"] for c in content if c.get("type") == "text")
        return content or ""

    # Detect assistant prefill: last message is assistant with short content
    force_json = False
    working_messages = list(messages)
    if working_messages and working_messages[-1].get("role") == "assistant":
        prefill = _text(working_messages[-1].get("content", "")).strip()
        if prefill in ("{", "[", '{"'):
            force_json = True
            working_messages = working_messages[:-1]

    conversation_parts: list[str] = []
    for msg in working_messages:
        role = msg.get("role", "user")
        text = _text(msg.get("content", ""))
        if role == "user":
            conversation_parts.append(f"User: {text}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {text}")

    user_msgs = [m for m in working_messages if m.get("role") == "user"]
    asst_msgs = [m for m in working_messages if m.get("role") == "assistant"]

    if len(user_msgs) == 1 and not asst_msgs:
        prompt = _text(user_msgs[0].get("content", ""))
    else:
        prompt = "\n\n".join(conversation_parts)

    return system_prompt or None, prompt, force_json


def build_prompt_responses(
    input_data,
    instructions: str | None = None,
) -> tuple[Optional[str], str]:
    """Convert OpenAI Responses API input to (system_prompt, user_prompt)."""
    messages: list[dict] = []

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            if "role" in item:
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    text_blocks: list[dict] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type in {"input_text", "text"}:
                            text_blocks.append(
                                {"type": "text", "text": block.get("text", "")},
                            )
                    messages.append({"role": role, "content": text_blocks})
            elif item.get("type") in {"input_text", "text"}:
                messages.append({"role": "user", "content": item.get("text", "")})

    system_prompt, prompt = build_prompt(messages)
    if instructions:
        system_prompt = f"{instructions}\n\n{system_prompt}" if system_prompt else instructions
    return system_prompt, prompt


# ---------------------------------------------------------------------------
# Claude CLI command
# ---------------------------------------------------------------------------


def build_claude_cmd(
    prompt: str,
    system_prompt: Optional[str],
    model: Optional[str],
    force_json: bool = False,
) -> tuple[list[str], str]:
    """Return *(argv, stdin_text)*.  Prompt is always piped via stdin to
    avoid OS ``ARG_MAX`` limits on large payloads."""
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages", "--dangerously-skip-permissions"]
    if model:
        cmd.extend(["--model", model])
    effective_system = system_prompt or ""
    if force_json:
        json_instruction = (
            "\n\nCRITICAL: Your ENTIRE response must be a single valid JSON object. "
            "Start with { and end with }. No markdown, no code blocks, no prose before or after. "
            "Just raw JSON."
        )
        effective_system = (effective_system + json_instruction).strip()
    if effective_system:
        cmd.extend(["--system-prompt", effective_system])
    return cmd, prompt


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


def _usage_openai_responses(usage: dict | None = None) -> dict:
    input_tokens = (usage or {}).get("input_tokens", 0)
    output_tokens = (usage or {}).get("output_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def make_responses_response(text: str, model: str, usage: dict | None = None) -> dict:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "usage": _usage_openai_responses(usage),
    }


# ---------------------------------------------------------------------------
# Anthropic response builders
# ---------------------------------------------------------------------------


def make_anthropic_response(text: str, model: str, usage: dict | None = None) -> dict:
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": (usage or {}).get("input_tokens", 0),
            "output_tokens": (usage or {}).get("output_tokens", 0),
        },
    }


def make_anthropic_stream_event(event_type: str, data: dict) -> str:
    """Build one Anthropic SSE frame (``event:`` + ``data:`` lines)."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


async def _read_cli_result(proc) -> tuple[str, dict]:
    """Read all stdout from a Claude CLI subprocess, return *(text, usage)*."""
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
    return result_text, usage


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    backend = os.environ.get("AGENT_RELAY_BACKEND", "claude")
    cli_found = shutil.which("claude") is not None
    if backend == "codex":
        cli_found = shutil.which("codex") is not None
    return {
        "status": "ok" if cli_found else "degraded",
        "version": __version__,
        "backend": backend,
        "claude_cli": cli_found,
        "smart_routing_enabled": _smart_routing_enabled,
        "smart_routing_force": _smart_routing_force,
        "routing_default_model": _routing_default_model,
        "routing_classifier_backend": _routing_classifier_backend,
        "routing_classifier_model": _routing_classifier_model,
        "routing_classifier_url": _routing_classifier_url,
        "routing_tier_models": _routing_tier_models,
        "pid": os.getpid(),
        "started_at": int(_started_at),
        "uptime_seconds": int(time.time() - _started_at),
        "active_requests": _active_count,
        "max_concurrent": _max_concurrent,
        "stats": _stats,
    }


@app.get("/metrics")
@app.get("/v1/metrics")
async def metrics():
    return {
        "uptime_seconds": int(time.time() - _started_at),
        "active_requests": _active_count,
        "max_concurrent": _max_concurrent,
        "stats": _stats,
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
    requested_model = body.get("model", _routing_default_model)
    stream = body.get("stream", False)
    structured_output = body.get("response_format") is not None

    system_prompt, prompt = build_prompt(messages)
    try:
        model, routed = _select_model(
            requested_model,
            prompt,
            message_count=len(messages),
            structured_output=structured_output,
        )
    except RuntimeError as exc:
        _stats["routing_failures_total"] += 1
        logger.error("routing_failed endpoint=chat_completions error=%s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"Routing classifier failed: {exc}",
                    "type": "routing_error",
                },
            },
        )
    if routed:
        _stats_key = f"routed_to_{model}_total"
        _stats[_stats_key] = _stats.get(_stats_key, 0) + 1
    cmd, stdin_text = build_claude_cmd(prompt, system_prompt, model)

    if not await _acquire_slot():
        _stats["capacity_rejections_total"] += 1
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Server at capacity", "type": "capacity_error"}},
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.environ.get("CLAUDE_RELAY_CWD", None),
            env=_subprocess_env,
        )
        proc.stdin.write(stdin_text.encode())
        proc.stdin.write_eof()
    except OSError as exc:
        _stats["subprocess_start_failures_total"] += 1
        logger.exception("subprocess_start_failed model=%s", model)
        if sentry_sdk is not None:
            sentry_sdk.capture_exception(exc)
        _release_slot()
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"Failed to start subprocess: {exc}", "type": "server_error"}},
        )

    _active_processes.add(proc)

    if stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        initial = make_stream_chunk("", model, chunk_id)
        initial["choices"][0]["delta"] = {"role": "assistant", "content": ""}

        async def generate():
            try:
                deadline = time.monotonic() + _request_timeout
                yield f"data: {json.dumps(initial)}\n\n"
                async for raw in proc.stdout:
                    if time.monotonic() > deadline:
                        break
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
            finally:
                await _cleanup_process(proc)
                _release_slot()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    try:
        result_text, usage = await asyncio.wait_for(
            _read_cli_result(proc), timeout=_request_timeout,
        )
    except asyncio.TimeoutError:
        _stats["timeouts_total"] += 1
        logger.warning("request_timeout model=%s endpoint=chat_completions", model)
        await _cleanup_process(proc)
        _release_slot()
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Request timed out", "type": "timeout_error"}},
        )

    _active_processes.discard(proc)
    _release_slot()

    if proc.returncode != 0:
        _stats["subprocess_errors_total"] += 1
        stderr = await proc.stderr.read()
        logger.error(
            "subprocess_failed model=%s endpoint=chat_completions returncode=%s",
            model,
            proc.returncode,
        )
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Claude CLI error: {stderr.decode()}", "type": "server_error"}},
        )

    return JSONResponse(content=make_chat_response(result_text, model, usage))


@app.post("/v1/responses")
@app.post("/responses")
async def responses(request: Request):
    body = await request.json()
    requested_model = body.get("model", _routing_default_model)
    stream = body.get("stream", False)
    input_data = body.get("input", "")
    instructions = body.get("instructions")
    text_blob = f"{instructions or ''}".lower()
    structured_output = any(k in text_blob for k in ["json", "schema", "structured"])

    system_prompt, prompt = build_prompt_responses(input_data, instructions)
    message_count = len(input_data) if isinstance(input_data, list) else 1
    try:
        model, routed = _select_model(
            requested_model,
            prompt,
            message_count=message_count,
            structured_output=structured_output,
        )
    except RuntimeError as exc:
        _stats["routing_failures_total"] += 1
        logger.error("routing_failed endpoint=responses error=%s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"Routing classifier failed: {exc}",
                    "type": "routing_error",
                },
            },
        )
    if routed:
        _stats_key = f"routed_to_{model}_total"
        _stats[_stats_key] = _stats.get(_stats_key, 0) + 1
    cmd, stdin_text = build_claude_cmd(prompt, system_prompt, model)

    if not await _acquire_slot():
        _stats["capacity_rejections_total"] += 1
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Server at capacity", "type": "capacity_error"}},
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_subprocess_env,
        )
        proc.stdin.write(stdin_text.encode())
        proc.stdin.write_eof()
    except OSError as exc:
        _stats["subprocess_start_failures_total"] += 1
        logger.exception("subprocess_start_failed model=%s", model)
        if sentry_sdk is not None:
            sentry_sdk.capture_exception(exc)
        _release_slot()
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"Failed to start subprocess: {exc}", "type": "server_error"}},
        )

    _active_processes.add(proc)

    if stream:
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        async def generate():
            output_text = ""
            usage = {}
            try:
                deadline = time.monotonic() + _request_timeout
                yield "data: " + json.dumps({
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": int(time.time()),
                        "status": "in_progress",
                        "model": model,
                        "output": [],
                        "usage": _usage_openai_responses(),
                    },
                }) + "\n\n"
                async for raw in proc.stdout:
                    if time.monotonic() > deadline:
                        break
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "result":
                        usage = event.get("usage", {})
                        if event.get("result"):
                            output_text = event.get("result")
                        continue

                    if event.get("type") != "stream_event":
                        continue

                    inner = event.get("event", {})
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                output_text += text
                                yield "data: " + json.dumps({
                                    "type": "response.output_text.delta",
                                    "response_id": response_id,
                                    "item_id": message_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": text,
                                }) + "\n\n"

                await proc.wait()
                yield "data: " + json.dumps({
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": output_text,
                }) + "\n\n"

                yield "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": int(time.time()),
                        "status": "completed",
                        "model": model,
                        "output": [
                            {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": output_text,
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "output_text": output_text,
                        "usage": _usage_openai_responses(usage),
                    },
                }) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await _cleanup_process(proc)
                _release_slot()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    try:
        result_text, usage = await asyncio.wait_for(
            _read_cli_result(proc), timeout=_request_timeout,
        )
    except asyncio.TimeoutError:
        _stats["timeouts_total"] += 1
        logger.warning("request_timeout model=%s endpoint=responses", model)
        await _cleanup_process(proc)
        _release_slot()
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Request timed out", "type": "timeout_error"}},
        )

    _active_processes.discard(proc)
    _release_slot()

    if proc.returncode != 0:
        _stats["subprocess_errors_total"] += 1
        stderr = await proc.stderr.read()
        logger.error(
            "subprocess_failed model=%s endpoint=responses returncode=%s",
            model,
            proc.returncode,
        )
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Claude CLI error: {stderr.decode()}", "type": "server_error"}},
        )

    return JSONResponse(content=make_responses_response(result_text, model, usage))


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
@app.post("/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    requested_model = body.get("model", _routing_default_model)
    system = body.get("system")
    stream = body.get("stream", False)

    system_prompt, prompt, force_json = build_prompt_anthropic(messages, system)
    try:
        model, routed = _select_model(
            requested_model,
            prompt,
            message_count=len(messages),
            structured_output=force_json,
        )
    except RuntimeError as exc:
        _stats["routing_failures_total"] += 1
        logger.error("routing_failed endpoint=messages error=%s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {
                    "type": "routing_error",
                    "message": f"Routing classifier failed: {exc}",
                },
            },
        )
    if routed:
        _stats_key = f"routed_to_{model}_total"
        _stats[_stats_key] = _stats.get(_stats_key, 0) + 1
    cmd, stdin_text = build_claude_cmd(prompt, system_prompt, model, force_json=force_json)

    if not await _acquire_slot():
        _stats["capacity_rejections_total"] += 1
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Server at capacity"},
            },
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.environ.get("CLAUDE_RELAY_CWD", None),
            env=_subprocess_env,
        )
        proc.stdin.write(stdin_text.encode())
        proc.stdin.write_eof()
    except OSError as exc:
        _stats["subprocess_start_failures_total"] += 1
        logger.exception("subprocess_start_failed model=%s", model)
        if sentry_sdk is not None:
            sentry_sdk.capture_exception(exc)
        _release_slot()
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {"type": "server_error", "message": f"Failed to start subprocess: {exc}"},
            },
        )

    _active_processes.add(proc)

    if stream:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        async def generate():
            try:
                deadline = time.monotonic() + _request_timeout
                # message_start
                yield make_anthropic_stream_event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                })
                # content_block_start
                yield make_anthropic_stream_event("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                output_tokens = 0
                async for raw in proc.stdout:
                    if time.monotonic() > deadline:
                        break
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
                                yield make_anthropic_stream_event("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": text},
                                })
                    elif inner.get("type") == "message_delta":
                        output_tokens = inner.get("usage", {}).get("output_tokens", 0)
                # content_block_stop
                yield make_anthropic_stream_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0,
                })
                # message_delta
                yield make_anthropic_stream_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                })
                # message_stop
                yield make_anthropic_stream_event("message_stop", {
                    "type": "message_stop",
                })
            finally:
                await _cleanup_process(proc)
                _release_slot()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    try:
        result_text, usage = await asyncio.wait_for(
            _read_cli_result(proc), timeout=_request_timeout,
        )
    except asyncio.TimeoutError:
        _stats["timeouts_total"] += 1
        logger.warning("request_timeout model=%s endpoint=messages", model)
        await _cleanup_process(proc)
        _release_slot()
        return JSONResponse(
            status_code=504,
            content={
                "type": "error",
                "error": {"type": "timeout_error", "message": "Request timed out"},
            },
        )

    _active_processes.discard(proc)
    _release_slot()

    if proc.returncode != 0:
        _stats["subprocess_errors_total"] += 1
        stderr = await proc.stderr.read()
        logger.error(
            "subprocess_failed model=%s endpoint=messages returncode=%s",
            model,
            proc.returncode,
        )
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "server_error", "message": f"Claude CLI error: {stderr.decode()}"},
            },
        )

    return JSONResponse(content=make_anthropic_response(result_text, model, usage))
