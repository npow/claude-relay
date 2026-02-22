"""Tests for claude-relay."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from claude_relay.server import app, build_claude_cmd, build_prompt, make_chat_response, make_stream_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE = "claude_relay.server"


def _make_claude_stream_lines(text_chunks, input_tokens=5, output_tokens=10):
    """Build fake claude stream-json stdout lines."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "fake"}),
        json.dumps({"type": "stream_event", "event": {"type": "message_start", "message": {"role": "assistant"}}}),
        json.dumps({"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}}),
    ]
    for chunk in text_chunks:
        lines.append(json.dumps({
            "type": "stream_event",
            "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk}},
        }))
    lines.append(json.dumps({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}}))
    lines.append(json.dumps({
        "type": "stream_event",
        "event": {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
    }))
    lines.append(json.dumps({
        "type": "result", "subtype": "success", "result": "".join(text_chunks),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }))
    return "\n".join(lines) + "\n"


def _mock_process(stdout_data: bytes, returncode: int = 0):
    proc = AsyncMock()
    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=returncode)
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"error")

    async def _aiter():
        for line in stdout_data.split(b"\n"):
            yield line + b"\n"

    proc.stdout = _aiter()
    return proc


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Unit tests: build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_single_user_message(self):
        system, prompt = build_prompt([{"role": "user", "content": "Hello"}])
        assert system is None
        assert prompt == "Hello"

    def test_system_and_user(self):
        system, prompt = build_prompt([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "You are helpful."
        assert prompt == "Hi"

    def test_multiple_system_messages(self):
        system, _ = build_prompt([
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Go"},
        ])
        assert system == "Rule 1\n\nRule 2"

    def test_multi_turn(self):
        _, prompt = build_prompt([
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "3+3?"},
        ])
        assert "User: 2+2?" in prompt
        assert "Assistant: 4" in prompt
        assert "User: 3+3?" in prompt

    def test_multimodal_content(self):
        _, prompt = build_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }])
        assert prompt == "Describe"

    def test_empty_messages(self):
        system, prompt = build_prompt([])
        assert system is None
        assert prompt == ""

    def test_unknown_role_ignored(self):
        _, prompt = build_prompt([
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "Hello"},
        ])
        assert prompt == "Hello"


# ---------------------------------------------------------------------------
# Unit tests: build_claude_cmd
# ---------------------------------------------------------------------------


class TestBuildClaudeCmd:
    def test_basic(self):
        cmd = build_claude_cmd("Hi", None, None)
        assert cmd[:2] == ["claude", "-p"]
        assert "--output-format" in cmd
        assert cmd[-1] == "Hi"

    def test_with_model(self):
        cmd = build_claude_cmd("Hi", None, "opus")
        assert cmd[cmd.index("--model") + 1] == "opus"

    def test_with_system_prompt(self):
        cmd = build_claude_cmd("Hi", "Be nice", None)
        assert cmd[cmd.index("--system-prompt") + 1] == "Be nice"

    def test_model_passthrough(self):
        cmd = build_claude_cmd("Hi", None, "my-custom-model")
        assert cmd[cmd.index("--model") + 1] == "my-custom-model"


# ---------------------------------------------------------------------------
# Unit tests: response builders
# ---------------------------------------------------------------------------


class TestMakeChatResponse:
    def test_basic(self):
        resp = make_chat_response("Hello!", "sonnet")
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["content"] == "Hello!"
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["id"].startswith("chatcmpl-")

    def test_with_usage(self):
        resp = make_chat_response("Hi", "opus", {"input_tokens": 10, "output_tokens": 20})
        assert resp["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_without_usage(self):
        resp = make_chat_response("Hi", "opus")
        assert resp["usage"]["total_tokens"] == 0


class TestMakeStreamChunk:
    def test_content(self):
        c = make_stream_chunk("Hello", "sonnet", "id-1")
        assert c["choices"][0]["delta"] == {"content": "Hello"}
        assert c["choices"][0]["finish_reason"] is None

    def test_finish(self):
        c = make_stream_chunk("", "sonnet", "id-1", finish_reason="stop")
        assert c["choices"][0]["delta"] == {}
        assert c["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Integration tests: endpoints
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "claude_cli" in body


@pytest.mark.anyio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert set(ids) == {"opus", "sonnet", "haiku"}


@pytest.mark.anyio
async def test_list_models_no_prefix(client):
    resp = await client.get("/models")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_cors_headers(client):
    resp = await client.options("/v1/chat/completions", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" in resp.headers


@pytest.mark.anyio
async def test_non_streaming(client):
    data = _make_claude_stream_lines(["Hello!"], input_tokens=12, output_tokens=5)
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/chat/completions", json={
            "model": "sonnet", "messages": [{"role": "user", "content": "Hi"}],
        })

    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


@pytest.mark.anyio
async def test_non_streaming_with_system_prompt(client):
    data = _make_claude_stream_lines(["Arrr!"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await client.post("/v1/chat/completions", json={
            "model": "haiku",
            "messages": [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello"},
            ],
        })

    cmd = mock_exec.call_args[0]
    idx = list(cmd).index("--system-prompt")
    assert cmd[idx + 1] == "You are a pirate."


@pytest.mark.anyio
async def test_streaming(client):
    data = _make_claude_stream_lines(["Hello", " world", "!"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/chat/completions", json={
            "model": "opus", "messages": [{"role": "user", "content": "Hi"}], "stream": True,
        })

    assert "text/event-stream" in resp.headers["content-type"]

    events = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
        elif line == "data: [DONE]":
            events.append("[DONE]")

    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    texts = [e["choices"][0]["delta"]["content"] for e in events[1:] if e != "[DONE]" and "content" in e["choices"][0]["delta"]]
    assert texts == ["Hello", " world", "!"]
    assert any(e != "[DONE]" and e["choices"][0]["finish_reason"] == "stop" for e in events)
    assert events[-1] == "[DONE]"


@pytest.mark.anyio
async def test_streaming_chunks_share_id(client):
    data = _make_claude_stream_lines(["a", "b"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/chat/completions", json={
            "model": "sonnet", "messages": [{"role": "user", "content": "Hi"}], "stream": True,
        })

    ids = set()
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            ids.add(json.loads(line[6:])["id"])
    assert len(ids) == 1


@pytest.mark.anyio
async def test_no_prefix(client):
    data = _make_claude_stream_lines(["Works"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/chat/completions", json={
            "model": "sonnet", "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.json()["choices"][0]["message"]["content"] == "Works"


@pytest.mark.anyio
async def test_default_model(client):
    data = _make_claude_stream_lines(["Hi"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.json()["model"] == "sonnet"


@pytest.mark.anyio
async def test_multimodal(client):
    data = _make_claude_stream_lines(["I see"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await client.post("/v1/chat/completions", json={
            "model": "sonnet",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }],
        })

    prompt_arg = mock_exec.call_args[0][-1]
    assert prompt_arg == "What is this?"


@pytest.mark.anyio
async def test_multi_turn(client):
    data = _make_claude_stream_lines(["6"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await client.post("/v1/chat/completions", json={
            "model": "sonnet",
            "messages": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "3+3?"},
            ],
        })

    prompt_arg = mock_exec.call_args[0][-1]
    assert "User: 2+2?" in prompt_arg
    assert "Assistant: 4" in prompt_arg
    assert "User: 3+3?" in prompt_arg


@pytest.mark.anyio
async def test_response_id_format(client):
    data = _make_claude_stream_lines(["Hi"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/chat/completions", json={
            "model": "sonnet", "messages": [{"role": "user", "content": "Hi"}],
        })

    assert resp.json()["id"].startswith("chatcmpl-")
    assert len(resp.json()["id"]) == len("chatcmpl-") + 12
