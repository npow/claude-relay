"""Tests for claude-relay."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from claude_relay import server as _server_mod
from claude_relay.server import (
    _cleanup_process,
    app,
    build_claude_cmd,
    build_prompt,
    build_prompt_anthropic,
    build_prompt_responses,
    configure,
    make_anthropic_response,
    make_anthropic_stream_event,
    make_chat_response,
    make_responses_response,
    make_stream_chunk,
)


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
    proc.kill = Mock()
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


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset concurrency state between tests."""
    configure(max_concurrent=10, request_timeout=300.0)
    yield
    configure(max_concurrent=10, request_timeout=300.0)


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


# ---------------------------------------------------------------------------
# Unit tests: build_prompt_anthropic
# ---------------------------------------------------------------------------


class TestBuildPromptAnthropic:
    def test_single_user_message(self):
        system, prompt = build_prompt_anthropic(
            [{"role": "user", "content": "Hello"}],
        )
        assert system is None
        assert prompt == "Hello"

    def test_system_string(self):
        system, prompt = build_prompt_anthropic(
            [{"role": "user", "content": "Hi"}],
            system="You are helpful.",
        )
        assert system == "You are helpful."
        assert prompt == "Hi"

    def test_system_content_blocks(self):
        system, _ = build_prompt_anthropic(
            [{"role": "user", "content": "Hi"}],
            system=[
                {"type": "text", "text": "Rule 1"},
                {"type": "text", "text": "Rule 2"},
            ],
        )
        assert system == "Rule 1\n\nRule 2"

    def test_multi_turn(self):
        _, prompt = build_prompt_anthropic([
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "3+3?"},
        ])
        assert "User: 2+2?" in prompt
        assert "Assistant: 4" in prompt
        assert "User: 3+3?" in prompt

    def test_content_array(self):
        _, prompt = build_prompt_anthropic([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }])
        assert prompt == "Describe this"

    def test_empty_messages(self):
        system, prompt = build_prompt_anthropic([])
        assert system is None
        assert prompt == ""

    def test_no_system(self):
        system, _ = build_prompt_anthropic(
            [{"role": "user", "content": "Hi"}],
            system=None,
        )
        assert system is None


class TestBuildPromptResponses:
    def test_string_input(self):
        system, prompt = build_prompt_responses("Hello")
        assert system is None
        assert prompt == "Hello"

    def test_message_input_blocks(self):
        system, prompt = build_prompt_responses([
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this"},
                    {"type": "input_image", "image_url": "http://example.com/img.png"},
                ],
            }
        ])
        assert system is None
        assert prompt == "Describe this"

    def test_instructions_become_system(self):
        system, prompt = build_prompt_responses("Hi", instructions="You are concise.")
        assert system == "You are concise."
        assert prompt == "Hi"


# ---------------------------------------------------------------------------
# Unit tests: Anthropic response builders
# ---------------------------------------------------------------------------


class TestMakeAnthropicResponse:
    def test_basic(self):
        resp = make_anthropic_response("Hello!", "sonnet")
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"
        assert resp["content"] == [{"type": "text", "text": "Hello!"}]
        assert resp["stop_reason"] == "end_turn"
        assert resp["id"].startswith("msg_")

    def test_with_usage(self):
        resp = make_anthropic_response("Hi", "opus", {"input_tokens": 10, "output_tokens": 20})
        assert resp["usage"] == {"input_tokens": 10, "output_tokens": 20}

    def test_without_usage(self):
        resp = make_anthropic_response("Hi", "opus")
        assert resp["usage"] == {"input_tokens": 0, "output_tokens": 0}


class TestMakeAnthropicStreamEvent:
    def test_format(self):
        result = make_anthropic_stream_event("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        })
        assert result.startswith("event: content_block_delta\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data = json.loads(result.split("data: ")[1].strip())
        assert data["delta"]["text"] == "hi"


class TestMakeResponsesResponse:
    def test_basic(self):
        resp = make_responses_response("Hello!", "sonnet")
        assert resp["object"] == "response"
        assert resp["status"] == "completed"
        assert resp["output_text"] == "Hello!"
        assert resp["output"][0]["content"][0]["type"] == "output_text"
        assert resp["id"].startswith("resp_")

    def test_with_usage(self):
        resp = make_responses_response("Hi", "opus", {"input_tokens": 10, "output_tokens": 20})
        assert resp["usage"] == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


# ---------------------------------------------------------------------------
# Integration tests: Anthropic endpoints
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_responses_non_streaming(client):
    data = _make_claude_stream_lines(["Hello!"], input_tokens=12, output_tokens=5)
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/responses", json={
            "model": "sonnet",
            "input": "Hi",
        })

    body = resp.json()
    assert body["object"] == "response"
    assert body["output_text"] == "Hello!"
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}


@pytest.mark.anyio
async def test_responses_streaming(client):
    data = _make_claude_stream_lines(["Hello", " world", "!"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/responses", json={
            "model": "sonnet",
            "input": "Hi",
            "stream": True,
        })

    assert "text/event-stream" in resp.headers["content-type"]
    events = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
        elif line == "data: [DONE]":
            events.append("[DONE]")

    assert events[0]["type"] == "response.created"
    delta_texts = [e["delta"] for e in events if e != "[DONE]" and e["type"] == "response.output_text.delta"]
    assert delta_texts == ["Hello", " world", "!"]
    assert any(e != "[DONE]" and e["type"] == "response.completed" for e in events)
    assert events[-1] == "[DONE]"


@pytest.mark.anyio
async def test_anthropic_non_streaming(client):
    data = _make_claude_stream_lines(["Hello!"], input_tokens=12, output_tokens=5)
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/messages", json={
            "model": "sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })

    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "Hello!"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 5}


@pytest.mark.anyio
async def test_anthropic_streaming(client):
    data = _make_claude_stream_lines(["Hello", " world", "!"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/messages", json={
            "model": "opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

    assert "text/event-stream" in resp.headers["content-type"]

    events = []
    current_event = {}
    for line in resp.text.split("\n"):
        line = line.strip()
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            current_event["data"] = json.loads(line[6:])
            events.append(current_event)
            current_event = {}

    # Check event sequence
    event_types = [e["event"] for e in events]
    assert event_types[0] == "message_start"
    assert event_types[1] == "content_block_start"
    assert "content_block_delta" in event_types
    assert event_types[-3] == "content_block_stop"
    assert event_types[-2] == "message_delta"
    assert event_types[-1] == "message_stop"

    # Check text content
    texts = [
        e["data"]["delta"]["text"]
        for e in events
        if e["event"] == "content_block_delta"
    ]
    assert texts == ["Hello", " world", "!"]

    # Check message_start structure
    msg = events[0]["data"]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == []

    # Check message_delta has stop_reason
    delta_event = [e for e in events if e["event"] == "message_delta"][0]
    assert delta_event["data"]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.anyio
async def test_anthropic_system_prompt(client):
    data = _make_claude_stream_lines(["Arrr!"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await client.post("/v1/messages", json={
            "model": "haiku",
            "max_tokens": 1024,
            "system": "You are a pirate.",
            "messages": [{"role": "user", "content": "Hello"}],
        })

    cmd = mock_exec.call_args[0]
    idx = list(cmd).index("--system-prompt")
    assert cmd[idx + 1] == "You are a pirate."


@pytest.mark.anyio
async def test_anthropic_multi_turn(client):
    data = _make_claude_stream_lines(["6"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await client.post("/v1/messages", json={
            "model": "sonnet",
            "max_tokens": 1024,
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
async def test_anthropic_no_prefix(client):
    data = _make_claude_stream_lines(["Works"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/messages", json={
            "model": "sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.json()["content"][0]["text"] == "Works"


@pytest.mark.anyio
async def test_anthropic_default_model(client):
    data = _make_claude_stream_lines(["Hi"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post("/v1/messages", json={
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.json()["model"] == "sonnet"


@pytest.mark.anyio
async def test_anthropic_content_array(client):
    data = _make_claude_stream_lines(["I see"])
    proc = _mock_process(data.encode())

    with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await client.post("/v1/messages", json={
            "model": "sonnet",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {"type": "base64", "data": "abc"}},
                ],
            }],
        })

    prompt_arg = mock_exec.call_args[0][-1]
    assert prompt_arg == "What is this?"


# ---------------------------------------------------------------------------
# Load management tests
# ---------------------------------------------------------------------------


class TestConcurrencyLimits:
    @pytest.mark.anyio
    async def test_rejects_when_full_openai(self, client):
        """All slots taken -> 503 for OpenAI endpoint."""
        configure(max_concurrent=0)
        resp = await client.post("/v1/chat/completions", json={
            "model": "sonnet", "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 503
        assert resp.json()["error"]["type"] == "capacity_error"

    @pytest.mark.anyio
    async def test_rejects_when_full_anthropic(self, client):
        """All slots taken -> 503 for Anthropic endpoint."""
        configure(max_concurrent=0)
        resp = await client.post("/v1/messages", json={
            "model": "sonnet", "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 503
        assert resp.json()["error"]["type"] == "overloaded_error"


class TestRequestTimeout:
    @pytest.mark.anyio
    async def test_timeout_openai(self, client):
        """Subprocess exceeding timeout -> 504."""
        configure(max_concurrent=10, request_timeout=0.05)

        proc = AsyncMock()
        proc.returncode = None
        proc.wait = AsyncMock(return_value=0)
        proc.kill = Mock()
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")

        async def _slow():
            await asyncio.sleep(100)
            yield b""

        proc.stdout = _slow()

        with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
            resp = await client.post("/v1/chat/completions", json={
                "model": "sonnet", "messages": [{"role": "user", "content": "Hi"}],
            })

        assert resp.status_code == 504
        assert resp.json()["error"]["type"] == "timeout_error"
        proc.kill.assert_called_once()

    @pytest.mark.anyio
    async def test_timeout_anthropic(self, client):
        """Subprocess exceeding timeout -> 504 for Anthropic endpoint."""
        configure(max_concurrent=10, request_timeout=0.05)

        proc = AsyncMock()
        proc.returncode = None
        proc.wait = AsyncMock(return_value=0)
        proc.kill = Mock()
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")

        async def _slow():
            await asyncio.sleep(100)
            yield b""

        proc.stdout = _slow()

        with patch(f"{MODULE}.asyncio.create_subprocess_exec", return_value=proc):
            resp = await client.post("/v1/messages", json={
                "model": "sonnet", "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
            })

        assert resp.status_code == 504
        assert resp.json()["error"]["type"] == "timeout_error"
        proc.kill.assert_called_once()


class TestSubprocessCleanup:
    @pytest.mark.anyio
    async def test_cleanup_kills_running_process(self):
        """_cleanup_process kills a subprocess that is still running."""
        proc = AsyncMock()
        proc.returncode = None
        proc.kill = Mock()
        proc.wait = AsyncMock()

        _server_mod._active_processes.add(proc)

        await _cleanup_process(proc)

        proc.kill.assert_called_once()
        proc.wait.assert_awaited_once()
        assert proc not in _server_mod._active_processes

    @pytest.mark.anyio
    async def test_cleanup_skips_finished_process(self):
        """_cleanup_process does not kill a completed subprocess."""
        proc = AsyncMock()
        proc.returncode = 0
        proc.kill = Mock()
        proc.wait = AsyncMock()

        _server_mod._active_processes.add(proc)

        await _cleanup_process(proc)

        proc.kill.assert_not_called()
        assert proc not in _server_mod._active_processes


class TestHealthLoadInfo:
    @pytest.mark.anyio
    async def test_reports_active_requests(self, client):
        """Health endpoint includes active_requests and max_concurrent."""
        configure(max_concurrent=42)
        resp = await client.get("/health")
        body = resp.json()
        assert body["active_requests"] == 0
        assert body["max_concurrent"] == 42


class TestConfigure:
    def test_sets_limits(self):
        """configure() updates module-level limits."""
        configure(max_concurrent=5, request_timeout=60.0)
        assert _server_mod._max_concurrent == 5
        assert _server_mod._request_timeout == 60.0

    @pytest.mark.anyio
    async def test_acquire_and_release(self):
        """Slots can be acquired up to the limit, then rejected."""
        configure(max_concurrent=2)
        assert await _server_mod._acquire_slot() is True
        assert await _server_mod._acquire_slot() is True
        assert await _server_mod._acquire_slot() is False  # full
        _server_mod._release_slot()
        assert await _server_mod._acquire_slot() is True  # freed one
        # Clean up acquired slots
        _server_mod._release_slot()
        _server_mod._release_slot()
