"""Focused tests for the Codex backend and Linux service configuration."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from claude_relay import server
from claude_relay.service import _env_lines, _generate_systemd_unit


def _codex_process(text: str = "relay-ok"):
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "item-1", "type": "agent_message", "text": text},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 12, "output_tokens": 4},
        },
    ]
    process = AsyncMock()
    process.returncode = 0
    process.wait = AsyncMock(return_value=0)

    async def stdout():
        for event in events:
            yield json.dumps(event).encode()

    process.stdout = stdout()
    return process


def _app_server_process(text_chunks=None):
    text_chunks = text_chunks or ["app", "-stream", "-ok"]
    output_text = "".join(text_chunks)
    messages = [
        {"id": 1, "result": {"userAgent": "agent-relay-test"}},
        {"method": "mcpServer/startupStatus/updated", "params": {}},
        {"id": 2, "result": {"thread": {"id": "thread-1"}}},
        {"id": 3, "result": {"turn": {"id": "turn-1"}}},
    ]
    messages.extend({
        "method": "item/agentMessage/delta",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "item-1",
            "delta": text,
        },
    } for text in text_chunks)
    messages.extend([
        {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "agentMessage", "text": output_text},
            },
        },
        {
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "tokenUsage": {
                    "last": {"inputTokens": 12, "outputTokens": 7},
                },
            },
        },
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {"id": "turn-1", "status": "completed"},
            },
        },
    ])

    reader = asyncio.StreamReader(limit=server._subprocess_stream_limit)
    for message in messages:
        reader.feed_data(json.dumps(message).encode() + b"\n")
    reader.feed_eof()

    process = Mock()
    process.returncode = 0
    process.wait = AsyncMock(return_value=0)
    process.kill = Mock()
    process.stdin = Mock()
    process.stdin.drain = AsyncMock()
    process.stdin.can_write_eof = Mock(return_value=True)
    process.stdout = reader
    process.stderr = Mock()
    process.stderr.read = AsyncMock(return_value=b"")
    return process


@pytest.mark.anyio
async def test_read_codex_result():
    text, usage = await server._read_cli_result(_codex_process())
    assert text == "relay-ok"
    assert usage == {"input_tokens": 12, "output_tokens": 4}


@pytest.mark.anyio
async def test_app_server_protocol_start_and_result():
    process = _app_server_process()
    with (
        patch.object(server, "_backend", "codex"),
        patch.object(server, "_codex_protocol", "app-server"),
    ):
        await server._prepare_cli_process(
            process,
            "Reply exactly",
            "gpt-5.6-sol",
            "/work/project",
        )
        text, usage = await server._read_cli_result(process)

    requests = [
        json.loads(call.args[0])
        for call in process.stdin.write.call_args_list
    ]
    assert [request["method"] for request in requests] == [
        "initialize",
        "initialized",
        "thread/start",
        "turn/start",
    ]
    assert requests[2]["params"]["model"] == "gpt-5.6-sol"
    assert requests[2]["params"]["cwd"] == "/work/project"
    assert text == "app-stream-ok"
    assert usage == {"input_tokens": 12, "output_tokens": 7}


@pytest.mark.anyio
async def test_anthropic_endpoint_preserves_app_server_deltas():
    process = _app_server_process()
    async with AsyncClient(
        transport=ASGITransport(app=server.app),
        base_url="http://test",
    ) as client:
        with (
            patch.object(server, "_backend", "codex"),
            patch.object(server, "_codex_protocol", "app-server"),
            patch.object(
                server.asyncio,
                "create_subprocess_exec",
                return_value=process,
            ),
        ):
            response = await client.post("/v1/messages", json={
                "model": "gpt-5.6-sol",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            })

    deltas = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event.get("type") == "content_block_delta":
            deltas.append(event["delta"]["text"])
    assert deltas == ["app", "-stream", "-ok"]


def test_codex_environment_uses_subscription_auth(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-be-forwarded")
    with patch.object(server, "_backend", "codex"):
        env = server._subprocess_environment()
    assert "OPENAI_API_KEY" not in env
    assert "CODEX_API_KEY" not in env


def test_large_codex_message_is_split_without_data_loss():
    text = "x" * 600
    chunks = list(server._stream_text_chunks(text))
    assert [len(chunk) for chunk in chunks] == [256, 256, 88]
    assert "".join(chunks) == text


@pytest.mark.anyio
async def test_stream_reader_limit_accepts_large_jsonl_event():
    reader = asyncio.StreamReader(limit=server._subprocess_stream_limit)
    reader.feed_data(json.dumps({"text": "x" * 70000}).encode() + b"\n")
    reader.feed_eof()
    line = await reader.readline()
    assert len(line) > 64 * 1024
    assert json.loads(line)["text"] == "x" * 70000


def test_systemd_unit_selects_codex_subscription_model():
    with patch("claude_relay.service._find_executable", return_value="/opt/agent-relay"):
        unit = _generate_systemd_unit(
            "127.0.0.1",
            18082,
            backend="codex",
            model="gpt-5.6-sol",
            working_directory=Path("/work/project"),
        )
    assert "--backend codex --model gpt-5.6-sol" in unit
    assert "OPENAI_API_KEY" not in unit
    assert "WantedBy=default.target" in unit
    assert "WorkingDirectory=/work/project" in unit


def test_claude_code_environment_targets_gateway():
    lines = _env_lines(
        "127.0.0.1",
        18082,
        backend="codex",
        model="gpt-5.6-sol",
    )
    assert 'export ANTHROPIC_BASE_URL="http://127.0.0.1:18082"' in lines
    assert "unset ANTHROPIC_API_KEY" in lines
    assert "unset ANTHROPIC_AUTH_TOKEN" in lines
    assert 'export ANTHROPIC_MODEL="gpt-5.6-sol"' in lines
