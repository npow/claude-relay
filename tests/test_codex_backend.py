"""Focused tests for the Codex backend and Linux service configuration."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

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


@pytest.mark.anyio
async def test_read_codex_result():
    text, usage = await server._read_cli_result(_codex_process())
    assert text == "relay-ok"
    assert usage == {"input_tokens": 12, "output_tokens": 4}


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
