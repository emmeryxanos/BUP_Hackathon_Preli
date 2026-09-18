"""Tests for app/interpreter/llm_client.py (AgentRouter, OpenAI-compatible over httpx).

httpx.AsyncClient.post is mocked throughout, so no network access or real API
key is used. Covers the primary call, retry-then-fallback-key/model behavior,
and response parsing/error handling for the OpenAI-style tool-calling
response shape. call_llm_for_directives never raises: on total failure it
returns an empty list rather than propagating an exception (the caller,
app.interpreter.interpret_notes, treats an empty list as "pad every note
with no_op").
"""
import json

import httpx
import pytest

import app.interpreter.llm_client as llm_client
from app.interpreter.llm_client import call_llm_for_directives


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-primary-fake-key")
    monkeypatch.delenv("LLM_FALLBACK_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "LLM_MAX_RETRIES", 1)


def _tool_call_response(directives, name="interpret_notes"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps({"directives": directives}),
                                }
                            }
                        ]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://agentrouter.org/v1/chat/completions"),
    )


def _no_tool_call_response():
    return httpx.Response(
        200,
        json={"choices": [{"message": {}}]},
        request=httpx.Request("POST", "https://agentrouter.org/v1/chat/completions"),
    )


def _patch_post(monkeypatch, fn):
    """Patch httpx.AsyncClient.post with an async function `fn(self, url, **kw)`."""

    async def _wrapper(self, url, **kwargs):
        return fn(url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", _wrapper)


async def test_call_llm_for_directives_returns_tool_input(monkeypatch):
    resp = _tool_call_response([{"note_index": 0, "directive_type": "no_op"}])
    _patch_post(monkeypatch, lambda *a, **kw: resp)

    directives = await call_llm_for_directives(["a note"])
    assert directives == [{"note_index": 0, "directive_type": "no_op"}]


async def test_call_llm_for_directives_sends_bearer_auth_header(monkeypatch):
    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        captured["url"] = url
        return _tool_call_response([{"note_index": 0, "directive_type": "no_op"}])

    _patch_post(monkeypatch, _fake_post)
    await call_llm_for_directives(["a note"])

    assert captured["headers"]["Authorization"] == "Bearer sk-primary-fake-key"
    assert captured["url"] == "https://agentrouter.org/v1/chat/completions"


async def test_missing_primary_key_returns_empty_list(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    directives = await call_llm_for_directives(["a note"])
    assert directives == []


async def test_no_tool_call_in_response_returns_empty_list(monkeypatch):
    _patch_post(monkeypatch, lambda *a, **kw: _no_tool_call_response())
    directives = await call_llm_for_directives(["a note"])
    assert directives == []


async def test_directives_not_a_list_returns_empty_list(monkeypatch):
    resp = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "interpret_notes", "arguments": json.dumps({"directives": "oops"})}}
                        ]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://agentrouter.org/v1/chat/completions"),
    )
    _patch_post(monkeypatch, lambda *a, **kw: resp)
    directives = await call_llm_for_directives(["a note"])
    assert directives == []


async def test_http_error_retries_then_returns_empty_list(monkeypatch):
    call_count = {"n": 0}

    def _fake_post(*a, **kw):
        call_count["n"] += 1
        raise httpx.ConnectError("connection refused")

    _patch_post(monkeypatch, _fake_post)
    directives = await call_llm_for_directives(["a note"])

    assert directives == []
    # LLM_MAX_RETRIES=1 means 2 primary attempts (no fallback key configured).
    assert call_count["n"] == 2


async def test_primary_failure_falls_back_to_fallback_key(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "sk-fallback-fake-key")
    calls = []

    def _fake_post(url, json=None, headers=None, timeout=None):
        calls.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer sk-primary-fake-key":
            raise httpx.ConnectError("primary down")
        return _tool_call_response([{"note_index": 0, "directive_type": "no_op"}])

    _patch_post(monkeypatch, _fake_post)
    directives = await call_llm_for_directives(["a note"])

    assert directives == [{"note_index": 0, "directive_type": "no_op"}]
    assert calls[-1] == "Bearer sk-fallback-fake-key"
    # Primary attempted LLM_MAX_RETRIES+1 = 2 times before falling back.
    assert calls.count("Bearer sk-primary-fake-key") == 2


async def test_both_primary_and_fallback_failing_returns_empty_list(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "sk-fallback-fake-key")

    def _fake_post(*a, **kw):
        raise httpx.ConnectError("everything is down")

    _patch_post(monkeypatch, _fake_post)
    directives = await call_llm_for_directives(["a note"])
    assert directives == []


async def test_malformed_completion_shape_returns_empty_list(monkeypatch):
    resp = httpx.Response(
        200, json={"unexpected": "shape"}, request=httpx.Request("POST", "https://agentrouter.org/v1/chat/completions")
    )
    _patch_post(monkeypatch, lambda *a, **kw: resp)
    directives = await call_llm_for_directives(["a note"])
    assert directives == []
