"""LLM client for operator-note interpretation.

Calls an OpenAI-compatible chat-completions endpoint via httpx.AsyncClient,
with tool-calling for reliable structured output. The LLM is squarely on the
interpretation critical path, satisfying the challenge's mandatory LLM
requirement. All output is treated as untrusted and passed through
app.interpreter.guardrails before use.

A primary key/model is tried first (LLM_MAX_RETRIES attempts on top of the
initial one); if every primary attempt fails, a fallback key/model is tried
once. This function never raises: on total failure it returns an empty
directive list, letting the caller (app.interpreter.interpret_notes) apply
guardrails.validate_all's safe no_op fallback for every note rather than the
service crashing (Problem Statement Section 08, "SAFE FAILURE").
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import httpx

from app.config import LLM_BASE_URL, LLM_FALLBACK_MODEL, LLM_MAX_RETRIES, LLM_MODEL, LLM_REQUEST_TIMEOUT_SECONDS
from app.interpreter.prompts import SYSTEM_PROMPT, TOOL_SCHEMA, build_user_message

logger = logging.getLogger("gridwise.llm")

LLM_API_KEY_ENV = "OPENAI_API_KEY"
LEGACY_LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_FALLBACK_API_KEY_ENV = "LLM_FALLBACK_API_KEY"

_OPENAI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_SCHEMA["name"],
        "description": TOOL_SCHEMA["description"],
        "parameters": TOOL_SCHEMA["input_schema"],
    },
}


class LLMUnavailableError(Exception):
    """Raised internally between attempts; never escapes call_llm_for_directives."""


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Lazily create and reuse one AsyncClient for the process lifetime.

    Creating/closing a fresh httpx.AsyncClient per call is not just wasteful
    (loses connection pooling); on Windows' default ProactorEventLoop it can
    also trigger native access-violation crashes when combined with
    TestClient's sync-to-async thread bridging. A single long-lived client
    avoids both problems.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


def _extract_directives_from_completion(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        choice = data["choices"][0]
        tool_calls = choice["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailableError(f"Malformed completion response: {exc}") from exc

    for call in tool_calls:
        if call.get("function", {}).get("name") == TOOL_SCHEMA["name"]:
            try:
                arguments = json.loads(call["function"]["arguments"])
            except (KeyError, ValueError) as exc:
                raise LLMUnavailableError(f"Malformed tool-call arguments: {exc}") from exc
            directives = arguments.get("directives")
            if isinstance(directives, list):
                return directives
            raise LLMUnavailableError("LLM tool output missing 'directives' list")

    raise LLMUnavailableError("LLM did not return a tool call")


async def _call_completion(
    client: httpx.AsyncClient, api_key: str, model: str, operator_notes: List[str]
) -> List[Dict[str, Any]]:
    if not api_key:
        raise LLMUnavailableError(f"{LLM_API_KEY_ENV} is not configured")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(operator_notes)},
        ],
        "tools": [_OPENAI_TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": TOOL_SCHEMA["name"]}},
        "max_tokens": 2048,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = await client.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise LLMUnavailableError(f"LLM call failed: {exc}") from exc

    return _extract_directives_from_completion(data)


async def call_llm_for_directives(operator_notes: List[str]) -> List[Dict[str, Any]]:
    """Call the LLM (with retries, then a fallback key/model).

    Returns the raw (untrusted) directive list on success, or an empty list
    if the primary key/model exhausted its retries and the fallback (if
    configured) also failed -- this function never raises.
    """
    primary_key = os.environ.get(LLM_API_KEY_ENV) or os.environ.get(LEGACY_LLM_API_KEY_ENV)
    fallback_key = os.environ.get(LLM_FALLBACK_API_KEY_ENV)

    last_error: Exception | None = None
    attempts = max(1, LLM_MAX_RETRIES + 1)
    client = _get_client()

    for attempt in range(1, attempts + 1):
        try:
            return await _call_completion(client, primary_key, LLM_MODEL, operator_notes)
        except LLMUnavailableError as exc:
            last_error = exc
            logger.warning("Primary LLM call attempt %d/%d failed: %s", attempt, attempts, exc)

    if fallback_key:
        try:
            return await _call_completion(client, fallback_key, LLM_FALLBACK_MODEL, operator_notes)
        except LLMUnavailableError as exc:
            last_error = exc
            logger.warning("Fallback LLM call failed: %s", exc)

    logger.error(
        "All LLM interpretation attempts failed; falling back to no_op for all notes. "
        "Last error: %s",
        last_error,
    )
    return []
