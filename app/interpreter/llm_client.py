"""LLM client for operator-note interpretation.

Uses the Anthropic API (Claude) with tool-use for reliable structured output.
The LLM is squarely on the interpretation critical path, satisfying the
challenge's mandatory LLM requirement. All output is treated as untrusted
and passed through app.interpreter.guardrails before use.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from app.interpreter.prompts import SYSTEM_PROMPT, TOOL_SCHEMA, build_user_message

logger = logging.getLogger("gridwise.llm")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "20"))


class LLMUnavailableError(Exception):
    """Raised when the LLM call fails or returns unusable output."""


def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMUnavailableError("ANTHROPIC_API_KEY is not configured")
    import anthropic  # imported lazily so the module loads without the SDK installed

    return anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)


def call_llm_for_directives(operator_notes: List[str]) -> List[Dict[str, Any]]:
    """Call the LLM once and return its raw (untrusted) directive list.

    Raises LLMUnavailableError on any provider/network/parsing failure so the
    caller can apply a safe no_op fallback instead of crashing the service.
    """
    client = _get_client()
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "interpret_notes"},
            messages=[{"role": "user", "content": build_user_message(operator_notes)}],
        )
    except Exception as exc:  # network, auth, rate-limit, timeout, etc.
        raise LLMUnavailableError(f"LLM call failed: {exc}") from exc

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "interpret_notes":
            directives = block.input.get("directives")
            if isinstance(directives, list):
                return directives
            raise LLMUnavailableError("LLM tool output missing 'directives' list")

    raise LLMUnavailableError("LLM did not return a tool_use block")


def call_llm_raw_text_fallback(operator_notes: List[str]) -> List[Dict[str, Any]]:
    """Secondary path for providers/models without tool-use: ask for raw JSON."""
    client = _get_client()
    prompt = (
        SYSTEM_PROMPT
        + "\n\nRespond with ONLY a JSON object of the form "
        + '{"directives": [...]} and no other text.\n\n'
        + build_user_message(operator_notes)
    )
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        parsed = json.loads(text)
        directives = parsed.get("directives")
        if isinstance(directives, list):
            return directives
        raise ValueError("missing 'directives' key")
    except Exception as exc:
        raise LLMUnavailableError(f"LLM raw-text fallback failed: {exc}") from exc
