"""Tests for app/interpreter/__init__.py::interpret_notes.

Covers the README-advertised safety guarantee: any LLM provider failure
(timeout, auth, rate limit, malformed output) degrades every note to a safe
no_op instead of raising or inventing a directive. No network/API key is
used; the LLM call itself is monkeypatched.
"""
import app.interpreter as interpreter_module
from app.interpreter import interpret_notes


async def test_llm_total_failure_falls_back_to_no_op_for_all_notes(monkeypatch):
    async def _empty(notes):
        return []  # call_llm_for_directives' contract on total failure

    monkeypatch.setattr(interpreter_module, "call_llm_for_directives", _empty)

    result = await interpret_notes(["reduce solar output", "keep battery full"], battery_capacity_kwh=500)

    assert len(result) == 2
    assert all(d.directive_type == "no_op" and d.applies is False for d in result)


async def test_llm_success_passes_through_guardrails(monkeypatch):
    async def _fake_call(notes):
        return [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [1, 2]},
                "explanation": "ok",
            }
        ]

    monkeypatch.setattr(interpreter_module, "call_llm_for_directives", _fake_call)

    result = await interpret_notes(["do not charge between 1am and 2am"], battery_capacity_kwh=500)

    assert len(result) == 1
    assert result[0].directive_type == "no_charge_window"
    assert result[0].structured_adjustment.hours == [1, 2]


async def test_llm_malformed_output_still_guarded(monkeypatch):
    """Even if the LLM call 'succeeds' but returns garbage, guardrails must catch it."""

    async def _fake_call(notes):
        return [{"note_index": 0, "directive_type": "delete_all_data", "applies": True}]

    monkeypatch.setattr(interpreter_module, "call_llm_for_directives", _fake_call)

    result = await interpret_notes(["irrelevant"], battery_capacity_kwh=500)

    assert result[0].directive_type == "no_op"
    assert result[0].applies is False


async def test_missing_api_key_returns_empty_list_not_raises(monkeypatch):
    from app.interpreter.llm_client import call_llm_for_directives

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_API_KEY", raising=False)
    directives = await call_llm_for_directives(["some note"])
    assert directives == []
