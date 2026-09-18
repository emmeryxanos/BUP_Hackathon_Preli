"""Operator-note interpretation: LLM call + deterministic guardrail validation."""
from __future__ import annotations

from typing import List

from app.interpreter.guardrails import validate_all
from app.interpreter.llm_client import call_llm_for_directives
from app.schemas import DirectiveInterpretation


async def interpret_notes(
    operator_notes: List[str], battery_capacity_kwh: float
) -> List[DirectiveInterpretation]:
    """Interpret operator notes into validated directive entries.

    The LLM produces raw candidate directives; guardrails.validate_all makes
    the result safe regardless of what the model returned. call_llm_for_directives
    never raises -- if the LLM is unavailable it returns an empty list, and
    validate_all pads that out to a safe no_op for every note rather than the
    service crashing or inventing a rule.
    """
    raw_directives = await call_llm_for_directives(operator_notes)
    validated = validate_all(raw_directives, len(operator_notes), battery_capacity_kwh)
    return [DirectiveInterpretation(**entry) for entry in validated]
