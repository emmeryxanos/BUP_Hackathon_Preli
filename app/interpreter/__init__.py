"""Operator-note interpretation: LLM call + deterministic guardrail validation."""
from __future__ import annotations

import logging
from typing import List

from app.interpreter.guardrails import validate_all
from app.interpreter.llm_client import LLMUnavailableError, call_llm_for_directives
from app.schemas import DirectiveInterpretation

logger = logging.getLogger("gridwise.interpreter")


def interpret_notes(
    operator_notes: List[str], battery_capacity_kwh: float
) -> List[DirectiveInterpretation]:
    """Interpret operator notes into validated directive entries.

    The LLM produces raw candidate directives; guardrails.validate_all makes
    the result safe regardless of what the model returned. If the LLM is
    unavailable, every note safely falls back to no_op rather than the
    service crashing or inventing a rule.
    """
    try:
        raw_directives = call_llm_for_directives(operator_notes)
    except LLMUnavailableError:
        logger.exception("LLM interpretation failed; falling back to no_op for all notes")
        raw_directives = []

    validated = validate_all(raw_directives, len(operator_notes), battery_capacity_kwh)
    return [DirectiveInterpretation(**entry) for entry in validated]
