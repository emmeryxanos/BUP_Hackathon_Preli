"""LLM interpretation layer.

This module is the ONLY place that talks to the language model. Its output
is treated as untrusted structured data: guardrails.py re-validates every
field before anything reaches the optimizer, per Problem Statement Section
08. Structured Outputs (output_config.format = json_schema) is used so the
model is constrained to emit a schema-valid JSON object -- this is what
keeps the extraction machine-checkable across paraphrases instead of relying
on free-text parsing.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List

import anthropic

from app.config import LLM_MAX_ATTEMPTS, LLM_MAX_TOKENS, LLM_MODEL, LLM_TIMEOUT_SECONDS
from app.directives import ALL_DIRECTIVE_TYPES

logger = logging.getLogger("gridwise.llm")

_client = anthropic.AsyncAnthropic(timeout=LLM_TIMEOUT_SECONDS)

SYSTEM_PROMPT = """You are the operator-note interpretation module of a campus energy \
optimization system (GridWise). You convert short natural-language notes from campus \
operators into structured directives for a downstream math optimizer.

Supported directive types (use exactly these strings, nothing else):
- solar_reduction: usable solar drops during specific hours. Needs hours + factor \
(factor = the FRACTION OF SOLAR THAT REMAINS, e.g. an 80% reduction means factor=0.2).
- minimum_battery_reserve: battery energy must stay at or above a level during specific \
hours. Needs hours + minimum_energy_kwh.
- no_charge_window: battery charging is unavailable during specific hours. Needs hours.
- no_discharge_window: battery discharging is unavailable during specific hours. Needs hours.
- max_grid_window: grid import may not exceed a stated amount during specific hours. \
Needs hours + max_grid_kwh.
- no_op: the note does not affect the 24-hour energy schedule (distractor, unrelated \
topic, or something already outside the system's control). No numeric fields apply.

Rules:
- Every note maps to exactly one directive type. If it is not clearly one of the five \
operational directive types above, use no_op -- never invent a new type.
- "hours" is a list of whole integer hours 0-23, ascending, no duplicates. Time windows \
are half-open: "1 PM to 3 PM" means hours [13, 14] (the end hour is EXCLUDED).
- Notes may paraphrase the same underlying rule in many ways (percentages, fractions, \
clock times, 24-hour times, indirect phrasing). Focus on the underlying operational \
meaning, not exact wording. For example, "PV production will drop to about 20% between \
13:00 and 15:00", "Panel washing from one until three will leave roughly one-fifth of \
normal solar output", and "Expect an 80% reduction in rooftop solar during the 1-3 PM \
maintenance window" all mean: solar_reduction, hours [13,14], factor 0.2.
- Never invent or restate specific demand, tariff, or battery-capacity numbers that were \
not given to you as directive parameters -- you only extract the directive itself.
- Return exactly one interpretation object per input note, using the same note_index \
values you were given, in ascending order.
- For directive types other than no_op, fill in only the fields that type needs; leave \
the other numeric fields null and hours as required. For no_op, hours must be an empty \
list and all numeric fields null.
- explanation should be a short (<20 words) human-readable reason for your interpretation.
"""

INTERPRETATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "applies": {"type": "boolean"},
                    "directive_type": {
                        "type": "string",
                        "enum": sorted(ALL_DIRECTIVE_TYPES),
                    },
                    "hours": {"type": "array", "items": {"type": "integer"}},
                    "factor": {"type": ["number", "null"]},
                    "minimum_energy_kwh": {"type": ["number", "null"]},
                    "max_grid_kwh": {"type": ["number", "null"]},
                    "explanation": {"type": "string"},
                },
                "required": [
                    "note_index",
                    "applies",
                    "directive_type",
                    "hours",
                    "factor",
                    "minimum_energy_kwh",
                    "max_grid_kwh",
                    "explanation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interpretations"],
    "additionalProperties": False,
}


def _build_user_message(operator_notes: List[str]) -> str:
    numbered = [{"note_index": i, "text": note} for i, note in enumerate(operator_notes)]
    return (
        "Interpret each of the following operator notes for today's 24-hour schedule.\n"
        f"{json.dumps(numbered, ensure_ascii=False)}"
    )


async def interpret_notes(operator_notes: List[str]) -> List[Dict[str, Any]]:
    """Call the LLM to interpret operator notes.

    Returns a list of raw (untrusted) interpretation dicts, one attempted per
    note. On any provider/parsing failure after retries, returns a safe
    fallback: every note marked no_op. This function never raises -- a
    malformed or unavailable model must not crash the service (Problem
    Statement Section 08, "SAFE FAILURE").
    """
    last_error: Exception | None = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            response = await _client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": _build_user_message(operator_notes)}],
                output_config={
                    "format": {"type": "json_schema", "schema": INTERPRETATION_JSON_SCHEMA}
                },
            )
            text = next(block.text for block in response.content if block.type == "text")
            parsed = json.loads(text)
            interpretations = parsed.get("interpretations", [])
            if isinstance(interpretations, list) and interpretations:
                return interpretations
            last_error = ValueError("LLM returned an empty interpretations array")
        except (anthropic.APIError, anthropic.APIConnectionError, asyncio.TimeoutError) as exc:
            last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, LLM_MAX_ATTEMPTS, exc)
        except (StopIteration, json.JSONDecodeError, KeyError, TypeError) as exc:
            last_error = exc
            logger.warning(
                "LLM returned unparseable output (attempt %d/%d): %s",
                attempt,
                LLM_MAX_ATTEMPTS,
                exc,
            )

    logger.error(
        "All %d LLM interpretation attempts failed; falling back to no_op for all notes. "
        "Last error: %s",
        LLM_MAX_ATTEMPTS,
        last_error,
    )
    return [
        {
            "note_index": i,
            "applies": False,
            "directive_type": "no_op",
            "hours": [],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "LLM interpretation unavailable; safe fallback applied.",
        }
        for i in range(len(operator_notes))
    ]
