"""Deterministic guardrail validation for LLM-produced directive interpretations.

Everything the LLM returns is untrusted until it passes here (Problem
Statement Section 08). This module never trusts note_index ordering, never
trusts numeric ranges, and never lets an unsupported directive type reach the
optimizer -- if a candidate interpretation fails validation it is safely
downgraded to no_op rather than raising, so a malformed/hallucinated model
response degrades gracefully instead of crashing the service.
"""
import math
from typing import Any, Dict, List

from app.directives import (
    ALL_DIRECTIVE_TYPES,
    MAX_GRID_WINDOW,
    MINIMUM_BATTERY_RESERVE,
    NO_OP,
    SOLAR_REDUCTION,
)
from app.schemas import DirectiveInterpretationEntry


def _safe_no_op(note_index: int, reason: str) -> DirectiveInterpretationEntry:
    return DirectiveInterpretationEntry(
        note_index=note_index,
        applies=False,
        directive_type=NO_OP,
        structured_adjustment=None,
        explanation=reason,
    )


def _normalize_hours(raw_hours: Any) -> List[int] | None:
    """Return a sorted list of unique ints in [0, 23], or None if invalid/empty."""
    if not isinstance(raw_hours, list) or len(raw_hours) == 0:
        return None
    try:
        hours = {int(h) for h in raw_hours}
    except (TypeError, ValueError):
        return None
    if any(h < 0 or h > 23 for h in hours):
        return None
    return sorted(hours)


def _finite_non_negative(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 0:
        return None
    return f


def validate_interpretation(
    raw_entries: List[Dict[str, Any]],
    num_notes: int,
    battery_capacity_kwh: float,
) -> List[DirectiveInterpretationEntry]:
    """Validate and repair raw LLM output into exactly one entry per note.

    Guarantees, regardless of what the model returned:
    - Exactly `num_notes` entries, in note_index order 0..num_notes-1.
    - directive_type is one of the six supported values.
    - applies is False iff directive_type == no_op.
    - structured_adjustment matches the required shape for its directive
      type, or is None for no_op.
    """
    by_index: Dict[int, DirectiveInterpretationEntry] = {}

    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        try:
            note_index = int(raw.get("note_index"))
        except (TypeError, ValueError):
            continue
        if note_index < 0 or note_index >= num_notes or note_index in by_index:
            continue

        directive_type = raw.get("directive_type")
        explanation = str(raw.get("explanation") or "")[:500]

        if directive_type not in ALL_DIRECTIVE_TYPES:
            by_index[note_index] = _safe_no_op(
                note_index, "guardrail: unsupported directive_type rejected"
            )
            continue

        if directive_type == NO_OP:
            by_index[note_index] = DirectiveInterpretationEntry(
                note_index=note_index,
                applies=False,
                directive_type=NO_OP,
                structured_adjustment=None,
                explanation=explanation or "Note does not affect the energy schedule.",
            )
            continue

        hours = _normalize_hours(raw.get("hours"))
        if hours is None:
            by_index[note_index] = _safe_no_op(
                note_index, "guardrail: missing/invalid hours for a non-no_op directive"
            )
            continue

        adjustment: Dict[str, Any] | None = None

        if directive_type == SOLAR_REDUCTION:
            try:
                factor = float(raw.get("factor"))
            except (TypeError, ValueError):
                factor = None
            if factor is None or not (0.0 <= factor <= 1.0) or not math.isfinite(factor):
                by_index[note_index] = _safe_no_op(
                    note_index, "guardrail: invalid solar_reduction factor"
                )
                continue
            adjustment = {"hours": hours, "factor": factor}

        elif directive_type == MINIMUM_BATTERY_RESERVE:
            reserve = _finite_non_negative(raw.get("minimum_energy_kwh"))
            if reserve is None or reserve > battery_capacity_kwh:
                by_index[note_index] = _safe_no_op(
                    note_index, "guardrail: invalid minimum_battery_reserve value"
                )
                continue
            adjustment = {"hours": hours, "minimum_energy_kwh": reserve}

        elif directive_type == MAX_GRID_WINDOW:
            max_grid = _finite_non_negative(raw.get("max_grid_kwh"))
            if max_grid is None:
                by_index[note_index] = _safe_no_op(
                    note_index, "guardrail: invalid max_grid_kwh value"
                )
                continue
            adjustment = {"hours": hours, "max_grid_kwh": max_grid}

        else:  # no_charge_window / no_discharge_window
            adjustment = {"hours": hours}

        by_index[note_index] = DirectiveInterpretationEntry(
            note_index=note_index,
            applies=True,
            directive_type=directive_type,
            structured_adjustment=adjustment,
            explanation=explanation or f"Applied {directive_type}.",
        )

    return [
        by_index.get(i)
        or _safe_no_op(i, "guardrail: no interpretation returned by the model for this note")
        for i in range(num_notes)
    ]
