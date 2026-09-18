"""Deterministic guardrails that validate raw LLM output before it is trusted.

LLM output is untrusted structured data until it passes every check here.
Anything that fails is coerced to a safe no_op rather than crashing the
service or being silently applied to the optimizer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

REQUIRED_ADJUSTMENT_KEYS: Dict[str, set] = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
    "no_op": set(),
}


def _safe_no_op(note_index: int, explanation: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": explanation,
    }


def _valid_hours(hours: Any) -> Optional[List[int]]:
    if not isinstance(hours, list) or len(hours) == 0:
        return None
    try:
        ints = [int(h) for h in hours]
    except (TypeError, ValueError):
        return None
    if any(h < 0 or h > 23 for h in ints):
        return None
    if len(set(ints)) != len(ints):
        return None
    if ints != sorted(ints):
        return None
    return ints


def validate_directive(
    raw: Dict[str, Any],
    note_index: int,
    battery_capacity_kwh: float,
) -> Dict[str, Any]:
    """Validate one raw LLM directive entry. Returns a guaranteed-safe entry.

    Falls back to no_op on any structural, type, or range violation so a
    malformed LLM response can never reach the optimizer or crash the service.
    """
    if not isinstance(raw, dict):
        return _safe_no_op(note_index, "Malformed model output; treated as no_op.")

    directive_type = raw.get("directive_type")
    if directive_type not in ALLOWED_TYPES:
        return _safe_no_op(note_index, "Unsupported directive type; treated as no_op.")

    explanation = str(raw.get("explanation", ""))[:500]

    if directive_type == "no_op":
        return _safe_no_op(note_index, explanation or "Note does not affect the schedule.")

    adjustment = raw.get("structured_adjustment")
    if not isinstance(adjustment, dict):
        return _safe_no_op(note_index, "Missing structured_adjustment; treated as no_op.")

    required_keys = REQUIRED_ADJUSTMENT_KEYS[directive_type]
    if not required_keys.issubset(adjustment.keys()):
        return _safe_no_op(note_index, "Incomplete structured_adjustment; treated as no_op.")

    hours = _valid_hours(adjustment.get("hours"))
    if hours is None:
        return _safe_no_op(note_index, "Invalid hours array; treated as no_op.")

    clean_adjustment: Dict[str, Any] = {"hours": hours}

    if directive_type == "solar_reduction":
        factor = adjustment.get("factor")
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            return _safe_no_op(note_index, "Invalid factor; treated as no_op.")
        if not (0.0 <= factor <= 1.0):
            return _safe_no_op(note_index, "factor out of [0,1]; treated as no_op.")
        clean_adjustment["factor"] = factor

    elif directive_type == "minimum_battery_reserve":
        reserve = adjustment.get("minimum_energy_kwh")
        try:
            reserve = float(reserve)
        except (TypeError, ValueError):
            return _safe_no_op(note_index, "Invalid minimum_energy_kwh; treated as no_op.")
        if reserve < 0 or reserve != reserve or reserve in (float("inf"), float("-inf")):
            return _safe_no_op(note_index, "Non-finite/negative reserve; treated as no_op.")
        if reserve > battery_capacity_kwh:
            reserve = battery_capacity_kwh
        clean_adjustment["minimum_energy_kwh"] = reserve

    elif directive_type == "max_grid_window":
        max_grid = adjustment.get("max_grid_kwh")
        try:
            max_grid = float(max_grid)
        except (TypeError, ValueError):
            return _safe_no_op(note_index, "Invalid max_grid_kwh; treated as no_op.")
        if max_grid < 0 or max_grid != max_grid or max_grid in (float("inf"), float("-inf")):
            return _safe_no_op(note_index, "Non-finite/negative max_grid_kwh; treated as no_op.")
        clean_adjustment["max_grid_kwh"] = max_grid

    # no_charge_window / no_discharge_window need only validated hours.

    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": directive_type,
        "structured_adjustment": clean_adjustment,
        "explanation": explanation or f"Applied {directive_type}.",
    }


def validate_all(
    raw_entries: List[Dict[str, Any]],
    num_notes: int,
    battery_capacity_kwh: float,
) -> List[Dict[str, Any]]:
    """Ensure exactly one entry per note, in note_index order, all validated.

    Missing note indices become safe no_ops; duplicate indices keep the first
    occurrence and drop the rest; extra/out-of-range indices are discarded.
    """
    by_index: Dict[int, Dict[str, Any]] = {}
    for i, raw in enumerate(raw_entries or []):
        idx = raw.get("note_index", i) if isinstance(raw, dict) else i
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = i
        if idx < 0 or idx >= num_notes or idx in by_index:
            continue
        by_index[idx] = validate_directive(raw, idx, battery_capacity_kwh)

    result = []
    for idx in range(num_notes):
        if idx in by_index:
            result.append(by_index[idx])
        else:
            result.append(_safe_no_op(idx, "No model output for this note; treated as no_op."))
    return result
