"""Pure functions mapping validated directives to their deterministic effect
on the optimization model, per Problem Statement Section 5.3.

Shared by both the LP builder (model.py) and the final replay validator
(validator.py) so both always agree on what each directive means.
"""
from __future__ import annotations

from typing import List

from app.schemas import Battery, DirectiveInterpretation, HourEntry


def effective_solar(hours: List[HourEntry], directives: List[DirectiveInterpretation]) -> List[float]:
    """effective_solar[h] = original_solar[h] * factor for each solar_reduction hour."""
    effective = [h.solar_kwh for h in hours]
    for d in directives:
        if d.applies and d.directive_type == "solar_reduction" and d.structured_adjustment:
            factor = d.structured_adjustment.factor
            for hr in d.structured_adjustment.hours or []:
                effective[hr] = effective[hr] * factor
    return effective


def reserve_by_hour(
    base_minimum: float, capacity: float, hours: range, directives: List[DirectiveInterpretation]
) -> List[float]:
    """battery_energy_after[h] >= max(base_minimum, directive minimum) for each listed hour."""
    reserve = [base_minimum for _ in hours]
    for d in directives:
        if d.applies and d.directive_type == "minimum_battery_reserve" and d.structured_adjustment:
            level = min(d.structured_adjustment.minimum_energy_kwh, capacity)
            for hr in d.structured_adjustment.hours or []:
                reserve[hr] = max(reserve[hr], level)
    return reserve


def no_charge_hours(directives: List[DirectiveInterpretation]) -> set:
    """Hours where battery charge amount must be 0."""
    hrs: set = set()
    for d in directives:
        if d.applies and d.directive_type == "no_charge_window" and d.structured_adjustment:
            hrs.update(d.structured_adjustment.hours or [])
    return hrs


def no_discharge_hours(directives: List[DirectiveInterpretation]) -> set:
    """Hours where battery discharge amount must be 0."""
    hrs: set = set()
    for d in directives:
        if d.applies and d.directive_type == "no_discharge_window" and d.structured_adjustment:
            hrs.update(d.structured_adjustment.hours or [])
    return hrs


def max_grid_by_hour(hours: range, directives: List[DirectiveInterpretation]) -> List[float]:
    """grid_kwh[h] <= max_grid_kwh in the listed hours (tightest cap wins if overlapping)."""
    caps = [float("inf") for _ in hours]
    for d in directives:
        if d.applies and d.directive_type == "max_grid_window" and d.structured_adjustment:
            cap = d.structured_adjustment.max_grid_kwh
            for hr in d.structured_adjustment.hours or []:
                caps[hr] = min(caps[hr], cap)
    return caps
