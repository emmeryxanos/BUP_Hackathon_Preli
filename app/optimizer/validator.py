"""Final self-validation: independently replay the produced schedule.

This mirrors the checks the hidden judge harness will run (Section 9 and 11
of the Problem Statement) so a broken plan is caught before it is ever sent
back to the caller, instead of failing silently in the judge.
"""
from __future__ import annotations

from typing import List

from app.optimizer.directives import (
    effective_solar,
    max_grid_by_hour,
    no_charge_hours,
    no_discharge_hours,
    reserve_by_hour,
)
from app.optimizer.model import OptimizationResult
from app.schemas import Battery, DirectiveInterpretation, HourEntry

TOL = 0.01  # absolute tolerance per the Problem Statement, Section 11.5


class ScheduleValidationError(Exception):
    pass


def replay_and_validate(
    hours: List[HourEntry],
    battery: Battery,
    directives: List[DirectiveInterpretation],
    result: OptimizationResult,
) -> None:
    hours_sorted = sorted(hours, key=lambda h: h.hour)
    n = len(hours_sorted)
    idx_range = range(n)

    if len(result.hourly_plan) != n:
        raise ScheduleValidationError("hourly_plan must contain exactly 24 entries")

    plan_by_hour = {p.hour: p for p in result.hourly_plan}
    if set(plan_by_hour.keys()) != set(range(n)):
        raise ScheduleValidationError("hourly_plan hours must be exactly 0..23, unique")

    eff_solar = effective_solar(hours_sorted, directives)
    reserve = reserve_by_hour(battery.minimum_energy_kwh, battery.capacity_kwh, idx_range, directives)
    no_charge = no_charge_hours(directives)
    no_discharge = no_discharge_hours(directives)
    max_grid = max_grid_by_hour(idx_range, directives)

    prev_energy = battery.initial_energy_kwh
    for h in idx_range:
        p = plan_by_hour[h]

        if p.grid_kwh < -TOL or p.solar_used_kwh < -TOL or p.battery_kwh < -TOL:
            raise ScheduleValidationError(f"hour {h}: negative energy value")

        if p.solar_used_kwh > eff_solar[h] + TOL:
            raise ScheduleValidationError(f"hour {h}: solar_used exceeds effective solar")

        if p.battery_action not in ("charge", "discharge", "idle"):
            raise ScheduleValidationError(f"hour {h}: invalid battery_action")

        if p.battery_action == "idle" and p.battery_kwh > TOL:
            raise ScheduleValidationError(f"hour {h}: idle hour must have battery_kwh 0")

        charge_kwh = p.battery_kwh if p.battery_action == "charge" else 0.0
        discharge_kwh = p.battery_kwh if p.battery_action == "discharge" else 0.0

        if charge_kwh > battery.max_charge_kwh_per_hour + TOL:
            raise ScheduleValidationError(f"hour {h}: exceeds max_charge_kwh_per_hour")
        if discharge_kwh > battery.max_discharge_kwh_per_hour + TOL:
            raise ScheduleValidationError(f"hour {h}: exceeds max_discharge_kwh_per_hour")

        if h in no_charge and charge_kwh > TOL:
            raise ScheduleValidationError(f"hour {h}: charging during no_charge_window")
        if h in no_discharge and discharge_kwh > TOL:
            raise ScheduleValidationError(f"hour {h}: discharging during no_discharge_window")

        if max_grid[h] != float("inf") and p.grid_kwh > max_grid[h] + TOL:
            raise ScheduleValidationError(f"hour {h}: grid_kwh exceeds max_grid_window cap")

        expected_after = prev_energy + charge_kwh - discharge_kwh
        if abs(expected_after - p.battery_energy_after_kwh) > TOL:
            raise ScheduleValidationError(f"hour {h}: battery state transition mismatch")

        if p.battery_energy_after_kwh < reserve[h] - TOL:
            raise ScheduleValidationError(f"hour {h}: battery below required reserve")
        if p.battery_energy_after_kwh > battery.capacity_kwh + TOL:
            raise ScheduleValidationError(f"hour {h}: battery exceeds capacity")

        lhs = p.grid_kwh + p.solar_used_kwh + discharge_kwh
        rhs = hours_sorted[h].demand_kwh + charge_kwh
        if abs(lhs - rhs) > TOL:
            raise ScheduleValidationError(f"hour {h}: energy balance equation violated")

        prev_energy = p.battery_energy_after_kwh

    if abs(prev_energy - battery.initial_energy_kwh) > TOL:
        raise ScheduleValidationError("end-of-day battery neutrality violated")

    recalculated_cost = sum(
        plan_by_hour[h].grid_kwh * hours_sorted[h].tariff_bdt_per_kwh for h in idx_range
    )
    if abs(recalculated_cost - result.total_cost_bdt) > TOL:
        raise ScheduleValidationError("total_cost_bdt does not match recalculated cost")

    recalculated_grid = sum(plan_by_hour[h].grid_kwh for h in idx_range)
    if abs(recalculated_grid - result.total_grid_kwh) > TOL:
        raise ScheduleValidationError("total_grid_kwh does not match recalculated sum")

    recalculated_peak = max(plan_by_hour[h].grid_kwh for h in idx_range)
    if abs(recalculated_peak - result.peak_grid_kwh) > TOL:
        raise ScheduleValidationError("peak_grid_kwh does not match recalculated max")
