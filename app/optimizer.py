"""Directive application and LP-based cost optimization.

Two responsibilities live here, matching Problem Statement Sections 05.3 and
09:

1. `apply_directives` turns the guardrail-validated directive list into the
   deterministic per-hour effective solar / battery-window / reserve / grid
   -cap arrays the optimizer must respect.
2. `solve_schedule` builds a linear program over 24 hours and solves it with
   SciPy's HiGHS solver (pure library call, no external solver binary, which
   keeps the Docker image small and avoids solver-availability flakiness).

Per-hour battery activity is modeled as a single signed variable
`delta[h]` (positive = charge, negative = discharge) instead of separate
charge/discharge variables, which makes "idle" and "no simultaneous
charge+discharge" automatic rather than something to police afterwards.
"""
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.optimize import linprog

from app.directives import (
    MAX_GRID_WINDOW,
    MINIMUM_BATTERY_RESERVE,
    NO_CHARGE_WINDOW,
    NO_DISCHARGE_WINDOW,
    SOLAR_REDUCTION,
)
from app.schemas import BatteryConfig, DirectiveInterpretationEntry, HourEntry, HourlyPlanEntry

TOLERANCE = 0.01


@dataclass
class DirectiveContext:
    effective_solar: List[float]
    no_charge_hours: set
    no_discharge_hours: set
    reserve_kwh: List[float]
    max_grid_kwh: List[float]  # math.inf where uncapped


def apply_directives(
    interpretations: List[DirectiveInterpretationEntry],
    hours: List[HourEntry],
    battery: BatteryConfig,
) -> DirectiveContext:
    effective_solar = [h.solar_kwh for h in hours]
    no_charge_hours: set = set()
    no_discharge_hours: set = set()
    reserve_kwh = [battery.minimum_energy_kwh] * 24
    max_grid_kwh = [math.inf] * 24

    for entry in interpretations:
        if not entry.applies or entry.structured_adjustment is None:
            continue
        adj = entry.structured_adjustment
        directive_hours = adj.get("hours", [])

        if entry.directive_type == SOLAR_REDUCTION:
            factor = adj["factor"]
            for h in directive_hours:
                effective_solar[h] *= factor
        elif entry.directive_type == MINIMUM_BATTERY_RESERVE:
            level = adj["minimum_energy_kwh"]
            for h in directive_hours:
                reserve_kwh[h] = max(reserve_kwh[h], level)
        elif entry.directive_type == NO_CHARGE_WINDOW:
            no_charge_hours.update(directive_hours)
        elif entry.directive_type == NO_DISCHARGE_WINDOW:
            no_discharge_hours.update(directive_hours)
        elif entry.directive_type == MAX_GRID_WINDOW:
            cap = adj["max_grid_kwh"]
            for h in directive_hours:
                max_grid_kwh[h] = min(max_grid_kwh[h], cap)

    return DirectiveContext(
        effective_solar=effective_solar,
        no_charge_hours=no_charge_hours,
        no_discharge_hours=no_discharge_hours,
        reserve_kwh=reserve_kwh,
        max_grid_kwh=max_grid_kwh,
    )


def _solve_lp(
    hours: List[HourEntry],
    battery: BatteryConfig,
    effective_solar: List[float],
    no_charge_hours: set,
    no_discharge_hours: set,
    reserve_kwh: List[float],
    max_grid_kwh: List[float],
):
    """Solve one LP instance. Variable layout: [grid(24), solar(24), delta(24)]."""
    n = 24
    demand = [h.demand_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]

    c = np.zeros(3 * n)
    c[0:n] = tariff  # minimize sum(grid[h] * tariff[h])

    # Energy balance equality per hour: grid[h] + solar[h] - delta[h] = demand[h]
    A_eq = np.zeros((n + 1, 3 * n))
    b_eq = np.zeros(n + 1)
    for h in range(n):
        A_eq[h, h] = 1.0
        A_eq[h, n + h] = 1.0
        A_eq[h, 2 * n + h] = -1.0
        b_eq[h] = demand[h]
    # End-of-day battery neutrality: sum(delta) = 0
    A_eq[n, 2 * n : 3 * n] = 1.0
    b_eq[n] = 0.0

    # Reserve/capacity bounds on cumulative battery energy, per hour.
    A_ub_rows = []
    b_ub_rows = []
    for h in range(n):
        upper = np.zeros(3 * n)
        upper[2 * n : 2 * n + h + 1] = 1.0
        A_ub_rows.append(upper)
        b_ub_rows.append(battery.capacity_kwh - battery.initial_energy_kwh)

        lower = np.zeros(3 * n)
        lower[2 * n : 2 * n + h + 1] = -1.0
        A_ub_rows.append(lower)
        b_ub_rows.append(-(reserve_kwh[h] - battery.initial_energy_kwh))
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    bounds = []
    for h in range(n):
        grid_cap = max_grid_kwh[h]
        bounds.append((0, None if math.isinf(grid_cap) else grid_cap))
    for h in range(n):
        bounds.append((0, effective_solar[h]))
    for h in range(n):
        charge_cap = 0.0 if h in no_charge_hours else battery.max_charge_kwh_per_hour
        discharge_cap = 0.0 if h in no_discharge_hours else battery.max_discharge_kwh_per_hour
        bounds.append((-discharge_cap, charge_cap))

    return linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )


def solve_schedule(
    hours: List[HourEntry], battery: BatteryConfig, context: DirectiveContext
) -> List[HourlyPlanEntry]:
    """Solve for the lowest-cost valid schedule, with graceful degradation.

    Organizer-valid scenarios are guaranteed feasible with every directive
    applied. If our own extracted directives are ever inconsistent (e.g. a
    misread numeric value makes the LP infeasible), progressively relax the
    directive-derived constraints -- reserve first, then charge/discharge
    windows, then grid caps -- before falling back to the always-feasible
    plain-GridWise baseline (grid+solar only, battery untouched). This keeps
    the service from ever failing to produce a schedule.
    """
    attempts = [
        context,
        DirectiveContext(
            context.effective_solar,
            context.no_charge_hours,
            context.no_discharge_hours,
            [battery.minimum_energy_kwh] * 24,
            context.max_grid_kwh,
        ),
        DirectiveContext(
            context.effective_solar,
            set(),
            set(),
            [battery.minimum_energy_kwh] * 24,
            context.max_grid_kwh,
        ),
        DirectiveContext(
            context.effective_solar,
            set(),
            set(),
            [battery.minimum_energy_kwh] * 24,
            [math.inf] * 24,
        ),
        DirectiveContext(
            [h.solar_kwh for h in hours],
            set(),
            set(),
            [battery.minimum_energy_kwh] * 24,
            [math.inf] * 24,
        ),
    ]

    result = None
    for candidate in attempts:
        result = _solve_lp(
            hours,
            battery,
            candidate.effective_solar,
            candidate.no_charge_hours,
            candidate.no_discharge_hours,
            candidate.reserve_kwh,
            candidate.max_grid_kwh,
        )
        if result.success:
            break

    if result is None or not result.success:
        raise RuntimeError("Energy schedule is infeasible even under baseline GridWise rules")

    x = result.x
    n = 24
    grid = x[0:n]
    solar_used = x[n : 2 * n]
    delta = x[2 * n : 3 * n]

    plan: List[HourlyPlanEntry] = []
    energy = battery.initial_energy_kwh
    for h in range(n):
        d = delta[h]
        if d > TOLERANCE:
            action, magnitude = "charge", d
        elif d < -TOLERANCE:
            action, magnitude = "discharge", -d
        else:
            action, magnitude = "idle", 0.0
        energy += d
        plan.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=max(0.0, round(float(grid[h]), 6)),
                solar_used_kwh=max(0.0, round(float(solar_used[h]), 6)),
                battery_action=action,
                battery_kwh=round(float(magnitude), 6),
                battery_energy_after_kwh=round(float(energy), 6),
            )
        )
    return plan
