"""Linear-programming model for the 24-hour GridWise cost-minimization problem.

Implements the exact rules from the Problem Statement, Section 9:
  - battery state transitions (charge / discharge / idle)
  - battery bounds (min_energy <= E_after <= capacity)
  - hourly charge/discharge rate limits
  - solar usage bounded by effective (post-directive) solar
  - energy balance: grid + solar_used + discharge = demand + charge
  - end-of-day battery neutrality: final energy == initial energy

Objective: minimize sum(grid_kwh[h] * tariff[h]) over h = 0..23.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pulp

from app.optimizer.directives import (
    effective_solar,
    max_grid_by_hour,
    no_charge_hours,
    no_discharge_hours,
    reserve_by_hour,
)
from app.schemas import Battery, DirectiveInterpretation, HourEntry

BIG_TOLERANCE = 1e-6


@dataclass
class HourResult:
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str
    battery_kwh: float
    battery_energy_after_kwh: float


@dataclass
class OptimizationResult:
    hourly_plan: List[HourResult]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float


class InfeasibleScenarioError(Exception):
    """Raised when no feasible schedule exists for the given inputs/directives."""


def solve_schedule(
    hours: List[HourEntry],
    battery: Battery,
    directives: List[DirectiveInterpretation],
) -> OptimizationResult:
    """Build and solve the LP, returning a validated, cost-minimal 24h schedule."""
    hours_sorted = sorted(hours, key=lambda h: h.hour)
    n = len(hours_sorted)
    idx_range = range(n)

    eff_solar = effective_solar(hours_sorted, directives)
    reserve = reserve_by_hour(battery.minimum_energy_kwh, battery.capacity_kwh, idx_range, directives)
    no_charge = no_charge_hours(directives)
    no_discharge = no_discharge_hours(directives)
    max_grid = max_grid_by_hour(idx_range, directives)

    prob = pulp.LpProblem("gridwise_schedule", pulp.LpMinimize)

    grid = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in idx_range]
    solar_used = [
        pulp.LpVariable(f"solar_used_{h}", lowBound=0, upBound=eff_solar[h]) for h in idx_range
    ]
    charge = [
        pulp.LpVariable(
            f"charge_{h}", lowBound=0, upBound=0 if h in no_charge else battery.max_charge_kwh_per_hour
        )
        for h in idx_range
    ]
    discharge = [
        pulp.LpVariable(
            f"discharge_{h}",
            lowBound=0,
            upBound=0 if h in no_discharge else battery.max_discharge_kwh_per_hour,
        )
        for h in idx_range
    ]
    energy_after = [
        pulp.LpVariable(f"energy_after_{h}", lowBound=reserve[h], upBound=battery.capacity_kwh)
        for h in idx_range
    ]

    # Objective: minimize total grid cost.
    prob += pulp.lpSum(grid[h] * hours_sorted[h].tariff_bdt_per_kwh for h in idx_range)

    # Energy balance each hour.
    for h in idx_range:
        prob += (
            grid[h] + solar_used[h] + discharge[h] == hours_sorted[h].demand_kwh + charge[h],
            f"balance_{h}",
        )

    # Battery state transitions.
    prev_energy = battery.initial_energy_kwh
    for h in idx_range:
        prob += (energy_after[h] == prev_energy + charge[h] - discharge[h], f"battery_state_{h}")
        prev_energy = energy_after[h]

    # Grid cap per directive.
    for h in idx_range:
        if max_grid[h] != float("inf"):
            prob += (grid[h] <= max_grid[h], f"max_grid_{h}")

    # End-of-day battery neutrality.
    prob += (energy_after[n - 1] == battery.initial_energy_kwh, "battery_neutrality")

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleScenarioError(
            f"No feasible schedule found (solver status: {pulp.LpStatus[status]})"
        )

    plan: List[HourResult] = []
    total_cost = 0.0
    for h in idx_range:
        g = max(0.0, pulp.value(grid[h]) or 0.0)
        su = max(0.0, pulp.value(solar_used[h]) or 0.0)
        c = max(0.0, pulp.value(charge[h]) or 0.0)
        dch = max(0.0, pulp.value(discharge[h]) or 0.0)
        e_after = pulp.value(energy_after[h]) or 0.0

        if c > BIG_TOLERANCE and c >= dch:
            action, magnitude = "charge", c
            dch = 0.0
        elif dch > BIG_TOLERANCE:
            action, magnitude = "discharge", dch
            c = 0.0
        else:
            action, magnitude = "idle", 0.0
            c = 0.0
            dch = 0.0

        plan.append(
            HourResult(
                hour=hours_sorted[h].hour,
                grid_kwh=round(g, 6),
                solar_used_kwh=round(su, 6),
                battery_action=action,
                battery_kwh=round(magnitude, 6),
                battery_energy_after_kwh=round(e_after, 6),
            )
        )
        total_cost += g * hours_sorted[h].tariff_bdt_per_kwh

    total_grid = sum(p.grid_kwh for p in plan)
    peak_grid = max(p.grid_kwh for p in plan)

    return OptimizationResult(
        hourly_plan=plan,
        total_grid_kwh=round(total_grid, 6),
        total_cost_bdt=round(total_cost, 6),
        peak_grid_kwh=round(peak_grid, 6),
    )
