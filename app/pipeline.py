"""End-to-end orchestration: notes -> LLM interpreter -> guardrails ->
optimizer -> final self-validation -> response assembly.

This is the single place that wires every stage of the mandated pipeline
together: Energy Data + Notes -> LLM Interpreter -> Guardrail Validator ->
Math Optimizer -> Final Validator -> API Response.
"""
from __future__ import annotations

import logging

from app.interpreter import interpret_notes
from app.optimizer.model import InfeasibleScenarioError, solve_schedule
from app.optimizer.validator import ScheduleValidationError, replay_and_validate
from app.schemas import HourlyPlanEntry, OptimizeEnergyRequest, OptimizeEnergyResponse

logger = logging.getLogger("gridwise.pipeline")


class OptimizationFailedError(Exception):
    """Raised when no valid schedule could be produced or validated."""


def _summarize(directive_count_applied: int, total_cost: float, total_grid: float) -> str:
    if directive_count_applied == 0:
        return (
            f"No operator directives applied. Optimized purely for cost: "
            f"{total_grid:.2f} kWh from grid at a total cost of {total_cost:.2f} BDT."
        )
    return (
        f"Applied {directive_count_applied} operator directive(s) and produced a "
        f"cost-minimized 24-hour schedule: {total_grid:.2f} kWh from grid, "
        f"total cost {total_cost:.2f} BDT."
    )


async def run_pipeline(request: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    directive_interpretation = await interpret_notes(
        request.operator_notes, request.battery.capacity_kwh
    )

    try:
        result = solve_schedule(request.hours, request.battery, directive_interpretation)
    except InfeasibleScenarioError as exc:
        logger.error("Scenario %s infeasible: %s", request.scenario_id, exc)
        raise OptimizationFailedError(str(exc)) from exc

    try:
        replay_and_validate(request.hours, request.battery, directive_interpretation, result)
    except ScheduleValidationError as exc:
        logger.error("Scenario %s failed self-validation: %s", request.scenario_id, exc)
        raise OptimizationFailedError(str(exc)) from exc

    applied_count = sum(
        1 for d in directive_interpretation if d.applies and d.directive_type != "no_op"
    )

    hourly_plan = [
        HourlyPlanEntry(
            hour=p.hour,
            grid_kwh=p.grid_kwh,
            solar_used_kwh=p.solar_used_kwh,
            battery_action=p.battery_action,
            battery_kwh=p.battery_kwh,
            battery_energy_after_kwh=p.battery_energy_after_kwh,
        )
        for p in sorted(result.hourly_plan, key=lambda x: x.hour)
    ]

    return OptimizeEnergyResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directive_interpretation,
        hourly_plan=hourly_plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
        plan_summary=_summarize(applied_count, result.total_cost_bdt, result.total_grid_kwh),
    )
