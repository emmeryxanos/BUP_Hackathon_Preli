"""GridWise LLM-assisted energy optimization service.

Pipeline (Problem Statement Section 03):
  operator notes -> LLM interpreter -> guardrail validator -> LP optimizer
  -> response

GET /health and POST /optimize-energy are the only two endpoints the judge
harness calls; both are implemented exactly to the Preliminary Problem
Statement contract.
"""
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import REQUEST_TIMEOUT_SECONDS
from app.guardrails import validate_interpretation
from app.llm_interpreter import interpret_notes
from app.optimizer import apply_directives, solve_schedule
from app.schemas import HealthResponse, OptimizeEnergyRequest, OptimizeEnergyResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise.api")

app = FastAPI(title="GridWise LLM Optimization Service")


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Problem Statement Section 6.1: malformed JSON / structurally invalid -> 400.
    return JSONResponse(status_code=400, content={"error": "malformed or invalid request"})


@app.exception_handler(ValidationError)
async def on_pydantic_error(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "malformed or invalid request"})


@app.exception_handler(Exception)
async def on_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal error"})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _build_plan_summary(
    interpretations, total_cost_bdt: float, peak_grid_kwh: float, peak_hour: int
) -> str:
    applied = [e for e in interpretations if e.applies]
    if applied:
        directive_list = ", ".join(f"{e.directive_type}(note {e.note_index})" for e in applied)
        directive_clause = f"Applied directives: {directive_list}."
    else:
        directive_clause = "No operator directives applied; all notes were no_op."
    return (
        f"{directive_clause} Minimized 24h grid cost to {total_cost_bdt:.2f} BDT "
        f"with peak grid draw {peak_grid_kwh:.2f} kWh at hour {peak_hour}."
    )


async def _process(payload: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    hours = payload.sorted_hours()

    raw_interpretations = await interpret_notes(payload.operator_notes)
    interpretations = validate_interpretation(
        raw_interpretations,
        num_notes=len(payload.operator_notes),
        battery_capacity_kwh=payload.battery.capacity_kwh,
    )

    context = apply_directives(interpretations, hours, payload.battery)
    hourly_plan = solve_schedule(hours, payload.battery, context)

    total_grid_kwh = sum(p.grid_kwh for p in hourly_plan)
    total_cost_bdt = sum(p.grid_kwh * h.tariff_bdt_per_kwh for p, h in zip(hourly_plan, hours))
    peak_entry = max(hourly_plan, key=lambda p: p.grid_kwh)

    return OptimizeEnergyResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=interpretations,
        hourly_plan=hourly_plan,
        total_grid_kwh=round(total_grid_kwh, 6),
        total_cost_bdt=round(total_cost_bdt, 6),
        peak_grid_kwh=round(peak_entry.grid_kwh, 6),
        plan_summary=_build_plan_summary(
            interpretations, total_cost_bdt, peak_entry.grid_kwh, peak_entry.hour
        ),
    )


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse)
async def optimize_energy(payload: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    try:
        return await asyncio.wait_for(_process(payload), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("Request for scenario %s exceeded %ss", payload.scenario_id, REQUEST_TIMEOUT_SECONDS)
        return JSONResponse(status_code=500, content={"error": "processing timeout"})
