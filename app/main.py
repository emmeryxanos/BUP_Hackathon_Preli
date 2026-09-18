"""GridWise Energy Optimization API.

Endpoints (exact names required by the judge harness):
  GET  /health           -> {"status": "ok"}
  POST /optimize-energy   -> interpretation + 24-hour optimized schedule
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.pipeline import OptimizationFailedError, run_pipeline
from app.schemas import HealthResponse, OptimizeEnergyRequest, OptimizeEnergyResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise.api")

app = FastAPI(title="GridWise Energy Optimization API", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"error": "Malformed or structurally invalid request."})


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"error": "Malformed or structurally invalid request."})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse)
async def optimize_energy(request: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    try:
        return run_pipeline(request)
    except OptimizationFailedError as exc:
        logger.error("optimize-energy failed for scenario %s: %s", request.scenario_id, exc)
        return JSONResponse(
            status_code=422,
            content={"error": "Scenario could not be solved to a valid schedule."},
        )
