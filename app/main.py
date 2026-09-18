"""GridWise Energy Optimization API.

Endpoints (exact names required by the judge harness):
  GET  /health           -> {"status": "ok"}
  POST /optimize-energy   -> interpretation + 24-hour optimized schedule
"""
from __future__ import annotations

import logging

import anyio
from dotenv import load_dotenv

load_dotenv()  # must run before app.interpreter.llm_client reads its env vars at import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import LOG_LEVEL, REQUEST_TIMEOUT_SECONDS
from app.pipeline import OptimizationFailedError, run_pipeline
from app.schemas import HealthResponse, OptimizeEnergyRequest, OptimizeEnergyResponse

logging.basicConfig(level=LOG_LEVEL)
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
    # anyio.move_on_after (not asyncio.wait_for) -- FastAPI/Starlette run on
    # AnyIO, and wait_for's cross-thread cancellation interacts badly with
    # Windows' default ProactorEventLoop when combined with TestClient's
    # thread-portal bridging (reproducible native access-violation crashes).
    # move_on_after is AnyIO's own primitive and doesn't hit that path.
    result: OptimizeEnergyResponse | None = None
    with anyio.move_on_after(REQUEST_TIMEOUT_SECONDS) as scope:
        try:
            result = await run_pipeline(request)
        except OptimizationFailedError as exc:
            logger.error("optimize-energy failed for scenario %s: %s", request.scenario_id, exc)
            return JSONResponse(
                status_code=422,
                content={"error": "Scenario could not be solved to a valid schedule."},
            )

    if scope.cancelled_caught:
        logger.error(
            "optimize-energy for scenario %s exceeded %ss", request.scenario_id, REQUEST_TIMEOUT_SECONDS
        )
        return JSONResponse(status_code=500, content={"error": "Request processing timeout."})

    return result
