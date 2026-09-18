"""Tests for app/pipeline.py orchestration, in particular plan_summary wording
and the infeasible/invalid-schedule error translation, which aren't covered
by the optimizer-only or API-only test suites.
"""
import pytest

import app.pipeline as pipeline_module
from app.pipeline import OptimizationFailedError, run_pipeline
from app.schemas import OptimizeEnergyRequest
from tests.conftest import make_battery, make_hours


def _request(operator_notes=None):
    hours = make_hours()
    battery = make_battery()
    return OptimizeEnergyRequest(
        scenario_id="TEST-1",
        operator_notes=operator_notes or ["irrelevant note"],
        hours=[h.model_dump() for h in hours],
        battery=battery.model_dump(),
    )


async def _stub_no_op(operator_notes, battery_capacity_kwh):
    from app.schemas import DirectiveInterpretation

    return [
        DirectiveInterpretation(
            note_index=i, applies=False, directive_type="no_op", structured_adjustment=None, explanation=""
        )
        for i in range(len(operator_notes))
    ]


async def _stub_one_directive(operator_notes, battery_capacity_kwh):
    from app.schemas import DirectiveInterpretation

    entries = [
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="no_charge_window",
            structured_adjustment={"hours": [1, 2]},
            explanation="",
        )
    ]
    for i in range(1, len(operator_notes)):
        entries.append(
            DirectiveInterpretation(
                note_index=i,
                applies=False,
                directive_type="no_op",
                structured_adjustment=None,
                explanation="",
            )
        )
    return entries


async def test_summary_mentions_no_directives_when_none_applied(monkeypatch):
    monkeypatch.setattr(pipeline_module, "interpret_notes", _stub_no_op)
    resp = await run_pipeline(_request())
    assert "No operator directives applied" in resp.plan_summary


async def test_summary_mentions_applied_count(monkeypatch):
    monkeypatch.setattr(pipeline_module, "interpret_notes", _stub_one_directive)
    resp = await run_pipeline(_request())
    assert "Applied 1 operator directive(s)" in resp.plan_summary


async def test_infeasible_pipeline_raises_optimization_failed(monkeypatch):
    from app.schemas import DirectiveInterpretation

    async def _stub_forces_infeasible(operator_notes, battery_capacity_kwh):
        return [
            DirectiveInterpretation(
                note_index=0,
                applies=True,
                directive_type="max_grid_window",
                structured_adjustment={"hours": list(range(24)), "max_grid_kwh": 0},
                explanation="",
            )
        ]

    monkeypatch.setattr(pipeline_module, "interpret_notes", _stub_forces_infeasible)

    battery = make_battery(max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0)
    request = OptimizeEnergyRequest(
        scenario_id="INFEASIBLE-1",
        operator_notes=["cap grid to zero"],
        hours=[h.model_dump() for h in make_hours(demand=[200] * 24, solar=[0] * 24)],
        battery=battery.model_dump(),
    )

    with pytest.raises(OptimizationFailedError):
        await run_pipeline(request)


async def test_hourly_plan_sorted_by_hour(monkeypatch):
    monkeypatch.setattr(pipeline_module, "interpret_notes", _stub_no_op)
    resp = await run_pipeline(_request())
    hours_seen = [p.hour for p in resp.hourly_plan]
    assert hours_seen == sorted(hours_seen)
    assert hours_seen == list(range(24))
