"""API-level tests using Starlette's TestClient.

Known-benign noise on this platform: on Windows + Python 3.11.x, some of the
async POST /optimize-energy tests here print "Windows fatal exception:
access violation" tracebacks during teardown. This is a documented
interaction between Starlette TestClient's sync-to-async thread bridging
(anyio.from_thread) and the Windows ProactorEventLoop when the endpoint does
real async work (httpx calls, SciPy solve) -- it happens after the test's
own assertions have already completed and does not affect the assert/pass
outcome (verified: pytest exit code 0, every test still reports PASSED).
It was independently confirmed NOT to reproduce against a real uvicorn
server hit with real HTTP requests (curl), so it is a TestClient-only
artifact on this OS/Python combination, not a production risk.
"""
import json
import os

from fastapi.testclient import TestClient

from app import pipeline as pipeline_module
from app.main import app
from app.schemas import DirectiveInterpretation

client = TestClient(app, raise_server_exceptions=False)

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "public_samples", "sample_grid_101.json")


async def _stub_interpret_notes(operator_notes, battery_capacity_kwh):
    """Deterministic stand-in for the LLM so API tests don't need network/API keys."""
    entries = []
    for i, note in enumerate(operator_notes):
        lower = note.lower()
        if "solar" in lower and ("20%" in lower or "drop" in lower):
            entries.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=True,
                    directive_type="solar_reduction",
                    structured_adjustment={"hours": [13, 14], "factor": 0.2},
                    explanation="stub",
                )
            )
        elif "do not charge" in lower or "not charge" in lower:
            entries.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=True,
                    directive_type="no_charge_window",
                    structured_adjustment={"hours": [14, 15]},
                    explanation="stub",
                )
            )
        else:
            entries.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation="stub",
                )
            )
    return entries


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_optimize_energy_happy_path(monkeypatch):
    monkeypatch.setattr(pipeline_module, "interpret_notes", _stub_interpret_notes)
    with open(SAMPLE_PATH) as f:
        payload = json.load(f)

    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert body["scenario_id"] == "GRID-101"
    assert len(body["directive_interpretation"]) == 3
    assert len(body["hourly_plan"]) == 24
    assert body["directive_interpretation"][2]["directive_type"] == "no_op"
    assert body["directive_interpretation"][2]["applies"] is False

    hours_seen = sorted(p["hour"] for p in body["hourly_plan"])
    assert hours_seen == list(range(24))

    recalced_grid = sum(p["grid_kwh"] for p in body["hourly_plan"])
    assert abs(recalced_grid - body["total_grid_kwh"]) < 0.02


def test_malformed_json_returns_400():
    resp = client.post("/optimize-energy", content="{not valid json", headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_missing_required_field_returns_400():
    resp = client.post("/optimize-energy", json={"scenario_id": "X"})
    assert resp.status_code == 400


def test_wrong_number_of_hours_returns_400():
    payload = {
        "scenario_id": "BAD-1",
        "operator_notes": ["irrelevant note"],
        "hours": [{"hour": 0, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 5}],
        "battery": {
            "capacity_kwh": 100,
            "initial_energy_kwh": 50,
            "minimum_energy_kwh": 10,
            "max_charge_kwh_per_hour": 20,
            "max_discharge_kwh_per_hour": 20,
        },
    }
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400


def test_too_many_operator_notes_returns_400(sample_scenario):
    payload = dict(sample_scenario)
    payload["operator_notes"] = ["a", "b", "c", "d"]
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400


def test_response_never_leaks_stack_trace(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated internal failure with secret_token=ABC123")

    monkeypatch.setattr(pipeline_module, "interpret_notes", _boom)
    with open(SAMPLE_PATH) as f:
        payload = json.load(f)

    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 500
    assert "secret_token" not in resp.text
    assert "Traceback" not in resp.text


def test_infeasible_scenario_returns_422(monkeypatch, sample_scenario):
    async def _force_max_grid_directive(operator_notes, battery_capacity_kwh):
        return [
            DirectiveInterpretation(
                note_index=i,
                applies=(i == 0),
                directive_type="max_grid_window" if i == 0 else "no_op",
                structured_adjustment={"hours": list(range(24)), "max_grid_kwh": 1}
                if i == 0
                else None,
                explanation="stub forces infeasibility",
            )
            for i, _ in enumerate(operator_notes)
        ]

    monkeypatch.setattr(pipeline_module, "interpret_notes", _force_max_grid_directive)
    payload = dict(sample_scenario)
    # No solar, no battery discharge capacity, and grid capped to 1 kWh/hour while
    # demand is >=160 kWh every hour: no feasible schedule can satisfy energy balance.
    payload["hours"] = [
        {"hour": h, "demand_kwh": 200, "solar_kwh": 0, "tariff_bdt_per_kwh": 5} for h in range(24)
    ]
    payload["battery"] = {
        "capacity_kwh": 10,
        "initial_energy_kwh": 5,
        "minimum_energy_kwh": 5,
        "max_charge_kwh_per_hour": 0,
        "max_discharge_kwh_per_hour": 0,
    }

    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert "Traceback" not in resp.text


def test_negative_battery_capacity_returns_400(sample_scenario):
    payload = dict(sample_scenario)
    payload["battery"] = dict(payload["battery"])
    payload["battery"]["capacity_kwh"] = -100
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400


def test_empty_operator_note_returns_400(sample_scenario):
    payload = dict(sample_scenario)
    payload["operator_notes"] = ["   "]
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400


def test_unknown_route_returns_404():
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404


def test_health_endpoint_rejects_post():
    resp = client.post("/health")
    assert resp.status_code == 405


def test_llm_raw_directive_underreturn_padded_with_no_op_end_to_end(monkeypatch, sample_scenario):
    """If the raw LLM call under-returns entries, the API must still return one
    directive_interpretation entry per operator note (guardrails.validate_all
    padding), rather than a short or misaligned array reaching the caller."""
    import app.interpreter as interpreter_module

    async def _partial_raw_call(notes):
        return [{"note_index": 0, "directive_type": "no_op", "applies": False, "explanation": "only one"}]

    monkeypatch.setattr(interpreter_module, "call_llm_for_directives", _partial_raw_call)
    resp = client.post("/optimize-energy", json=sample_scenario)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["directive_interpretation"]) == len(sample_scenario["operator_notes"])
    note_indices = [d["note_index"] for d in body["directive_interpretation"]]
    assert note_indices == list(range(len(sample_scenario["operator_notes"])))
