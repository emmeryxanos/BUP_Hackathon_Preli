import json
import os

from fastapi.testclient import TestClient

from app import pipeline as pipeline_module
from app.main import app
from app.schemas import DirectiveInterpretation

client = TestClient(app, raise_server_exceptions=False)

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "public_samples", "sample_grid_101.json")


def _stub_interpret_notes(operator_notes, battery_capacity_kwh):
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
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated internal failure with secret_token=ABC123")

    monkeypatch.setattr(pipeline_module, "interpret_notes", _boom)
    with open(SAMPLE_PATH) as f:
        payload = json.load(f)

    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 500
    assert "secret_token" not in resp.text
    assert "Traceback" not in resp.text
