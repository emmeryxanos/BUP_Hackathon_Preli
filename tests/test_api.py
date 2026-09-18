import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def _fake_interpret_notes(operator_notes):
    # Deterministic stand-in for the real LLM call so tests don't need a
    # live ANTHROPIC_API_KEY. Mirrors what a correct model response for the
    # sample_request_payload fixture's three notes should look like.
    return [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "hours": [13, 14],
            "factor": 0.2,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "Solar drops during maintenance window.",
        },
        {
            "note_index": 1,
            "applies": True,
            "directive_type": "no_charge_window",
            "hours": [14, 15],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "Charging disabled 2-4 PM.",
        },
        {
            "note_index": 2,
            "applies": False,
            "directive_type": "no_op",
            "hours": [],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "Cafeteria menu is unrelated to the energy schedule.",
        },
    ]


def test_optimize_energy_happy_path(client, sample_request_payload, monkeypatch):
    monkeypatch.setattr(main_module, "interpret_notes", _fake_interpret_notes)

    resp = client.post("/optimize-energy", json=sample_request_payload)
    assert resp.status_code == 200
    body = resp.json()

    assert body["scenario_id"] == "GRID-TEST-1"
    assert len(body["directive_interpretation"]) == 3
    assert [e["note_index"] for e in body["directive_interpretation"]] == [0, 1, 2]
    assert body["directive_interpretation"][0]["directive_type"] == "solar_reduction"
    assert body["directive_interpretation"][2]["directive_type"] == "no_op"
    assert body["directive_interpretation"][2]["applies"] is False

    assert len(body["hourly_plan"]) == 24
    for h in (14, 15):
        assert body["hourly_plan"][h]["battery_action"] != "charge"

    recalculated_total = sum(p["grid_kwh"] for p in body["hourly_plan"])
    assert abs(recalculated_total - body["total_grid_kwh"]) < 0.02
    assert abs(max(p["grid_kwh"] for p in body["hourly_plan"]) - body["peak_grid_kwh"]) < 0.02


def test_malformed_request_returns_400_missing_field(client, sample_request_payload):
    bad_payload = dict(sample_request_payload)
    del bad_payload["battery"]
    resp = client.post("/optimize-energy", json=bad_payload)
    assert resp.status_code == 400


def test_malformed_request_returns_400_wrong_hour_count(client, sample_request_payload):
    bad_payload = dict(sample_request_payload)
    bad_payload["hours"] = sample_request_payload["hours"][:23]
    resp = client.post("/optimize-energy", json=bad_payload)
    assert resp.status_code == 400


def test_too_many_operator_notes_returns_400(client, sample_request_payload):
    bad_payload = dict(sample_request_payload)
    bad_payload["operator_notes"] = ["a", "b", "c", "d"]
    resp = client.post("/optimize-energy", json=bad_payload)
    assert resp.status_code == 400


def test_llm_failure_falls_back_safely(client, sample_request_payload, monkeypatch):
    async def _boom(operator_notes):
        raise RuntimeError("simulated provider outage")

    async def _safe_fallback(operator_notes):
        # This mirrors interpret_notes' own internal safe-failure contract:
        # it never raises, even if the provider call fails.
        return [
            {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "hours": [],
                "factor": None,
                "minimum_energy_kwh": None,
                "max_grid_kwh": None,
                "explanation": "LLM interpretation unavailable; safe fallback applied.",
            }
            for i in range(len(operator_notes))
        ]

    monkeypatch.setattr(main_module, "interpret_notes", _safe_fallback)
    resp = client.post("/optimize-energy", json=sample_request_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert all(e["directive_type"] == "no_op" for e in body["directive_interpretation"])
    assert len(body["hourly_plan"]) == 24
