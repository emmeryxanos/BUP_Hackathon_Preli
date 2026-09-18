"""Tests for Pydantic request-schema validation (app/schemas.py).

These validators are the first line of defense before anything reaches the
pipeline; test_api.py exercises a couple of them over HTTP, but the specific
bound-checking rules on Battery and OptimizeEnergyRequest are otherwise
unverified.
"""
import pytest
from pydantic import ValidationError

from app.schemas import Battery, HourEntry, OptimizeEnergyRequest


def _valid_hours():
    return [
        {"hour": h, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 5} for h in range(24)
    ]


def _valid_battery_kwargs():
    return dict(
        capacity_kwh=500,
        initial_energy_kwh=200,
        minimum_energy_kwh=50,
        max_charge_kwh_per_hour=100,
        max_discharge_kwh_per_hour=100,
    )


def test_battery_minimum_exceeding_capacity_rejected():
    kwargs = _valid_battery_kwargs()
    kwargs["minimum_energy_kwh"] = 600
    with pytest.raises(ValidationError, match="minimum_energy_kwh cannot exceed capacity_kwh"):
        Battery(**kwargs)


def test_battery_initial_exceeding_capacity_rejected():
    kwargs = _valid_battery_kwargs()
    kwargs["initial_energy_kwh"] = 600
    with pytest.raises(ValidationError, match="initial_energy_kwh cannot exceed capacity_kwh"):
        Battery(**kwargs)


def test_battery_initial_below_minimum_rejected():
    kwargs = _valid_battery_kwargs()
    kwargs["initial_energy_kwh"] = 10
    kwargs["minimum_energy_kwh"] = 50
    with pytest.raises(ValidationError, match="initial_energy_kwh cannot be below minimum_energy_kwh"):
        Battery(**kwargs)


def test_battery_valid_bounds_accepted():
    battery = Battery(**_valid_battery_kwargs())
    assert battery.capacity_kwh == 500


def test_battery_negative_capacity_rejected():
    kwargs = _valid_battery_kwargs()
    kwargs["capacity_kwh"] = -1
    with pytest.raises(ValidationError):
        Battery(**kwargs)


def test_hour_entry_hour_out_of_range_rejected():
    with pytest.raises(ValidationError):
        HourEntry(hour=24, demand_kwh=1, solar_kwh=0, tariff_bdt_per_kwh=1)


def test_hour_entry_negative_demand_rejected():
    with pytest.raises(ValidationError):
        HourEntry(hour=0, demand_kwh=-1, solar_kwh=0, tariff_bdt_per_kwh=1)


def test_request_duplicate_hour_entries_rejected():
    hours = _valid_hours()
    hours[1]["hour"] = hours[0]["hour"]  # duplicate hour 0, missing hour 1
    with pytest.raises(ValidationError, match="one entry for each hour"):
        OptimizeEnergyRequest(
            scenario_id="X",
            operator_notes=["a note"],
            hours=hours,
            battery=_valid_battery_kwargs(),
        )


def test_request_empty_operator_note_rejected():
    with pytest.raises(ValidationError, match="non-empty strings"):
        OptimizeEnergyRequest(
            scenario_id="X",
            operator_notes=["   "],
            hours=_valid_hours(),
            battery=_valid_battery_kwargs(),
        )


def test_request_zero_operator_notes_rejected():
    with pytest.raises(ValidationError):
        OptimizeEnergyRequest(
            scenario_id="X",
            operator_notes=[],
            hours=_valid_hours(),
            battery=_valid_battery_kwargs(),
        )


def test_request_more_than_three_operator_notes_rejected():
    with pytest.raises(ValidationError):
        OptimizeEnergyRequest(
            scenario_id="X",
            operator_notes=["a", "b", "c", "d"],
            hours=_valid_hours(),
            battery=_valid_battery_kwargs(),
        )


def test_request_valid_payload_accepted():
    req = OptimizeEnergyRequest(
        scenario_id="X",
        operator_notes=["a note"],
        hours=_valid_hours(),
        battery=_valid_battery_kwargs(),
    )
    assert req.scenario_id == "X"
    assert len(req.hours) == 24
