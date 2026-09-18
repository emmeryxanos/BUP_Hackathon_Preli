import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.schemas import BatteryConfig, HourEntry


def make_hours() -> list[HourEntry]:
    demand = [
        180, 170, 160, 160, 170, 190, 220, 260,
        300, 310, 300, 290, 280, 290, 300, 310,
        330, 360, 380, 360, 320, 280, 230, 200,
    ]
    solar = [
        0, 0, 0, 0, 0, 20, 80, 160,
        240, 300, 340, 360, 360, 340, 300, 240,
        160, 80, 20, 0, 0, 0, 0, 0,
    ]
    tariff = [
        6, 6, 6, 6, 6, 6, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 8,
        9, 10, 10, 9, 8, 7, 7, 6,
    ]
    return [
        HourEntry(hour=h, demand_kwh=demand[h], solar_kwh=solar[h], tariff_bdt_per_kwh=tariff[h])
        for h in range(24)
    ]


def make_battery() -> BatteryConfig:
    return BatteryConfig(
        capacity_kwh=500,
        initial_energy_kwh=200,
        minimum_energy_kwh=50,
        max_charge_kwh_per_hour=100,
        max_discharge_kwh_per_hour=100,
    )


@pytest.fixture
def hours():
    return make_hours()


@pytest.fixture
def battery():
    return make_battery()


@pytest.fixture
def sample_request_payload():
    return {
        "scenario_id": "GRID-TEST-1",
        "operator_notes": [
            "Solar output will drop to about 20% from 1 PM to 3 PM.",
            "Do not charge the battery between 2 PM and 4 PM.",
            "The cafeteria menu changes tomorrow.",
        ],
        "hours": [h.model_dump() for h in make_hours()],
        "battery": make_battery().model_dump(),
    }
