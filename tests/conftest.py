import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas import Battery, HourEntry  # noqa: E402

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "public_samples", "sample_grid_101.json")


@pytest.fixture
def sample_scenario():
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def make_hours(demand=None, solar=None, tariff=None):
    demand = demand or [200] * 24
    solar = solar or [0] * 24
    tariff = tariff or [8] * 24
    return [
        HourEntry(hour=h, demand_kwh=demand[h], solar_kwh=solar[h], tariff_bdt_per_kwh=tariff[h])
        for h in range(24)
    ]


def make_battery(**overrides):
    defaults = dict(
        capacity_kwh=500,
        initial_energy_kwh=200,
        minimum_energy_kwh=50,
        max_charge_kwh_per_hour=100,
        max_discharge_kwh_per_hour=100,
    )
    defaults.update(overrides)
    return Battery(**defaults)
