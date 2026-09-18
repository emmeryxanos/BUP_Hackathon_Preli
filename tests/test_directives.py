"""Direct unit tests for app/optimizer/directives.py.

These pure functions are shared by both the LP builder and the final replay
validator; testing them directly (rather than only transitively through a
solved LP) pins down their exact per-hour semantics, including overlap and
"tightest wins" rules.
"""
from app.optimizer.directives import (
    effective_solar,
    max_grid_by_hour,
    no_charge_hours,
    no_discharge_hours,
    reserve_by_hour,
)
from app.schemas import DirectiveInterpretation, StructuredAdjustment
from tests.conftest import make_hours


def _directive(directive_type, hours, applies=True, **adjustment_kwargs):
    return DirectiveInterpretation(
        note_index=0,
        applies=applies,
        directive_type=directive_type,
        structured_adjustment=StructuredAdjustment(hours=hours, **adjustment_kwargs)
        if hours is not None
        else None,
        explanation="test",
    )


def test_effective_solar_no_directives_returns_original():
    hours = make_hours(solar=[10] * 24)
    assert effective_solar(hours, []) == [10.0] * 24


def test_effective_solar_applies_factor_only_to_listed_hours():
    hours = make_hours(solar=[100] * 24)
    d = _directive("solar_reduction", [5, 6], factor=0.25)
    result = effective_solar(hours, [d])
    assert result[5] == 25.0
    assert result[6] == 25.0
    assert result[0] == 100.0


def test_effective_solar_ignores_directive_that_does_not_apply():
    hours = make_hours(solar=[100] * 24)
    d = _directive("solar_reduction", [5], applies=False, factor=0.0)
    result = effective_solar(hours, [d])
    assert result[5] == 100.0


def test_reserve_by_hour_uses_base_minimum_by_default():
    reserve = reserve_by_hour(50, 500, range(24), [])
    assert reserve == [50] * 24


def test_reserve_by_hour_directive_raises_only_listed_hours():
    d = _directive("minimum_battery_reserve", [10, 11], minimum_energy_kwh=200)
    reserve = reserve_by_hour(50, 500, range(24), [d])
    assert reserve[10] == 200
    assert reserve[11] == 200
    assert reserve[0] == 50


def test_reserve_by_hour_clamped_to_capacity():
    d = _directive("minimum_battery_reserve", [0], minimum_energy_kwh=9999)
    reserve = reserve_by_hour(50, 500, range(24), [d])
    assert reserve[0] == 500


def test_reserve_by_hour_overlapping_directives_take_max():
    d1 = _directive("minimum_battery_reserve", [0], minimum_energy_kwh=100)
    d2 = _directive("minimum_battery_reserve", [0], minimum_energy_kwh=300)
    reserve = reserve_by_hour(50, 500, range(24), [d1, d2])
    assert reserve[0] == 300


def test_no_charge_hours_collects_all_listed_hours():
    d1 = _directive("no_charge_window", [1, 2])
    d2 = _directive("no_charge_window", [2, 3])
    assert no_charge_hours([d1, d2]) == {1, 2, 3}


def test_no_charge_hours_empty_when_no_directives():
    assert no_charge_hours([]) == set()


def test_no_discharge_hours_collects_all_listed_hours():
    d = _directive("no_discharge_window", [7, 8])
    assert no_discharge_hours([d]) == {7, 8}


def test_no_discharge_hours_ignores_other_directive_types():
    d = _directive("no_charge_window", [7, 8])
    assert no_discharge_hours([d]) == set()


def test_max_grid_by_hour_defaults_to_infinity():
    caps = max_grid_by_hour(range(24), [])
    assert all(c == float("inf") for c in caps)


def test_max_grid_by_hour_tightest_cap_wins_on_overlap():
    d1 = _directive("max_grid_window", [3], max_grid_kwh=100)
    d2 = _directive("max_grid_window", [3], max_grid_kwh=40)
    caps = max_grid_by_hour(range(24), [d1, d2])
    assert caps[3] == 40


def test_max_grid_by_hour_only_affects_listed_hours():
    d = _directive("max_grid_window", [3], max_grid_kwh=40)
    caps = max_grid_by_hour(range(24), [d])
    assert caps[3] == 40
    assert caps[4] == float("inf")
