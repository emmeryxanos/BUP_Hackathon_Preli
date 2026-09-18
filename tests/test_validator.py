"""Direct tests for the final self-validation replay (app/optimizer/validator.py).

This module is the last safety net before a response is ever returned, so its
rejection branches need to be exercised directly rather than only implicitly
via optimizer tests that always pass it an already-correct solver result.
"""
import pytest

from app.optimizer.model import HourResult, OptimizationResult
from app.optimizer.validator import ScheduleValidationError, replay_and_validate
from tests.conftest import make_battery, make_hours


def _valid_plan(battery, hours):
    """A trivially correct 24h idle plan: grid covers demand, battery untouched."""
    plan = []
    for h in hours:
        plan.append(
            HourResult(
                hour=h.hour,
                grid_kwh=h.demand_kwh,
                solar_used_kwh=0.0,
                battery_action="idle",
                battery_kwh=0.0,
                battery_energy_after_kwh=battery.initial_energy_kwh,
            )
        )
    total_grid = sum(p.grid_kwh for p in plan)
    total_cost = sum(p.grid_kwh * h.tariff_bdt_per_kwh for p, h in zip(plan, hours))
    peak = max(p.grid_kwh for p in plan)
    return OptimizationResult(
        hourly_plan=plan, total_grid_kwh=total_grid, total_cost_bdt=total_cost, peak_grid_kwh=peak
    )


def test_valid_plan_passes():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    replay_and_validate(hours, battery, [], result)  # must not raise


def test_wrong_number_of_plan_entries_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan = result.hourly_plan[:-1]
    with pytest.raises(ScheduleValidationError, match="exactly 24 entries"):
        replay_and_validate(hours, battery, [], result)


def test_duplicate_plan_hours_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[1].hour = result.hourly_plan[0].hour
    with pytest.raises(ScheduleValidationError, match="0..23, unique"):
        replay_and_validate(hours, battery, [], result)


def test_negative_grid_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].grid_kwh = -5.0
    with pytest.raises(ScheduleValidationError, match="negative energy value"):
        replay_and_validate(hours, battery, [], result)


def test_solar_used_exceeding_effective_solar_rejected():
    solar = [50] * 24
    hours = make_hours(solar=solar)
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].solar_used_kwh = 999.0
    with pytest.raises(ScheduleValidationError, match="exceeds effective solar"):
        replay_and_validate(hours, battery, [], result)


def test_invalid_battery_action_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_action = "teleport"
    with pytest.raises(ScheduleValidationError, match="invalid battery_action"):
        replay_and_validate(hours, battery, [], result)


def test_idle_with_nonzero_battery_kwh_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_kwh = 10.0  # action stays "idle"
    with pytest.raises(ScheduleValidationError, match="idle hour must have battery_kwh 0"):
        replay_and_validate(hours, battery, [], result)


def test_charge_exceeding_rate_limit_rejected():
    hours = make_hours()
    battery = make_battery(max_charge_kwh_per_hour=50)
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_action = "charge"
    result.hourly_plan[0].battery_kwh = 999.0
    result.hourly_plan[0].battery_energy_after_kwh = battery.initial_energy_kwh + 999.0
    with pytest.raises(ScheduleValidationError, match="exceeds max_charge_kwh_per_hour"):
        replay_and_validate(hours, battery, [], result)


def test_discharge_exceeding_rate_limit_rejected():
    hours = make_hours()
    battery = make_battery(max_discharge_kwh_per_hour=50)
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_action = "discharge"
    result.hourly_plan[0].battery_kwh = 999.0
    result.hourly_plan[0].battery_energy_after_kwh = battery.initial_energy_kwh - 999.0
    with pytest.raises(ScheduleValidationError, match="exceeds max_discharge_kwh_per_hour"):
        replay_and_validate(hours, battery, [], result)


def test_charging_during_no_charge_window_rejected():
    from app.schemas import DirectiveInterpretation, StructuredAdjustment

    hours = make_hours()
    battery = make_battery()
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="no_charge_window",
        structured_adjustment=StructuredAdjustment(hours=[0]),
        explanation="test",
    )
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_action = "charge"
    result.hourly_plan[0].battery_kwh = 20.0
    result.hourly_plan[0].battery_energy_after_kwh = battery.initial_energy_kwh + 20.0
    with pytest.raises(ScheduleValidationError, match="charging during no_charge_window"):
        replay_and_validate(hours, battery, [directive], result)


def test_discharging_during_no_discharge_window_rejected():
    from app.schemas import DirectiveInterpretation, StructuredAdjustment

    hours = make_hours()
    battery = make_battery()
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="no_discharge_window",
        structured_adjustment=StructuredAdjustment(hours=[0]),
        explanation="test",
    )
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_action = "discharge"
    result.hourly_plan[0].battery_kwh = 20.0
    result.hourly_plan[0].battery_energy_after_kwh = battery.initial_energy_kwh - 20.0
    with pytest.raises(ScheduleValidationError, match="discharging during no_discharge_window"):
        replay_and_validate(hours, battery, [directive], result)


def test_grid_exceeding_max_grid_window_rejected():
    from app.schemas import DirectiveInterpretation, StructuredAdjustment

    hours = make_hours()
    battery = make_battery()
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="max_grid_window",
        structured_adjustment=StructuredAdjustment(hours=[0], max_grid_kwh=10),
        explanation="test",
    )
    result = _valid_plan(battery, hours)  # hour 0 grid_kwh = demand (200) > cap of 10
    with pytest.raises(ScheduleValidationError, match="exceeds max_grid_window cap"):
        replay_and_validate(hours, battery, [directive], result)


def test_battery_state_transition_mismatch_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_energy_after_kwh = battery.initial_energy_kwh + 500.0
    with pytest.raises(ScheduleValidationError, match="battery state transition mismatch"):
        replay_and_validate(hours, battery, [], result)


def test_battery_below_reserve_rejected():
    from app.schemas import DirectiveInterpretation, StructuredAdjustment

    hours = make_hours()
    battery = make_battery(initial_energy_kwh=200, minimum_energy_kwh=50)
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="minimum_battery_reserve",
        structured_adjustment=StructuredAdjustment(hours=[0], minimum_energy_kwh=150),
        explanation="test",
    )
    result = _valid_plan(battery, hours)  # battery stays at 200, still above 150; force below
    result.hourly_plan[0].battery_action = "discharge"
    result.hourly_plan[0].battery_kwh = 100.0
    result.hourly_plan[0].battery_energy_after_kwh = 100.0
    with pytest.raises(ScheduleValidationError, match="below required reserve"):
        replay_and_validate(hours, battery, [directive], result)


def test_battery_exceeding_capacity_rejected():
    hours = make_hours()
    battery = make_battery(capacity_kwh=500, max_charge_kwh_per_hour=1000)
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].battery_action = "charge"
    result.hourly_plan[0].battery_kwh = 400.0
    result.hourly_plan[0].battery_energy_after_kwh = battery.initial_energy_kwh + 400.0
    with pytest.raises(ScheduleValidationError, match="exceeds capacity"):
        replay_and_validate(hours, battery, [], result)


def test_energy_balance_violation_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[0].grid_kwh = 1.0  # demand is 200, no compensating source
    with pytest.raises(ScheduleValidationError, match="energy balance equation violated"):
        replay_and_validate(hours, battery, [], result)


def test_end_of_day_neutrality_violation_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.hourly_plan[-1].battery_action = "discharge"
    result.hourly_plan[-1].battery_kwh = 30.0
    result.hourly_plan[-1].battery_energy_after_kwh = battery.initial_energy_kwh - 30.0
    result.hourly_plan[-1].grid_kwh -= 30.0  # keep balance equation satisfied for that hour
    with pytest.raises(ScheduleValidationError, match="neutrality violated"):
        replay_and_validate(hours, battery, [], result)


def test_total_cost_mismatch_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.total_cost_bdt += 1000.0
    with pytest.raises(ScheduleValidationError, match="total_cost_bdt does not match"):
        replay_and_validate(hours, battery, [], result)


def test_total_grid_mismatch_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.total_grid_kwh += 1000.0
    with pytest.raises(ScheduleValidationError, match="total_grid_kwh does not match"):
        replay_and_validate(hours, battery, [], result)


def test_peak_grid_mismatch_rejected():
    hours = make_hours()
    battery = make_battery()
    result = _valid_plan(battery, hours)
    result.peak_grid_kwh += 1000.0
    with pytest.raises(ScheduleValidationError, match="peak_grid_kwh does not match"):
        replay_and_validate(hours, battery, [], result)
