import pytest

from app.optimizer.model import InfeasibleScenarioError, solve_schedule
from app.optimizer.validator import ScheduleValidationError, replay_and_validate
from app.schemas import DirectiveInterpretation, StructuredAdjustment
from tests.conftest import make_battery, make_hours


def test_no_directives_produces_valid_balanced_schedule():
    hours = make_hours()
    battery = make_battery()
    result = solve_schedule(hours, battery, [])
    replay_and_validate(hours, battery, [], result)  # must not raise
    assert len(result.hourly_plan) == 24
    assert result.total_grid_kwh > 0
    assert abs(result.hourly_plan[-1].battery_energy_after_kwh - battery.initial_energy_kwh) < 0.01


def test_solar_reduction_lowers_usable_solar():
    solar = [0] * 24
    solar[13] = 200
    solar[14] = 200
    hours = make_hours(solar=solar)
    battery = make_battery()

    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="solar_reduction",
        structured_adjustment=StructuredAdjustment(hours=[13, 14], factor=0.2),
        explanation="test",
    )
    result = solve_schedule(hours, battery, [directive])
    replay_and_validate(hours, battery, [directive], result)

    plan_by_hour = {p.hour: p for p in result.hourly_plan}
    # Usable solar in hour 13/14 is capped at 200*0.2 = 40, so solar_used cannot exceed that.
    assert plan_by_hour[13].solar_used_kwh <= 40.01
    assert plan_by_hour[14].solar_used_kwh <= 40.01


def test_no_charge_window_blocks_charging():
    hours = make_hours(solar=[300] * 24)  # abundant solar to tempt charging
    battery = make_battery()
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="no_charge_window",
        structured_adjustment=StructuredAdjustment(hours=[10, 11]),
        explanation="test",
    )
    result = solve_schedule(hours, battery, [directive])
    replay_and_validate(hours, battery, [directive], result)

    plan_by_hour = {p.hour: p for p in result.hourly_plan}
    assert plan_by_hour[10].battery_action != "charge"
    assert plan_by_hour[11].battery_action != "charge"


def test_no_discharge_window_blocks_discharging():
    hours = make_hours(demand=[400] * 24)  # high demand to tempt discharging
    battery = make_battery(initial_energy_kwh=300, minimum_energy_kwh=50)
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="no_discharge_window",
        structured_adjustment=StructuredAdjustment(hours=[5, 6]),
        explanation="test",
    )
    result = solve_schedule(hours, battery, [directive])
    replay_and_validate(hours, battery, [directive], result)

    plan_by_hour = {p.hour: p for p in result.hourly_plan}
    assert plan_by_hour[5].battery_action != "discharge"
    assert plan_by_hour[6].battery_action != "discharge"


def test_minimum_battery_reserve_enforced():
    hours = make_hours(demand=[400] * 24)
    battery = make_battery(initial_energy_kwh=300)
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="minimum_battery_reserve",
        structured_adjustment=StructuredAdjustment(hours=[18, 19, 20], minimum_energy_kwh=120),
        explanation="test",
    )
    result = solve_schedule(hours, battery, [directive])
    replay_and_validate(hours, battery, [directive], result)

    plan_by_hour = {p.hour: p for p in result.hourly_plan}
    for h in (18, 19, 20):
        assert plan_by_hour[h].battery_energy_after_kwh >= 120 - 0.01


def test_max_grid_window_caps_grid_import():
    hours = make_hours(demand=[300] * 24, solar=[0] * 24)
    battery = make_battery(initial_energy_kwh=300)
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="max_grid_window",
        structured_adjustment=StructuredAdjustment(hours=[8, 9], max_grid_kwh=250),
        explanation="test",
    )
    result = solve_schedule(hours, battery, [directive])
    replay_and_validate(hours, battery, [directive], result)

    plan_by_hour = {p.hour: p for p in result.hourly_plan}
    assert plan_by_hour[8].grid_kwh <= 300.01
    assert plan_by_hour[9].grid_kwh <= 300.01


def test_combined_directives_all_hold_simultaneously():
    solar = [0] * 24
    solar[13] = 300
    solar[14] = 300
    hours = make_hours(solar=solar, demand=[300] * 24)
    battery = make_battery(initial_energy_kwh=250)

    directives = [
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment=StructuredAdjustment(hours=[13, 14], factor=0.5),
            explanation="",
        ),
        DirectiveInterpretation(
            note_index=1,
            applies=True,
            directive_type="no_charge_window",
            structured_adjustment=StructuredAdjustment(hours=[14, 15]),
            explanation="",
        ),
        DirectiveInterpretation(
            note_index=2,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="",
        ),
    ]
    result = solve_schedule(hours, battery, directives)
    replay_and_validate(hours, battery, directives, result)  # must not raise


def test_infeasible_scenario_raises():
    hours = make_hours(demand=[1000] * 24, solar=[0] * 24)
    battery = make_battery(
        capacity_kwh=100,
        initial_energy_kwh=50,
        minimum_energy_kwh=50,
        max_charge_kwh_per_hour=10,
        max_discharge_kwh_per_hour=10,
    )
    directive = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="max_grid_window",
        structured_adjustment=StructuredAdjustment(hours=list(range(24)), max_grid_kwh=1),
        explanation="test",
    )
    with pytest.raises(InfeasibleScenarioError):
        solve_schedule(hours, battery, [directive])


def test_energy_balance_holds_every_hour():
    hours = make_hours()
    battery = make_battery()
    result = solve_schedule(hours, battery, [])
    for p, h in zip(result.hourly_plan, hours):
        charge = p.battery_kwh if p.battery_action == "charge" else 0.0
        discharge = p.battery_kwh if p.battery_action == "discharge" else 0.0
        lhs = p.grid_kwh + p.solar_used_kwh + discharge
        rhs = h.demand_kwh + charge
        assert abs(lhs - rhs) < 0.01
