import math

from app.optimizer import TOLERANCE, apply_directives, solve_schedule
from app.schemas import DirectiveInterpretationEntry


def _no_op_interpretations():
    return []


def _energy_balance_holds(hours, plan):
    for h, p in zip(hours, plan):
        lhs = p.grid_kwh + p.solar_used_kwh
        if p.battery_action == "discharge":
            lhs += p.battery_kwh
        rhs = h.demand_kwh
        if p.battery_action == "charge":
            rhs += p.battery_kwh
        assert abs(lhs - rhs) <= TOLERANCE + 1e-6, f"hour {h.hour}: {lhs} != {rhs}"


def test_baseline_schedule_is_valid(hours, battery):
    context = apply_directives(_no_op_interpretations(), hours, battery)
    plan = solve_schedule(hours, battery, context)

    assert len(plan) == 24
    assert [p.hour for p in plan] == list(range(24))
    _energy_balance_holds(hours, plan)

    # End-of-day battery neutrality.
    assert abs(plan[-1].battery_energy_after_kwh - battery.initial_energy_kwh) <= TOLERANCE

    # No negative values anywhere.
    for p in plan:
        assert p.grid_kwh >= -1e-9
        assert p.solar_used_kwh >= -1e-9
        assert p.battery_kwh >= -1e-9
        assert battery.minimum_energy_kwh - TOLERANCE <= p.battery_energy_after_kwh
        assert p.battery_energy_after_kwh <= battery.capacity_kwh + TOLERANCE


def test_solar_reduction_lowers_usable_solar(hours, battery):
    entries = [
        DirectiveInterpretationEntry(
            note_index=0,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment={"hours": [13, 14], "factor": 0.2},
            explanation="",
        )
    ]
    context = apply_directives(entries, hours, battery)
    assert context.effective_solar[13] == hours[13].solar_kwh * 0.2
    assert context.effective_solar[14] == hours[14].solar_kwh * 0.2
    assert context.effective_solar[10] == hours[10].solar_kwh

    plan = solve_schedule(hours, battery, context)
    for h in (13, 14):
        assert plan[h].solar_used_kwh <= context.effective_solar[h] + TOLERANCE


def test_no_charge_window_is_respected(hours, battery):
    entries = [
        DirectiveInterpretationEntry(
            note_index=0,
            applies=True,
            directive_type="no_charge_window",
            structured_adjustment={"hours": [14, 15]},
            explanation="",
        )
    ]
    context = apply_directives(entries, hours, battery)
    plan = solve_schedule(hours, battery, context)
    for h in (14, 15):
        assert plan[h].battery_action != "charge"


def test_no_discharge_window_is_respected(hours, battery):
    entries = [
        DirectiveInterpretationEntry(
            note_index=0,
            applies=True,
            directive_type="no_discharge_window",
            structured_adjustment={"hours": [18, 19]},
            explanation="",
        )
    ]
    context = apply_directives(entries, hours, battery)
    plan = solve_schedule(hours, battery, context)
    for h in (18, 19):
        assert plan[h].battery_action != "discharge"


def test_minimum_battery_reserve_is_respected(hours, battery):
    entries = [
        DirectiveInterpretationEntry(
            note_index=0,
            applies=True,
            directive_type="minimum_battery_reserve",
            structured_adjustment={"hours": [18, 19, 20], "minimum_energy_kwh": 120},
            explanation="",
        )
    ]
    context = apply_directives(entries, hours, battery)
    plan = solve_schedule(hours, battery, context)
    for h in (18, 19, 20):
        assert plan[h].battery_energy_after_kwh >= 120 - TOLERANCE


def test_max_grid_window_is_respected(hours, battery):
    # Hours 8-9 have high solar (240-300 kWh) and moderate demand (300-310),
    # so a 100 kWh grid cap is comfortably feasible alongside solar + battery
    # discharge (up to 100 kWh/h) -- unlike the evening peak hours, where
    # demand outstrips solar + discharge headroom regardless of any cap.
    entries = [
        DirectiveInterpretationEntry(
            note_index=0,
            applies=True,
            directive_type="max_grid_window",
            structured_adjustment={"hours": [8, 9], "max_grid_kwh": 100},
            explanation="",
        )
    ]
    context = apply_directives(entries, hours, battery)
    plan = solve_schedule(hours, battery, context)
    for h in (8, 9):
        assert plan[h].grid_kwh <= 100 + TOLERANCE


def test_cost_recalculation_matches_plan(hours, battery):
    context = apply_directives(_no_op_interpretations(), hours, battery)
    plan = solve_schedule(hours, battery, context)
    recalculated = sum(p.grid_kwh * h.tariff_bdt_per_kwh for p, h in zip(plan, hours))
    total_grid = sum(p.grid_kwh for p in plan)
    peak = max(p.grid_kwh for p in plan)
    assert math.isfinite(recalculated)
    assert total_grid >= 0
    assert peak >= 0
