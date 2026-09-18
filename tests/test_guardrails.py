from app.guardrails import validate_interpretation


def test_valid_solar_reduction_passes_through():
    raw = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "hours": [14, 13],  # deliberately unsorted to test normalization
            "factor": 0.2,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "panel cleaning",
        }
    ]
    result = validate_interpretation(raw, num_notes=1, battery_capacity_kwh=500)
    assert len(result) == 1
    entry = result[0]
    assert entry.applies is True
    assert entry.directive_type == "solar_reduction"
    assert entry.structured_adjustment == {"hours": [13, 14], "factor": 0.2}


def test_unsupported_directive_type_becomes_no_op():
    raw = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "shutdown_campus",
            "hours": [1, 2],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "invented type",
        }
    ]
    result = validate_interpretation(raw, num_notes=1, battery_capacity_kwh=500)
    assert result[0].directive_type == "no_op"
    assert result[0].applies is False
    assert result[0].structured_adjustment is None


def test_missing_hours_on_non_no_op_falls_back_to_no_op():
    raw = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "hours": [],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "",
        }
    ]
    result = validate_interpretation(raw, num_notes=1, battery_capacity_kwh=500)
    assert result[0].directive_type == "no_op"


def test_out_of_range_solar_factor_falls_back_to_no_op():
    raw = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "hours": [10],
            "factor": 1.5,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "",
        }
    ]
    result = validate_interpretation(raw, num_notes=1, battery_capacity_kwh=500)
    assert result[0].directive_type == "no_op"


def test_reserve_exceeding_capacity_falls_back_to_no_op():
    raw = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "hours": [5, 6],
            "factor": None,
            "minimum_energy_kwh": 9999,
            "max_grid_kwh": None,
            "explanation": "",
        }
    ]
    result = validate_interpretation(raw, num_notes=1, battery_capacity_kwh=500)
    assert result[0].directive_type == "no_op"


def test_missing_note_gets_safe_no_op_and_order_is_preserved():
    raw = [
        {
            "note_index": 1,
            "applies": False,
            "directive_type": "no_op",
            "hours": [],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "cafeteria menu",
        }
    ]
    result = validate_interpretation(raw, num_notes=3, battery_capacity_kwh=500)
    assert [e.note_index for e in result] == [0, 1, 2]
    assert result[0].directive_type == "no_op"
    assert result[2].directive_type == "no_op"


def test_duplicate_note_index_only_kept_once():
    raw = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "hours": [1],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "first",
        },
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_discharge_window",
            "hours": [2],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": None,
            "explanation": "duplicate, should be ignored",
        },
    ]
    result = validate_interpretation(raw, num_notes=1, battery_capacity_kwh=500)
    assert len(result) == 1
    assert result[0].directive_type == "no_charge_window"


def test_applies_semantics_no_op_only_false():
    raw = [
        {
            "note_index": 0,
            "applies": False,
            "directive_type": "max_grid_window",
            "hours": [1, 2],
            "factor": None,
            "minimum_energy_kwh": None,
            "max_grid_kwh": 50,
            "explanation": "model incorrectly said applies=false",
        }
    ]
    result = validate_interpretation(raw, num_notes=1, battery_capacity_kwh=500)
    # Non-no_op directives are always forced to applies=True per the schema.
    assert result[0].directive_type == "max_grid_window"
    assert result[0].applies is True
