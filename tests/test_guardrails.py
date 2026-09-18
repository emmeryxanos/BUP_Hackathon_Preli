from app.interpreter.guardrails import validate_all, validate_directive


def test_valid_solar_reduction_passes_through():
    raw = {
        "note_index": 0,
        "directive_type": "solar_reduction",
        "applies": True,
        "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
        "explanation": "panel cleaning",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "solar_reduction"
    assert out["applies"] is True
    assert out["structured_adjustment"]["hours"] == [13, 14]
    assert out["structured_adjustment"]["factor"] == 0.2


def test_no_op_note_normalized():
    raw = {
        "note_index": 1,
        "directive_type": "no_op",
        "applies": False,
        "structured_adjustment": None,
        "explanation": "irrelevant",
    }
    out = validate_directive(raw, 1, battery_capacity_kwh=500)
    assert out["applies"] is False
    assert out["directive_type"] == "no_op"
    assert out["structured_adjustment"] is None


def test_unsupported_directive_type_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "shutdown_grid",  # not in Section 04
        "applies": True,
        "structured_adjustment": {"hours": [1]},
        "explanation": "invented",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"
    assert out["applies"] is False


def test_out_of_range_hours_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": [24, 25]},
        "explanation": "bad hours",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_unordered_or_duplicate_hours_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": [14, 13]},
        "explanation": "unordered",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"

    raw2 = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": [13, 13]},
        "explanation": "duplicate",
    }
    out2 = validate_directive(raw2, 0, battery_capacity_kwh=500)
    assert out2["directive_type"] == "no_op"


def test_factor_out_of_bounds_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "solar_reduction",
        "applies": True,
        "structured_adjustment": {"hours": [1], "factor": 1.5},
        "explanation": "bad factor",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_negative_reserve_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "minimum_battery_reserve",
        "applies": True,
        "structured_adjustment": {"hours": [1], "minimum_energy_kwh": -50},
        "explanation": "negative",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_reserve_exceeding_capacity_clamped_not_rejected():
    raw = {
        "note_index": 0,
        "directive_type": "minimum_battery_reserve",
        "applies": True,
        "structured_adjustment": {"hours": [1], "minimum_energy_kwh": 9999},
        "explanation": "too high",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "minimum_battery_reserve"
    assert out["structured_adjustment"]["minimum_energy_kwh"] == 500


def test_missing_required_key_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "max_grid_window",
        "applies": True,
        "structured_adjustment": {"hours": [1]},  # missing max_grid_kwh
        "explanation": "incomplete",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_malformed_entry_falls_back_to_no_op():
    out = validate_directive("not a dict", 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"
    assert out["applies"] is False


def test_validate_all_fills_missing_notes_with_no_op():
    raw_entries = [
        {
            "note_index": 0,
            "directive_type": "no_charge_window",
            "applies": True,
            "structured_adjustment": {"hours": [1, 2]},
            "explanation": "ok",
        }
    ]
    out = validate_all(raw_entries, num_notes=3, battery_capacity_kwh=500)
    assert len(out) == 3
    assert [e["note_index"] for e in out] == [0, 1, 2]
    assert out[1]["directive_type"] == "no_op"
    assert out[2]["directive_type"] == "no_op"


def test_validate_all_drops_duplicate_note_index():
    raw_entries = [
        {
            "note_index": 0,
            "directive_type": "no_op",
            "applies": False,
            "structured_adjustment": None,
            "explanation": "first",
        },
        {
            "note_index": 0,
            "directive_type": "no_charge_window",
            "applies": True,
            "structured_adjustment": {"hours": [1]},
            "explanation": "duplicate index, ignored",
        },
    ]
    out = validate_all(raw_entries, num_notes=1, battery_capacity_kwh=500)
    assert len(out) == 1
    assert out[0]["directive_type"] == "no_op"


def test_validate_all_empty_llm_output_produces_all_no_op():
    out = validate_all([], num_notes=3, battery_capacity_kwh=500)
    assert len(out) == 3
    assert all(e["directive_type"] == "no_op" and e["applies"] is False for e in out)


def test_hours_as_numeric_strings_coerced():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": ["1", "2"]},
        "explanation": "string hours",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_charge_window"
    assert out["structured_adjustment"]["hours"] == [1, 2]


def test_non_numeric_hours_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": ["a", "b"]},
        "explanation": "bad",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_empty_hours_list_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": []},
        "explanation": "empty",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_hours_not_a_list_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": "not-a-list"},
        "explanation": "wrong type",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_structured_adjustment_not_a_dict_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": "not-a-dict",
        "explanation": "wrong type",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_negative_max_grid_kwh_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "max_grid_window",
        "applies": True,
        "structured_adjustment": {"hours": [1], "max_grid_kwh": -10},
        "explanation": "negative",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_nan_factor_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "directive_type": "solar_reduction",
        "applies": True,
        "structured_adjustment": {"hours": [1], "factor": float("nan")},
        "explanation": "nan",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_boundary_factor_values_accepted():
    for factor in (0.0, 1.0):
        raw = {
            "note_index": 0,
            "directive_type": "solar_reduction",
            "applies": True,
            "structured_adjustment": {"hours": [1], "factor": factor},
            "explanation": "boundary",
        }
        out = validate_directive(raw, 0, battery_capacity_kwh=500)
        assert out["directive_type"] == "solar_reduction"
        assert out["structured_adjustment"]["factor"] == factor


def test_missing_directive_type_falls_back_to_no_op():
    raw = {
        "note_index": 0,
        "applies": True,
        "structured_adjustment": {"hours": [1]},
        "explanation": "no type field",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_op"


def test_explanation_truncated_to_500_chars():
    raw = {
        "note_index": 0,
        "directive_type": "no_op",
        "applies": False,
        "structured_adjustment": None,
        "explanation": "x" * 1000,
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert len(out["explanation"]) == 500


def test_hour_at_upper_and_lower_bound_accepted():
    raw = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "applies": True,
        "structured_adjustment": {"hours": [0, 23]},
        "explanation": "boundary hours",
    }
    out = validate_directive(raw, 0, battery_capacity_kwh=500)
    assert out["directive_type"] == "no_charge_window"
    assert out["structured_adjustment"]["hours"] == [0, 23]


def test_validate_all_out_of_range_note_index_dropped():
    raw_entries = [
        {
            "note_index": 5,  # out of range for num_notes=2
            "directive_type": "no_charge_window",
            "applies": True,
            "structured_adjustment": {"hours": [1]},
            "explanation": "out of range",
        }
    ]
    out = validate_all(raw_entries, num_notes=2, battery_capacity_kwh=500)
    assert len(out) == 2
    assert all(e["directive_type"] == "no_op" for e in out)


def test_validate_all_non_integer_note_index_falls_back_to_position():
    raw_entries = [
        {
            "note_index": "not-a-number",
            "directive_type": "no_charge_window",
            "applies": True,
            "structured_adjustment": {"hours": [1]},
            "explanation": "bad index type",
        }
    ]
    out = validate_all(raw_entries, num_notes=1, battery_capacity_kwh=500)
    assert len(out) == 1
    assert out[0]["note_index"] == 0
    assert out[0]["directive_type"] == "no_charge_window"


def test_validate_all_preserves_order_for_multiple_valid_notes():
    raw_entries = [
        {
            "note_index": 1,
            "directive_type": "no_charge_window",
            "applies": True,
            "structured_adjustment": {"hours": [5]},
            "explanation": "second note",
        },
        {
            "note_index": 0,
            "directive_type": "no_discharge_window",
            "applies": True,
            "structured_adjustment": {"hours": [6]},
            "explanation": "first note",
        },
    ]
    out = validate_all(raw_entries, num_notes=2, battery_capacity_kwh=500)
    assert [e["note_index"] for e in out] == [0, 1]
    assert out[0]["directive_type"] == "no_discharge_window"
    assert out[1]["directive_type"] == "no_charge_window"
