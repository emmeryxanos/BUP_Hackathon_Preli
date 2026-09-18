"""Shared constants describing the supported operator-note directive types.

This is the single source of truth for directive names and their required
structured_adjustment shape, per the Problem Statement Section 04. Both the
LLM prompt/schema and the guardrail validator import from here so the two
can never drift apart.
"""

SOLAR_REDUCTION = "solar_reduction"
MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
NO_CHARGE_WINDOW = "no_charge_window"
NO_DISCHARGE_WINDOW = "no_discharge_window"
MAX_GRID_WINDOW = "max_grid_window"
NO_OP = "no_op"

ALL_DIRECTIVE_TYPES = frozenset(
    {
        SOLAR_REDUCTION,
        MINIMUM_BATTERY_RESERVE,
        NO_CHARGE_WINDOW,
        NO_DISCHARGE_WINDOW,
        MAX_GRID_WINDOW,
        NO_OP,
    }
)

# Directive types that require a non-empty "hours" list.
HOURS_REQUIRED_TYPES = ALL_DIRECTIVE_TYPES - {NO_OP}
