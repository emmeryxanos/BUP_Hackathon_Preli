"""Prompt content for LLM operator-note interpretation."""

SYSTEM_PROMPT = """You are an energy-operations note interpreter for a campus microgrid \
scheduling system called GridWise. You convert short natural-language operator notes into \
STRICT structured directives that a downstream optimizer will apply mechanically.

Supported directive types (use exactly one per note):

1. solar_reduction - usable solar is reduced during specific hours.
   structured_adjustment: {"hours": [int...], "factor": number}
   factor is the FRACTION OF SOLAR THAT REMAINS (not the reduction amount).
   Example: "drops to 20%" or "80% reduction" both mean factor = 0.2.

2. minimum_battery_reserve - battery must stay at or above a level during specific hours.
   structured_adjustment: {"hours": [int...], "minimum_energy_kwh": number}

3. no_charge_window - battery charging is unavailable during specific hours.
   structured_adjustment: {"hours": [int...]}

4. no_discharge_window - battery discharging is unavailable during specific hours.
   structured_adjustment: {"hours": [int...]}

5. max_grid_window - grid import is capped during specific hours.
   structured_adjustment: {"hours": [int...], "max_grid_kwh": number}

6. no_op - the note does NOT affect the 24-hour energy schedule (distractor, irrelevant \
chit-chat, unrelated logistics, etc.). structured_adjustment: null.

RULES:
- Time windows are HALF-OPEN: "1 PM to 3 PM" means hours [13, 14] (3 PM itself is excluded).
  Convert any clock time, 24-hour time, or relative time phrase to whole-hour integers 0-23.
- hours arrays must be unique integers 0-23 in ascending order.
- Never invent demand, tariff, or battery capacity/rate values. Only extract what the note says.
- Never invent a directive type that is not in the list above.
- If a note is ambiguous, irrelevant to the current day's schedule, or purely informational
  (e.g. "the cafeteria menu changes tomorrow", "the manager is on leave"), classify it as no_op.
- Every note must map to EXACTLY ONE directive type.
- For no_op: applies must be false and structured_adjustment must be null.
- For every other type: applies must be true and structured_adjustment must be fully populated.

You will be given a numbered list of operator notes. For each note, in order, produce one \
interpretation entry using the interpret_notes tool. Do not skip or merge notes."""


def build_user_message(operator_notes: list[str]) -> str:
    lines = ["Interpret the following operator notes for today's 24-hour schedule:"]
    for i, note in enumerate(operator_notes):
        lines.append(f"{i}. {note}")
    lines.append(
        "\nReturn exactly one directive entry per note above, in note_index order "
        "(0-based), using the interpret_notes tool."
    )
    return "\n".join(lines)


TOOL_SCHEMA = {
    "name": "interpret_notes",
    "description": "Return one structured directive interpretation per operator note.",
    "input_schema": {
        "type": "object",
        "properties": {
            "directives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "note_index": {"type": "integer"},
                        "directive_type": {
                            "type": "string",
                            "enum": [
                                "solar_reduction",
                                "minimum_battery_reserve",
                                "no_charge_window",
                                "no_discharge_window",
                                "max_grid_window",
                                "no_op",
                            ],
                        },
                        "applies": {"type": "boolean"},
                        "structured_adjustment": {
                            "type": ["object", "null"],
                            "properties": {
                                "hours": {"type": "array", "items": {"type": "integer"}},
                                "factor": {"type": "number"},
                                "minimum_energy_kwh": {"type": "number"},
                                "max_grid_kwh": {"type": "number"},
                            },
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": ["note_index", "directive_type", "applies", "explanation"],
                },
            }
        },
        "required": ["directives"],
    },
}
