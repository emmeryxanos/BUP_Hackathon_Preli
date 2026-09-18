"""Tests for app/interpreter/prompts.py::build_user_message and TOOL_SCHEMA shape.

Pure formatting logic with no prior coverage: the note numbering/ordering it
produces is what the LLM uses to map its response back to note_index, so an
off-by-one here would silently misalign every directive.
"""
from app.interpreter.prompts import TOOL_SCHEMA, build_user_message


def test_build_user_message_numbers_notes_from_zero():
    msg = build_user_message(["first note", "second note"])
    assert "0. first note" in msg
    assert "1. second note" in msg


def test_build_user_message_single_note():
    msg = build_user_message(["only note"])
    assert "0. only note" in msg
    assert "1." not in msg


def test_build_user_message_preserves_note_order():
    notes = ["alpha", "beta", "gamma"]
    msg = build_user_message(notes)
    assert msg.index("0. alpha") < msg.index("1. beta") < msg.index("2. gamma")


def test_build_user_message_includes_instruction_footer():
    msg = build_user_message(["a note"])
    assert "interpret_notes" in msg
    assert "note_index order" in msg


def test_tool_schema_enum_matches_guardrails_allowed_types():
    from app.interpreter.guardrails import ALLOWED_TYPES

    schema_enum = set(
        TOOL_SCHEMA["input_schema"]["properties"]["directives"]["items"]["properties"][
            "directive_type"
        ]["enum"]
    )
    assert schema_enum == ALLOWED_TYPES


def test_tool_schema_requires_core_fields():
    required = TOOL_SCHEMA["input_schema"]["properties"]["directives"]["items"]["required"]
    assert set(required) == {"note_index", "directive_type", "applies", "explanation"}
