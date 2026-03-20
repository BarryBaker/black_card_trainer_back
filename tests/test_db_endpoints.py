from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Allow running this file directly: `python tests/test_db_endpoints.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from api.db_endpoints import get_board_filter_clauses, get_board_filter_columns


def test_flop_board_filters_only_use_flop_columns():
    assert get_board_filter_columns("flop") == (
        ("flopSuited", "is_flopsuited"),
        ("flopPaired", "is_floppaired"),
        ("flopStraight", "is_flopstr8"),
        ("flopFlush", "is_flopflush"),
    )


def test_turn_board_filters_use_flop_and_turn_columns():
    assert get_board_filter_columns("turn") == (
        ("flopSuited", "is_flopsuited"),
        ("flopPaired", "is_floppaired"),
        ("flopStraight", "is_flopstr8"),
        ("flopFlush", "is_flopflush"),
        ("turnSuited", "is_suited"),
        ("turnPaired", "is_paired"),
        ("turnStraight", "is_str8"),
        ("turnFlush", "is_flush"),
    )


def test_board_filter_clauses_skip_nulls_and_keep_street_order():
    clauses, params = get_board_filter_clauses(
        "turn",
        {
            "flopPaired": True,
            "flopStraight": None,
            "flopFlush": False,
            "turnPaired": True,
            "turnStraight": False,
            "turnFlush": None,
        },
    )

    assert len(clauses) == 4
    assert params == [True, False, True, False]


def test_board_filter_clauses_require_object_payload():
    with pytest.raises(ValueError, match="Field 'board' must be an object"):
        get_board_filter_clauses("flop", [])


def test_board_filter_clauses_require_boolean_or_null_values():
    with pytest.raises(
        ValueError, match="Board field 'flopPaired' must be boolean or null"
    ):
        get_board_filter_clauses("flop", {"flopPaired": "yes"})
