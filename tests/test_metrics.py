from __future__ import annotations

import json
from pathlib import Path

import pytest

from umafactor.evaluation.metrics import (
    BASIC_FIELDS,
    collect_golden_diffs,
    compare_golden,
    evaluate,
    filter_expected_rows,
    normalize_for_golden,
)


def test_evaluate_counts_field_accuracy_and_image_errors() -> None:
    expected_rows = [
        {
            "image_name": "a.png",
            "role": "main",
            "character": "A",
            "blue_type": "Speed",
            "blue_star": "3",
            "red_type": "Turf",
            "red_star": "2",
            "green_name": "Green",
            "green_star": "1",
        },
        {
            "image_name": "b.png",
            "role": "main",
            "character": "B",
            "blue_type": "Stamina",
            "blue_star": "2",
            "red_type": "Dirt",
            "red_star": "1",
            "green_name": "Other",
            "green_star": "0",
        },
    ]
    rec = {
        "a.png": {
            "main": {
                "character": "A",
                "blue": {"type": "Speed", "star": 3},
                "red": {"type": "Turf", "star": 1},
                "green": {"name": "Green", "star": 1},
            }
        },
        "b.png": {"error": "failed"},
    }

    metrics = evaluate(expected_rows, rec)

    assert metrics["total_expected_rows"] == 2
    assert metrics["total_checks"] == 14
    assert metrics["total_wrong"] == 7
    assert metrics["accuracy"] == pytest.approx(7 / 14)
    assert metrics["image_errors"] == {"b.png": "failed"}
    assert metrics["fields"]["red_star"]["wrong"] == 2
    assert {
        (failure["image_name"], failure["field"]) for failure in metrics["failures"]
    } >= {("a.png", "red_star"), ("b.png", "character")}


def test_basic_fields_exclude_green_name() -> None:
    assert [field for field, _slot, _attr in BASIC_FIELDS] == [
        "character",
        "blue_type",
        "blue_star",
        "red_type",
        "red_star",
        "green_star",
    ]


def test_filter_expected_rows_limits_to_result_images() -> None:
    rows = [
        {"image_name": "a.png", "role": "main"},
        {"image_name": "b.png", "role": "main"},
    ]

    assert filter_expected_rows(rows, {"b.png": {}}) == [
        {"image_name": "b.png", "role": "main"}
    ]


def test_normalize_for_golden_removes_volatile_keys_and_sorts_dicts() -> None:
    value = {
        "b": {"submitted_at": "now", "kept": 2},
        "a": {"submission_id": "id", "kept": 1},
    }

    assert list(normalize_for_golden(value).keys()) == ["a", "b"]
    assert normalize_for_golden(value) == {"a": {"kept": 1}, "b": {"kept": 2}}


def test_collect_golden_diffs_reports_missing_extra_value_and_length() -> None:
    diffs = collect_golden_diffs(
        {"same": 1, "missing": 2, "list": [1, 2], "changed": "old"},
        {"same": 1, "extra": 3, "list": [1], "changed": "new"},
    )

    assert {"kind": "missing", "path": "$.missing", "expected": 2, "actual": None} in diffs
    assert {"kind": "extra", "path": "$.extra", "expected": None, "actual": 3} in diffs
    assert {"kind": "length", "path": "$.list", "expected": 2, "actual": 1} in diffs
    assert {
        "kind": "value",
        "path": "$.changed",
        "expected": "old",
        "actual": "new",
    } in diffs


def test_compare_golden_ignores_volatile_keys() -> None:
    golden_path = Path("outputs") / "test_metrics_golden.json"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        golden_path.write_text(
            json.dumps(
                {
                    "a.png": {
                        "submission_id": "old",
                        "submitted_at": "old",
                        "main": {"character": "A"},
                    }
                }
            ),
            encoding="utf-8",
        )
        current = {
            "a.png": {
                "submission_id": "new",
                "submitted_at": "new",
                "main": {"character": "A"},
            }
        }

        result = compare_golden(current, golden_path)
    finally:
        if golden_path.exists():
            golden_path.unlink()

    assert result["matched"] is True
    assert result["diff_count"] == 0
