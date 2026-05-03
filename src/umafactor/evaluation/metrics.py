"""Metrics and golden comparison helpers for recognition results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VOLATILE_KEYS = {"submission_id", "submitted_at"}

FIELDS = [
    ("character", "character", None),
    ("blue_type", "blue", "type"),
    ("blue_star", "blue", "star"),
    ("red_type", "red", "type"),
    ("red_star", "red", "star"),
    ("green_name", "green", "name"),
    ("green_star", "green", "star"),
]
BASIC_FIELDS = [field for field in FIELDS if field[0] != "green_name"]


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_rec_value(
    rec: dict[str, Any],
    image: str,
    role: str,
    slot: str,
    attr: str | None,
) -> Any:
    image_rec = rec.get(image, {})
    if "error" in image_rec:
        return None
    role_rec = image_rec.get(role, {}) or {}
    if attr is None:
        return role_rec.get(slot, "")
    slot_rec = role_rec.get(slot, {}) or {}
    return slot_rec.get(attr, "")


def _normalize_expected(value: str, field: str) -> Any:
    if field.endswith("_star"):
        try:
            return int(value or 0)
        except ValueError:
            return 0
    return value or ""


def _normalize_actual(value: Any, field: str) -> Any:
    if field.endswith("_star"):
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    return value or ""


def evaluate(
    expected_rows: list[dict[str, str]],
    rec: dict[str, Any],
    fields: list[tuple[str, str, str | None]] | None = None,
) -> dict[str, Any]:
    fields = fields or FIELDS
    totals = {field: 0 for field, _slot, _attr in fields}
    correct = {field: 0 for field, _slot, _attr in fields}
    failures: list[dict[str, Any]] = []
    image_errors = {
        image: data.get("error")
        for image, data in rec.items()
        if isinstance(data, dict) and data.get("error")
    }

    for row in expected_rows:
        image = row["image_name"]
        role = row["role"]
        for field, slot, attr in fields:
            expected = _normalize_expected(row.get(field, ""), field)
            actual = _normalize_actual(_get_rec_value(rec, image, role, slot, attr), field)
            totals[field] += 1
            if actual == expected:
                correct[field] += 1
            else:
                failures.append(
                    {
                        "image_name": image,
                        "role": role,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )

    field_metrics = {}
    for field in totals:
        total = totals[field]
        ok = correct[field]
        field_metrics[field] = {
            "total": total,
            "correct": ok,
            "wrong": total - ok,
            "accuracy": ok / total if total else 0.0,
        }

    total_checks = sum(totals.values())
    total_correct = sum(correct.values())
    return {
        "total_expected_rows": len(expected_rows),
        "total_checks": total_checks,
        "total_correct": total_correct,
        "total_wrong": total_checks - total_correct,
        "accuracy": total_correct / total_checks if total_checks else 0.0,
        "image_error_count": len(image_errors),
        "image_errors": image_errors,
        "fields": field_metrics,
        "failures": failures,
    }


def filter_expected_rows(
    expected_rows: list[dict[str, str]],
    rec: dict[str, Any],
) -> list[dict[str, str]]:
    images = set(rec.keys())
    return [row for row in expected_rows if row.get("image_name") in images]


def normalize_for_golden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_for_golden(child)
            for key, child in sorted(value.items())
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [normalize_for_golden(child) for child in value]
    return value


def collect_golden_diffs(
    expected: Any,
    actual: Any,
    path: str = "$",
    limit: int = 200,
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []

    def add(kind: str, at: str, expected_value: Any = None, actual_value: Any = None) -> None:
        if len(diffs) >= limit:
            return
        diffs.append(
            {
                "kind": kind,
                "path": at,
                "expected": expected_value,
                "actual": actual_value,
            }
        )

    def walk(left: Any, right: Any, at: str) -> None:
        if len(diffs) >= limit:
            return
        if type(left) is not type(right):
            add("type", at, type(left).__name__, type(right).__name__)
            return
        if isinstance(left, dict):
            left_keys = set(left)
            right_keys = set(right)
            for key in sorted(left_keys - right_keys):
                add("missing", f"{at}.{key}", left[key], None)
            for key in sorted(right_keys - left_keys):
                add("extra", f"{at}.{key}", None, right[key])
            for key in sorted(left_keys & right_keys):
                walk(left[key], right[key], f"{at}.{key}")
            return
        if isinstance(left, list):
            if len(left) != len(right):
                add("length", at, len(left), len(right))
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                walk(left_item, right_item, f"{at}[{index}]")
            return
        if left != right:
            add("value", at, left, right)

    walk(expected, actual, path)
    if len(diffs) >= limit:
        diffs.append(
            {
                "kind": "truncated",
                "path": "$",
                "expected": f"diff output limited to {limit} entries",
                "actual": f"diff output limited to {limit} entries",
            }
        )
    return diffs


def compare_golden(current: dict[str, Any], golden_path: Path) -> dict[str, Any]:
    golden = _load_results(golden_path)
    expected = normalize_for_golden(golden)
    actual = normalize_for_golden(current)
    diffs = collect_golden_diffs(expected, actual)
    image_errors = {
        image: data.get("error")
        for image, data in actual.items()
        if isinstance(data, dict) and data.get("error")
    }
    return {
        "profile": "golden",
        "golden_path": str(golden_path),
        "matched": not diffs,
        "total_images": len(actual),
        "golden_images": len(expected),
        "diff_count": len(diffs),
        "diffs": diffs,
        "image_error_count": len(image_errors),
        "image_errors": image_errors,
    }

