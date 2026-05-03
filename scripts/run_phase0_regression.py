"""Run the Phase 0 recognition regression baseline.

The script can optionally refresh recognition_results.json, then either compares
the current recognition output with tests/fixtures/expected_labels.csv or checks
it against a golden snapshot for behavior-preserving refactors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "tests" / "fixtures" / "colored_factors" / "recognition_results.json"
DEFAULT_EXPECTED = ROOT / "tests" / "fixtures" / "expected_labels.csv"
DEFAULT_GOLDEN = ROOT / "tests" / "fixtures" / "colored_factors" / "phase0_golden_skip_ocr.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase0"
STAR_FALLBACK_ENV = "UMAFACTOR_ALLOW_MISSING_STAR_CLASSIFIER"
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
BASIC_PYTEST_K_EXPR = (
    "test_character or test_blue_type or test_blue_star or "
    "test_red_type or test_red_star or test_green_star"
)


def _load_expected(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_rec_value(rec: dict[str, Any], image: str, role: str, slot: str, attr: str | None) -> Any:
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


def filter_expected_rows(expected_rows: list[dict[str, str]], rec: dict[str, Any]) -> list[dict[str, str]]:
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


def collect_golden_diffs(expected: Any, actual: Any, path: str = "$", limit: int = 200) -> list[dict[str, Any]]:
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_summary_md(
    path: Path,
    metrics: dict[str, Any],
    pytest_result: dict[str, Any] | None,
    golden_result: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Phase 0 Regression Report",
        "",
        f"- star classifier mode: {metrics.get('star_classifier_mode', 'onnx')}",
        f"- OCR mode: {metrics.get('ocr_mode', 'enabled')}",
        f"- evaluation profile: {metrics.get('evaluation_profile', 'full')}",
    ]
    if golden_result is not None:
        lines.extend(
            [
                f"- golden matched: {golden_result['matched']}",
                f"- golden diffs: {golden_result['diff_count']}",
                f"- image errors: {golden_result['image_error_count']}",
                f"- golden path: `{golden_result['golden_path']}`",
            ]
        )
        if golden_result["diffs"]:
            lines.extend(["", "## Golden Diffs", ""])
            for diff in golden_result["diffs"][:20]:
                lines.append(
                    f"- {diff['kind']} `{diff['path']}`: "
                    f"expected={diff['expected']!r}, actual={diff['actual']!r}"
                )
    else:
        lines.extend(
            [
                f"- total expected rows: {metrics['total_expected_rows']}",
                f"- total checks: {metrics['total_checks']}",
                f"- total wrong: {metrics['total_wrong']}",
                f"- accuracy: {metrics['accuracy']:.2%}",
                f"- image errors: {metrics['image_error_count']}",
                "",
                "## Field Metrics",
                "",
                "| field | correct | total | wrong | accuracy |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for field, data in metrics["fields"].items():
            lines.append(
                f"| {field} | {data['correct']} | {data['total']} | "
                f"{data['wrong']} | {data['accuracy']:.2%} |"
            )
    if pytest_result is not None:
        lines.extend(
            [
                "",
                "## Pytest",
                "",
                f"- exit code: {pytest_result['returncode']}",
                f"- command: `{pytest_result['command']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _env_truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Regenerate recognition_results.json first.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-pytest", action="store_true", help="Only write metrics; do not run pytest.")
    parser.add_argument("--limit", type=int, help="Refresh only the first N fixture images for smoke runs.")
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Evaluate only images present in recognition_results.json.",
    )
    parser.add_argument(
        "--allow-missing-star-classifier",
        action="store_true",
        help="Use HSV star fallback for a partial baseline when star_classifier/prediction.onnx is absent.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip EasyOCR calls while refreshing recognition results.",
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Evaluate basic non-OCR-heavy fields only; currently excludes green_name.",
    )
    parser.add_argument(
        "--compare-golden",
        action="store_true",
        help="Compare refreshed or provided results with a golden snapshot instead of expected_labels.csv.",
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Write the normalized current results to --golden, then compare against it.",
    )
    args = parser.parse_args()
    if args.update_golden:
        args.compare_golden = True

    results_path = args.results
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    expected_path = args.expected
    if not expected_path.is_absolute():
        expected_path = ROOT / expected_path
    golden_path = args.golden
    if not golden_path.is_absolute():
        golden_path = ROOT / golden_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_results_path = output_dir / "recognition_results.json"
    if args.refresh and (args.limit is not None or args.skip_ocr) and results_path == DEFAULT_RESULTS:
        results_path = output_results_path

    allow_missing_star_classifier = (
        args.allow_missing_star_classifier or _env_truthy(os.environ.get(STAR_FALLBACK_ENV))
    )
    base_env = dict(os.environ)
    if allow_missing_star_classifier:
        base_env[STAR_FALLBACK_ENV] = "1"

    env_report = output_dir / "env_check.json"
    env_cmd = [sys.executable, "scripts/check_test_env.py", "--json", str(env_report)]
    if allow_missing_star_classifier:
        env_cmd.append("--allow-missing-star-classifier")
    if args.skip_ocr:
        env_cmd.append("--skip-ocr")
    env_check = run_command(env_cmd, env=base_env)
    (output_dir / "env_check.log").write_text(env_check.stdout, encoding="utf-8")
    if args.refresh and env_check.returncode != 0:
        print(env_check.stdout)
        print(f"environment check failed; see {output_dir / 'env_check.log'}")
        return env_check.returncode

    if args.refresh:
        refresh_cmd = [sys.executable, "scripts/batch_recognize.py"]
        if args.limit is not None:
            refresh_cmd.extend(["--limit", str(args.limit)])
        if results_path != DEFAULT_RESULTS:
            refresh_cmd.extend(["--output", str(results_path)])
        if args.skip_ocr:
            refresh_cmd.append("--skip-ocr")
        refresh = run_command(refresh_cmd, env=base_env)
        (output_dir / "batch_recognize.log").write_text(refresh.stdout, encoding="utf-8")
        if refresh.returncode != 0:
            print(refresh.stdout)
            print(f"batch recognition failed; see {output_dir / 'batch_recognize.log'}")
            return refresh.returncode

    if not results_path.exists():
        print(f"recognition results are missing: {results_path}")
        print("run with --refresh after resolving environment check failures.")
        return 2
    if not args.compare_golden and not expected_path.exists():
        print(f"expected labels are missing: {expected_path}")
        return 2

    rec = _load_results(results_path)
    metrics: dict[str, Any] = {}
    metrics["star_classifier_mode"] = (
        "hsv_fallback_partial" if allow_missing_star_classifier else "onnx"
    )
    metrics["ocr_mode"] = "skipped" if args.skip_ocr else "enabled"
    metrics["evaluation_profile"] = "golden" if args.compare_golden else (
        "basic_no_green_name" if args.basic_only else "full"
    )

    golden_result = None
    if args.compare_golden:
        normalized = normalize_for_golden(rec)
        if args.update_golden:
            write_json(golden_path, normalized)
        if not golden_path.exists():
            print(f"golden results are missing: {golden_path}")
            print("run with --update-golden to initialize the refactor baseline.")
            return 2
        golden_result = compare_golden(rec, golden_path)
        metrics.update(
            {
                "profile": "golden",
                "golden_path": str(golden_path),
                "golden_matched": golden_result["matched"],
                "golden_diff_count": golden_result["diff_count"],
                "total_images": golden_result["total_images"],
                "image_error_count": golden_result["image_error_count"],
                "image_errors": golden_result["image_errors"],
            }
        )
    else:
        expected_rows = _load_expected(expected_path)
        if args.allow_partial_results or args.limit is not None:
            expected_rows = filter_expected_rows(expected_rows, rec)
        fields = BASIC_FIELDS if args.basic_only else FIELDS
        metrics.update(evaluate(expected_rows, rec, fields=fields))
        metrics["evaluated_fields"] = [field for field, _slot, _attr in fields]

    write_json(output_dir / "metrics.json", metrics)
    if golden_result is not None:
        write_json(output_dir / "golden_diff.json", golden_result["diffs"])
    else:
        write_json(output_dir / "failures.json", metrics["failures"])
    if results_path != output_results_path:
        shutil.copy2(results_path, output_results_path)

    pytest_result = None
    if not args.skip_pytest and not args.compare_golden:
        env = dict(base_env)
        env["UMAFACTOR_RECOGNITION_RESULTS"] = str(results_path)
        pytest_cmd = [sys.executable, "-m", "pytest", "tests/test_recognition.py", "-q"]
        if args.basic_only:
            pytest_cmd.extend(["-k", BASIC_PYTEST_K_EXPR])
        pytest_proc = run_command(pytest_cmd, env=env)
        (output_dir / "pytest.log").write_text(pytest_proc.stdout, encoding="utf-8")
        pytest_result = {
            "returncode": pytest_proc.returncode,
            "command": " ".join(pytest_cmd),
        }

    write_summary_md(output_dir / "summary.md", metrics, pytest_result, golden_result)

    print(f"Phase 0 report: {output_dir}")
    if golden_result is not None:
        print(f"golden matched: {golden_result['matched']}")
        print(f"golden diffs: {golden_result['diff_count']}")
        print(f"image errors: {golden_result['image_error_count']}")
    else:
        print(f"accuracy: {metrics['accuracy']:.2%}")
        print(f"wrong: {metrics['total_wrong']} / {metrics['total_checks']}")
        print(f"image errors: {metrics['image_error_count']}")
    if pytest_result is not None:
        print(f"pytest exit code: {pytest_result['returncode']}")
        if pytest_result["returncode"] != 0:
            return int(pytest_result["returncode"])
    if golden_result is not None:
        return 0 if golden_result["matched"] else 1
    return 0 if metrics["total_wrong"] == 0 and metrics["image_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
