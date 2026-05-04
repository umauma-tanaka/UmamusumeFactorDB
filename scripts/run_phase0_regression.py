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
sys.path.insert(0, str(ROOT / "src"))

from umafactor import __version__  # noqa: E402
from umafactor.core.debug import (  # noqa: E402
    DebugManifest,
    StageDebug,
    write_debug_manifest,
)
from umafactor.evaluation.metrics import (  # noqa: E402
    BASIC_FIELDS,
    FIELDS,
    compare_golden,
    evaluate,
    filter_expected_rows,
    normalize_for_golden,
)

BASIC_PYTEST_K_EXPR = (
    "test_character or test_blue_type or test_blue_star or "
    "test_red_type or test_red_star or test_green_star"
)


def _load_expected(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _relative_artifact(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _build_debug_manifest(
    *,
    run_id: str,
    output_dir: Path,
    results_path: Path,
    expected_path: Path,
    golden_path: Path,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    env_check: subprocess.CompletedProcess[str],
    refresh_result: subprocess.CompletedProcess[str] | None,
    pytest_result: dict[str, Any] | None,
    golden_result: dict[str, Any] | None,
) -> DebugManifest:
    stages = [
        StageDebug(
            name="environment_check",
            status="ok" if env_check.returncode == 0 else "failed",
            artifacts={
                "json": _relative_artifact(output_dir / "env_check.json"),
                "log": _relative_artifact(output_dir / "env_check.log"),
            },
            metrics={"returncode": env_check.returncode},
        )
    ]
    if args.refresh:
        stages.append(
            StageDebug(
                name="recognition_refresh",
                status=(
                    "ok"
                    if refresh_result is not None and refresh_result.returncode == 0
                    else "failed"
                ),
                artifacts={
                    "log": _relative_artifact(output_dir / "batch_recognize.log"),
                    "results": _relative_artifact(results_path),
                },
                metrics={
                    "returncode": refresh_result.returncode
                    if refresh_result is not None
                    else None
                },
            )
        )
    else:
        stages.append(StageDebug(name="recognition_refresh", status="skipped"))

    evaluation_ok = (
        bool(golden_result and golden_result["matched"])
        if golden_result is not None
        else metrics.get("total_wrong") == 0 and metrics.get("image_error_count") == 0
    )
    evaluation_artifacts = {
        "metrics": _relative_artifact(output_dir / "metrics.json"),
        "summary": _relative_artifact(output_dir / "summary.md"),
        "results": _relative_artifact(output_dir / "recognition_results.json"),
    }
    if golden_result is not None:
        evaluation_artifacts["golden_diff"] = _relative_artifact(
            output_dir / "golden_diff.json"
        )
    else:
        evaluation_artifacts["failures"] = _relative_artifact(output_dir / "failures.json")
    stages.append(
        StageDebug(
            name="evaluation",
            status="ok" if evaluation_ok else "failed",
            artifacts=evaluation_artifacts,
            metrics={
                "profile": metrics.get("evaluation_profile"),
                "golden_matched": metrics.get("golden_matched"),
                "golden_diff_count": metrics.get("golden_diff_count"),
                "total_wrong": metrics.get("total_wrong"),
                "image_error_count": metrics.get("image_error_count"),
            },
        )
    )
    if pytest_result is not None:
        stages.append(
            StageDebug(
                name="pytest",
                status="ok" if pytest_result["returncode"] == 0 else "failed",
                artifacts={"log": _relative_artifact(output_dir / "pytest.log")},
                metrics={"returncode": pytest_result["returncode"]},
                message=pytest_result["command"],
            )
        )
    else:
        stages.append(StageDebug(name="pytest", status="skipped"))

    input_files = [_relative_artifact(results_path)]
    if args.compare_golden:
        input_files.append(_relative_artifact(golden_path))
    else:
        input_files.append(_relative_artifact(expected_path))

    return DebugManifest(
        run_id=run_id,
        case_id="phase0_regression",
        input_files=input_files,
        pipeline_version=__version__,
        model_versions={
            "star_classifier": str(metrics.get("star_classifier_mode", "")),
            "ocr": str(metrics.get("ocr_mode", "")),
        },
        stages=stages,
        metrics={
            "evaluation_profile": metrics.get("evaluation_profile"),
            "golden_matched": metrics.get("golden_matched"),
            "golden_diff_count": metrics.get("golden_diff_count"),
            "total_wrong": metrics.get("total_wrong"),
            "image_error_count": metrics.get("image_error_count"),
        },
    )


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

    refresh_result = None
    if args.refresh:
        refresh_cmd = [sys.executable, "scripts/batch_recognize.py"]
        if args.limit is not None:
            refresh_cmd.extend(["--limit", str(args.limit)])
        if results_path != DEFAULT_RESULTS:
            refresh_cmd.extend(["--output", str(results_path)])
        if args.skip_ocr:
            refresh_cmd.append("--skip-ocr")
        refresh_result = run_command(refresh_cmd, env=base_env)
        (output_dir / "batch_recognize.log").write_text(
            refresh_result.stdout,
            encoding="utf-8",
        )
        if refresh_result.returncode != 0:
            print(refresh_result.stdout)
            print(f"batch recognition failed; see {output_dir / 'batch_recognize.log'}")
            return refresh_result.returncode

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
    write_debug_manifest(
        output_dir / "debug_manifest.json",
        _build_debug_manifest(
            run_id=run_id,
            output_dir=output_dir,
            results_path=results_path,
            expected_path=expected_path,
            golden_path=golden_path,
            args=args,
            metrics=metrics,
            env_check=env_check,
            refresh_result=refresh_result,
            pytest_result=pytest_result,
            golden_result=golden_result,
        ),
    )

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
