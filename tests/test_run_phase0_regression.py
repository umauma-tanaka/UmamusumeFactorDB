from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import scripts.run_phase0_regression as regression


def test_phase0_regression_full_mode_uses_default_fields(monkeypatch) -> None:
    base_path = Path("outputs") / "test_run_phase0_regression"
    if base_path.exists():
        shutil.rmtree(base_path)
    results_path = base_path / "recognition_results.json"
    expected_path = base_path / "expected_labels.csv"
    output_root = base_path / "phase0"
    base_path.mkdir(parents=True)
    results_path.write_text(json.dumps({"case.png": {}}), encoding="utf-8")
    expected_path.write_text(
        "image_name,role,character,blue_type,blue_star,red_type,red_star,green_name,green_star\n"
        "case.png,main,,,,,,,\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_command(command, env=None):
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    def fake_evaluate(expected_rows, rec, fields=None):
        captured["expected_rows"] = expected_rows
        captured["rec"] = rec
        captured["fields"] = fields
        return {
            "total_expected_rows": len(expected_rows),
            "total_checks": 0,
            "total_correct": 0,
            "total_wrong": 0,
            "accuracy": 0.0,
            "image_error_count": 0,
            "image_errors": {},
            "fields": {},
            "failures": [],
        }

    monkeypatch.setattr(regression, "run_command", fake_run_command)
    monkeypatch.setattr(regression, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase0_regression.py",
            "--results",
            str(results_path),
            "--expected",
            str(expected_path),
            "--output-root",
            str(output_root),
            "--skip-pytest",
        ],
    )

    try:
        assert regression.main() == 0
        assert captured["fields"] is regression.FIELDS
    finally:
        if base_path.exists():
            shutil.rmtree(base_path)
