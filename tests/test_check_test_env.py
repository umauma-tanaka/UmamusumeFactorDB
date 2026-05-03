from __future__ import annotations

import pytest

import scripts.check_test_env as check_test_env
from umafactor.recognition.model_registry import ModelValidationResult


def test_check_models_uses_registry_optional_star_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_star = (
        check_test_env.ROOT
        / "models"
        / "modules"
        / "missing_star_classifier"
        / "prediction.onnx"
    )

    monkeypatch.setattr(check_test_env, "_module_zip_entries", lambda: set())
    monkeypatch.setattr(
        check_test_env,
        "validate_required_models",
        lambda allow_missing_star_classifier=False: [
            ModelValidationResult(
                model_name="star_classifier",
                path=missing_star,
                exists=False,
                required=False,
            )
        ],
    )

    results = check_test_env.check_models(allow_missing_star_classifier=True)

    assert len(results) == 1
    assert results[0].status == "WARN"
    assert results[0].required is False
    assert "HSV fallback" in results[0].detail


def test_build_report_can_include_model_io(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = check_test_env.CheckResult("check", "OK", "ok")

    monkeypatch.setattr(check_test_env, "check_imports", lambda skip_ocr=False: [ok])
    monkeypatch.setattr(check_test_env, "check_files", lambda: [])
    monkeypatch.setattr(check_test_env, "check_easyocr_models", lambda: [])
    monkeypatch.setattr(check_test_env, "check_models", lambda allow_missing_star_classifier=False: [])
    monkeypatch.setattr(check_test_env, "check_fixtures", lambda: [])
    monkeypatch.setattr(
        check_test_env,
        "describe_loaded_model_io",
        lambda: {
            "factor": {
                "input_name": "images",
                "input_shape": ["batch", 16, 168, 3],
                "output_names": ["index", "confidence"],
            }
        },
    )

    report = check_test_env.build_report(include_model_io=True)

    assert report["ok"] is True
    assert report["model_io"]["factor"]["input_name"] == "images"
