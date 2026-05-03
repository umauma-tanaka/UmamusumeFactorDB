"""Phase 0 test environment checker.

This script verifies whether the current workspace can generate and run the
recognition regression baseline. It does not download dependencies or mutate
files.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"
RECOGNITION_RESULTS = FIXTURES_DIR / "colored_factors" / "recognition_results.json"
EXPECTED_LABELS = FIXTURES_DIR / "expected_labels.csv"
MODULES_ZIP = ROOT / "models" / "modules.zip"
EASYOCR_DIR = ROOT / "models" / "easyocr"
STAR_FALLBACK_ENV = "UMAFACTOR_ALLOW_MISSING_STAR_CLASSIFIER"

IMAGE_PATTERNS = [
    "receipt_*.png",
    "combine_*.png",
    "sample_*.png",
    "umamusume_*.png",
    "image0_*.png",
    "new_*.png",
    "unseen_*.png",
]

REQUIRED_IMPORTS = [
    ("cv2", "opencv-python-headless"),
    ("numpy", "numpy"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("easyocr", "easyocr"),
    ("pytest", "pytest"),
    ("rapidfuzz", "rapidfuzz"),
]

REQUIRED_FILES = [
    EXPECTED_LABELS,
    FIXTURES_DIR / "colored_factors" / "README.md",
    ROOT / "config" / "recognizer.json",
    ROOT / "config" / "scene_stitcher.json",
    ROOT / "models" / "modules" / "labels.json",
    ROOT / "models" / "modules" / "factor_info.json",
]

EASYOCR_MODEL_FILES = [
    EASYOCR_DIR / "craft_mlt_25k.pth",
    EASYOCR_DIR / "japanese_g2.pth",
]

REQUIRED_MODELS = [
    ROOT / "models" / "modules" / "factor" / "prediction.onnx",
    ROOT / "models" / "modules" / "factor_rank" / "prediction.onnx",
    ROOT / "models" / "modules" / "character" / "prediction.onnx",
    ROOT / "models" / "modules" / "star_classifier" / "prediction.onnx",
]
STAR_CLASSIFIER_MODEL = ROOT / "models" / "modules" / "star_classifier" / "prediction.onnx"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _module_zip_entries() -> set[str]:
    if not MODULES_ZIP.exists():
        return set()
    try:
        with zipfile.ZipFile(MODULES_ZIP) as zf:
            return set(zf.namelist())
    except zipfile.BadZipFile:
        return set()


def _zip_name_for_model(path: Path) -> str:
    rel = path.relative_to(ROOT / "models").as_posix()
    return rel


def _env_truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def check_imports(skip_ocr: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    for module_name, package_name in REQUIRED_IMPORTS:
        if skip_ocr and module_name == "easyocr":
            results.append(
                CheckResult(
                    name=f"import:{module_name}",
                    status="WARN",
                    detail="skipped because OCR is disabled for this run",
                    required=False,
                )
            )
            continue
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            results.append(
                CheckResult(
                    name=f"import:{module_name}",
                    status="FAIL",
                    detail=f"{package_name} is not importable",
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"import:{module_name}",
                    status="OK",
                    detail="importable",
                )
            )
    return results


def check_files() -> list[CheckResult]:
    results: list[CheckResult] = []
    for path in REQUIRED_FILES:
        results.append(
            CheckResult(
                name=f"file:{_rel(path)}",
                status="OK" if path.exists() else "FAIL",
                detail="exists" if path.exists() else "missing",
            )
        )

    if RECOGNITION_RESULTS.exists():
        detail = f"exists ({RECOGNITION_RESULTS.stat().st_size} bytes)"
        status = "OK"
    else:
        detail = "missing; run scripts/batch_recognize.py or scripts/run_phase0_regression.py --refresh"
        status = "WARN"
    results.append(
        CheckResult(
            name=f"file:{_rel(RECOGNITION_RESULTS)}",
            status=status,
            detail=detail,
            required=False,
        )
    )
    return results


def check_easyocr_models() -> list[CheckResult]:
    missing = [path for path in EASYOCR_MODEL_FILES if not path.exists()]
    if not missing:
        return [
            CheckResult(
                name=f"easyocr:{_rel(EASYOCR_DIR)}",
                status="OK",
                detail="required local EasyOCR model files exist",
                required=False,
            )
        ]
    return [
        CheckResult(
            name=f"easyocr:{_rel(EASYOCR_DIR)}",
            status="WARN",
            detail="missing " + ", ".join(_rel(path) for path in missing),
            required=False,
        )
    ]


def check_models(allow_missing_star_classifier: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    zip_entries = _module_zip_entries()
    for path in REQUIRED_MODELS:
        if path.exists():
            results.append(
                CheckResult(
                    name=f"model:{_rel(path)}",
                    status="OK",
                    detail="exists",
                )
            )
            continue

        zip_name = _zip_name_for_model(path)
        if zip_name in zip_entries:
            detail = f"missing; available in {_rel(MODULES_ZIP)} as {zip_name}"
        elif MODULES_ZIP.exists():
            detail = f"missing; not found in {_rel(MODULES_ZIP)}"
        else:
            detail = f"missing; {_rel(MODULES_ZIP)} is also missing"
        if path == STAR_CLASSIFIER_MODEL and allow_missing_star_classifier:
            results.append(
                CheckResult(
                    name=f"model:{_rel(path)}",
                    status="WARN",
                    detail=detail + "; HSV fallback is enabled for partial Phase 0 runs",
                    required=False,
                )
            )
            continue
        results.append(
            CheckResult(
                name=f"model:{_rel(path)}",
                status="FAIL",
                detail=detail,
            )
        )
    return results


def check_fixtures() -> list[CheckResult]:
    images: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        images.extend(FIXTURES_DIR.glob(pattern))
    images = sorted(set(images), key=lambda p: p.name)

    results = [
        CheckResult(
            name="fixtures:recognition_images",
            status="OK" if images else "FAIL",
            detail=f"{len(images)} images matched Phase 0 patterns",
        )
    ]

    if EXPECTED_LABELS.exists():
        with EXPECTED_LABELS.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        status = "OK" if rows else "FAIL"
        detail = f"{len(rows)} expected rows"
    else:
        status = "FAIL"
        detail = "expected_labels.csv is missing"
    results.append(
        CheckResult(
            name="fixtures:expected_labels",
            status=status,
            detail=detail,
        )
    )
    return results


def build_report(
    allow_missing_star_classifier: bool = False,
    skip_ocr: bool = False,
) -> dict:
    checks = []
    checks.extend(check_imports(skip_ocr=skip_ocr))
    checks.extend(check_files())
    if skip_ocr:
        checks.append(
            CheckResult(
                name=f"easyocr:{_rel(EASYOCR_DIR)}",
                status="WARN",
                detail="skipped because OCR is disabled for this run",
                required=False,
            )
        )
    else:
        checks.extend(check_easyocr_models())
    checks.extend(check_models(allow_missing_star_classifier))
    checks.extend(check_fixtures())
    has_failures = any(c.required and c.status == "FAIL" for c in checks)
    has_warnings = any(c.status == "WARN" for c in checks)
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "root": str(ROOT),
        "allow_missing_star_classifier": allow_missing_star_classifier,
        "skip_ocr": skip_ocr,
        "ok": not has_failures,
        "has_warnings": has_warnings,
        "checks": [asdict(c) for c in checks],
    }


def print_report(report: dict) -> None:
    print("Phase 0 test environment check")
    print(f"root: {report['root']}")
    print(f"python: {report['python']}")
    print("")
    for check in report["checks"]:
        marker = {
            "OK": "[OK]",
            "WARN": "[WARN]",
            "FAIL": "[FAIL]",
        }.get(check["status"], f"[{check['status']}]")
        print(f"{marker} {check['name']}: {check['detail']}")
    print("")
    if report["ok"]:
        print("result: OK")
    else:
        print("result: FAIL")
        print("action: install missing packages, extract ONNX models, or place missing model files.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="Write the full check report as JSON.")
    parser.add_argument(
        "--allow-missing-star-classifier",
        action="store_true",
        help="Downgrade missing star_classifier/prediction.onnx to WARN for partial Phase 0 runs.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Do not require EasyOCR import or local model files for this check.",
    )
    args = parser.parse_args()

    allow_missing_star_classifier = (
        args.allow_missing_star_classifier or _env_truthy(os.environ.get(STAR_FALLBACK_ENV))
    )
    report = build_report(allow_missing_star_classifier, skip_ocr=args.skip_ocr)
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
