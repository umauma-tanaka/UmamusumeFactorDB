"""Benchmark factor-list OCR execution modes on a stitched sample.

The benchmark intentionally evaluates the full factor-list pipeline while
scoring only the parent role against expected_ocr.csv, because the current
fixture contains parent ground truth only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from umafactor.evaluation.ocr_dataset import (  # noqa: E402
    evaluate_ocr_factors,
    load_expected_ocr_factors,
)
from umafactor.factor_list import FactorOcrOptions, recognize_factor_list_image  # noqa: E402


@dataclass(frozen=True)
class _DetectedForEval:
    order: int
    raw_name: str
    star: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark factor-list OCR modes.")
    parser.add_argument(
        "--case",
        type=Path,
        default=ROOT / "datasets" / "test_factor_01",
        help="Case directory containing expected_stitched.png and expected_ocr.csv.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Stitched image path. Defaults to expected_stitched.png in --case.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="Expected parent OCR CSV. Defaults to expected_ocr.csv in --case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "ocr_mode_benchmark",
        help="Output directory for summary.json.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "rapidocr_batch_single_candidate",
            "rapidocr_batch_single_candidate_w320",
            "rapidocr_batch_single_candidate_w640",
            "rapidocr_batch_single_candidate_v5_ch_w480",
            "batch_recognition_single_candidate",
            "role_sheet_single_candidate_12",
            "role_sheet_single_candidate_8",
        ],
        help="Variant names to run.",
    )
    args = parser.parse_args()

    case_dir = _resolve_path(args.case)
    image_path = _resolve_path(args.image) if args.image else case_dir / "expected_stitched.png"
    expected_path = _resolve_path(args.expected) if args.expected else case_dir / "expected_ocr.csv"
    output_dir = _resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = load_expected_ocr_factors(expected_path)
    summary: dict[str, object] = {}
    for variant_name in args.variants:
        options = _variant_options(variant_name)
        start = time.perf_counter()
        try:
            result = recognize_factor_list_image(image_path, options=options)
            elapsed = time.perf_counter() - start
            entry = _summarize_result(result, expected)
            entry["elapsed_sec"] = elapsed
            entry["options"] = _options_summary(options)
        except Exception as exc:  # pragma: no cover - diagnostic CLI
            entry = {
                "elapsed_sec": time.perf_counter() - start,
                "error": repr(exc),
            }
        summary[variant_name] = entry
        (output_dir / "summary_partial.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _variant_options(name: str) -> FactorOcrOptions:
    common_single_candidate = {
        "ocr_roi_profiles": ("body_name",),
        "ocr_preprocess_modes": ("raw_upscaled", "gray_sharpen", "color_text_safe"),
    }
    if name.startswith("rapidocr_batch_single_candidate"):
        width = 480
        version = "PP-OCRv4"
        lang = "japan"
        if name.endswith("_w320"):
            width = 320
        elif name.endswith("_w640"):
            width = 640
        if "_v5_ch" in name:
            version = "PP-OCRv5"
            lang = "ch"
        return FactorOcrOptions(
            ocr_mode="rapidocr",
            ocr_execution_mode="batch",
            ocr_batch_size=12,
            rapidocr_ocr_version=version,
            rapidocr_lang_type=lang,
            rapidocr_model_type="mobile",
            rapidocr_rec_img_shape=(3, 48, width),
            rapidocr_rec_batch_num=12,
            **common_single_candidate,
        )
    if name == "batch_recognition_single_candidate":
        return FactorOcrOptions(
            ocr_mode="paddle",
            paddle_mode="recognition",
            ocr_execution_mode="batch",
            ocr_batch_size=12,
            **common_single_candidate,
        )
    if name == "role_sheet_single_candidate_12":
        return FactorOcrOptions(
            ocr_mode="paddle",
            paddle_mode="ocr",
            ocr_execution_mode="role_sheet",
            ocr_batch_size=12,
            ocr_sheet_max_side=3000,
            **common_single_candidate,
        )
    if name == "role_sheet_single_candidate_8":
        return FactorOcrOptions(
            ocr_mode="paddle",
            paddle_mode="ocr",
            ocr_execution_mode="role_sheet",
            ocr_batch_size=8,
            ocr_sheet_max_side=3000,
            **common_single_candidate,
        )
    if name == "role_sheet_single_candidate_all":
        return FactorOcrOptions(
            ocr_mode="paddle",
            paddle_mode="ocr",
            ocr_execution_mode="role_sheet",
            ocr_batch_size=0,
            ocr_sheet_max_side=3600,
            **common_single_candidate,
        )
    raise ValueError(f"unknown benchmark variant: {name}")


def _summarize_result(result, expected) -> dict[str, object]:
    role_counts: dict[str, int] = {}
    for factor in result.factors:
        role_counts[factor.role] = role_counts.get(factor.role, 0) + 1
    parent = sorted(
        [factor for factor in result.factors if factor.role == "parent"],
        key=lambda factor: factor.order,
    )
    detected = [
        _DetectedForEval(
            order=factor.order,
            raw_name=factor.normalized_name or factor.raw_name,
            star=factor.stars,
        )
        for factor in parent
    ]
    metrics = evaluate_ocr_factors(expected, detected, evaluate_names=True)
    return {
        "factor_count": len(result.factors),
        "role_counts": role_counts,
        "parent_name_correct": metrics["name_correct"],
        "parent_name_accuracy": metrics["name_accuracy"],
        "parent_star_correct": metrics["star_correct"],
        "parent_star_accuracy": metrics["star_accuracy"],
        "blank_name_count": metrics["blank_name_count"],
        "needs_review": sum(1 for factor in result.factors if factor.needs_review),
        "similarity_buckets": metrics["name_similarity_buckets"],
        "failures_head": metrics["failures"][:8],
    }


def _options_summary(options: FactorOcrOptions) -> dict[str, object]:
    return {
        "ocr_mode": options.ocr_mode,
        "paddle_mode": options.paddle_mode,
        "ocr_execution_mode": options.ocr_execution_mode,
        "ocr_batch_size": options.ocr_batch_size,
        "ocr_sheet_max_side": options.ocr_sheet_max_side,
        "ocr_sheet_columns": options.ocr_sheet_columns,
        "ocr_roi_profiles": list(options.ocr_roi_profiles),
        "ocr_preprocess_modes": list(options.ocr_preprocess_modes),
        "rapidocr_ocr_version": options.rapidocr_ocr_version,
        "rapidocr_lang_type": options.rapidocr_lang_type,
        "rapidocr_model_type": options.rapidocr_model_type,
        "rapidocr_rec_img_shape": list(options.rapidocr_rec_img_shape),
    }


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
