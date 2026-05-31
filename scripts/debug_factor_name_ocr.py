"""Debug factor-name OCR on a stitched image.

This CLI exercises the production card-body detector and the recognition-only
factor-name OCR path.  It writes the same debug crops/CSV/contact sheet as the
factor-list pipeline and adds a compact summary for quick comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from umafactor.factor_list import FactorOcrOptions, recognize_factor_list_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug factor-name OCR.")
    parser.add_argument("image", type=Path, help="Stitched factor-list image.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "factor_name_ocr_debug")
    parser.add_argument("--expected", type=Path, default=None, help="Optional expected OCR CSV.")
    parser.add_argument("--ocr-engine", choices=["rapidocr", "paddle"], default="rapidocr")
    parser.add_argument("--rapidocr-ocr-version", choices=["PP-OCRv4", "PP-OCRv5"], default="PP-OCRv4")
    parser.add_argument("--rapidocr-lang-type", choices=["japan", "ch"], default="japan")
    parser.add_argument("--rapidocr-model-type", choices=["mobile", "server"], default="mobile")
    parser.add_argument("--rapidocr-rec-img-width", type=int, choices=[320, 480, 640], default=480)
    parser.add_argument("--ocr-batch-size", type=int, default=12)
    parser.add_argument(
        "--preprocess",
        nargs="*",
        default=["raw_upscaled", "gray_sharpen", "color_text_safe"],
        choices=["raw_upscaled", "gray_sharpen", "color_text_safe"],
    )
    args = parser.parse_args()

    output_dir = _resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = recognize_factor_list_image(
        _resolve_path(args.image),
        options=FactorOcrOptions(
            debug_dir=output_dir,
            enable_overlay=True,
            enable_stitch=False,
            ocr_mode=args.ocr_engine,
            ocr_roi_profiles=("body_name",),
            ocr_preprocess_modes=tuple(args.preprocess),
            ocr_execution_mode="batch",
            ocr_batch_size=args.ocr_batch_size,
            rapidocr_ocr_version=args.rapidocr_ocr_version,
            rapidocr_lang_type=args.rapidocr_lang_type,
            rapidocr_model_type=args.rapidocr_model_type,
            rapidocr_rec_img_shape=(3, 48, args.rapidocr_rec_img_width),
            rapidocr_rec_batch_num=args.ocr_batch_size,
        ),
    )
    elapsed = time.perf_counter() - started

    expected = _load_expected(_resolve_path(args.expected) if args.expected else None)
    accuracy = _evaluate(result.factors, expected)
    summary = {
        "image": str(_resolve_path(args.image)),
        "output": str(output_dir),
        "ocr_engine": args.ocr_engine,
        "factor_count": len(result.factors),
        "role_counts": _role_counts(result.factors),
        "needs_review_count": sum(1 for factor in result.factors if factor.needs_review),
        "fallback_recommended_count": sum(
            1 for factor in result.factors if factor.fallback_recommended
        ),
        "profile_counts": _count_by(
            factor.ocr_roi_profile or "" for factor in result.factors
        ),
        "preprocess_counts": _count_by(
            factor.ocr_preprocess_mode or "" for factor in result.factors
        ),
        "elapsed_sec": elapsed,
        "ms_per_card": elapsed * 1000.0 / max(1, len(result.factors)),
        **accuracy,
    }
    (output_dir / "factor_name_ocr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def _load_expected(path: Path | None) -> dict[tuple[str, int], str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (row.get("role", "parent"), int(row.get("order") or row.get("index") or 0)): row.get("name", "")
            for row in rows
            if row.get("name")
        }


def _evaluate(factors, expected: dict[tuple[str, int], str]) -> dict[str, object]:
    if not expected:
        return {"exact_canonical_accuracy": None, "mismatches": []}
    total = 0
    ok = 0
    mismatches = []
    for factor in factors:
        key = (factor.role, factor.order)
        expected_name = expected.get(key)
        if expected_name is None:
            continue
        total += 1
        actual = factor.normalized_name or factor.raw_name
        if actual == expected_name:
            ok += 1
        else:
            mismatches.append(
                {
                    "role": factor.role,
                    "order": factor.order,
                    "expected": expected_name,
                    "actual": actual,
                    "raw": factor.raw_name,
                    "score": factor.match_confidence,
                }
            )
    return {
        "exact_canonical_accuracy": ok / total if total else None,
        "evaluated_count": total,
        "mismatches": mismatches[:30],
    }


def _role_counts(factors) -> dict[str, int]:
    return _count_by(factor.role for factor in factors)


def _count_by(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
