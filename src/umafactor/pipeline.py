"""Compatibility facade for image analysis pipeline helpers."""

from __future__ import annotations

from .app.analyzer import UmaFactorAnalyzer
from .app.result_builder import apply_review_results, build_submission
from .recognition.image_crops import (
    crop_from_original as _crop_from_original,
    crop_rank_from_original as _crop_rank_from_original,
    display_crop_for_slot as _display_crop_for_slot,
    display_crop_from_original as _display_crop_from_original,
    extract_character_icon_bgr as _extract_character_icon_bgr,
)
from .recognition.image_preprocessing import dump_debug_crops as _dump_debug_crops
from .review import ReviewQueue
from .schema import Submission


def analyze_image(
    image_path: str,
    submitter_id: str,
    debug_crops_dir: str | None = None,
    auto_debug: bool = True,
    skip_ocr: bool = False,
) -> tuple[Submission, ReviewQueue]:
    return UmaFactorAnalyzer().analyze_image(
        image_path,
        submitter_id,
        debug_crops_dir=debug_crops_dir,
        auto_debug=auto_debug,
        skip_ocr=skip_ocr,
    )


__all__ = [
    "UmaFactorAnalyzer",
    "_crop_from_original",
    "_crop_rank_from_original",
    "_display_crop_for_slot",
    "_display_crop_from_original",
    "_dump_debug_crops",
    "_extract_character_icon_bgr",
    "analyze_image",
    "apply_review_results",
    "build_submission",
]
