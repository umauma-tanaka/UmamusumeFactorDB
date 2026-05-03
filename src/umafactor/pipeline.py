"""画像 → (Submission, ReviewQueue) への統合パイプライン。

ReviewQueue は低信頼度の因子をユーザにレビューしてもらうための候補リスト。
"""

from __future__ import annotations

from .config import load_unique_skill_to_character
from .recognition.context import build_recognition_context
from .recognition.factor_recognition import run_factor_recognition
from .recognition.image_crops import (
    crop_from_original as _crop_from_original,
    crop_rank_from_original as _crop_rank_from_original,
    display_crop_for_slot as _display_crop_for_slot,
    display_crop_from_original as _display_crop_from_original,
    extract_character_icon_bgr as _extract_character_icon_bgr,
)
from .recognition.image_preprocessing import (
    dump_debug_crops as _dump_debug_crops,
    prepare_factor_image,
)
from .results import apply_review_results, build_submission
from .review import ReviewQueue
from .schema import Submission


def analyze_image(
    image_path: str,
    submitter_id: str,
    debug_crops_dir: str | None = None,
    auto_debug: bool = True,
    skip_ocr: bool = False,
) -> tuple[Submission, ReviewQueue]:
    prepared = prepare_factor_image(
        image_path,
        debug_crops_dir=debug_crops_dir,
        auto_debug=auto_debug,
    )

    context = build_recognition_context(skip_ocr=skip_ocr)

    recognition = run_factor_recognition(
        prepared,
        context,
        load_unique_skill_to_character(),
    )
    umas = recognition.umas

    submission = build_submission(
        submitter_id=submitter_id,
        image_path=image_path,
        umas=umas,
    )
    return submission, recognition.review
