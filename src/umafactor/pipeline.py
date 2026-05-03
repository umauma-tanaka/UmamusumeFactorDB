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

    import os
    submission = Submission(
        submitter_id=submitter_id,
        image_filename=os.path.basename(image_path),
        main=umas[0],
        parent1=umas[1],
        parent2=umas[2],
    )
    return submission, recognition.review


def apply_review_results(submission: Submission, review: ReviewQueue) -> None:
    """ユーザレビュー後の ReviewItem の reviewed_name / reviewed_star を Submission に反映。"""
    umas = [submission.main, submission.parent1, submission.parent2]
    for item in review.items:
        if item.reviewed_name is None:
            continue
        uma = umas[item.uma_index]
        star = item.reviewed_star if item.reviewed_star is not None else item.current_star
        if item.slot == "blue":
            uma.blue_type = item.reviewed_name
            uma.blue_star = star
        elif item.slot == "red":
            uma.red_type = item.reviewed_name
            uma.red_star = star
        elif item.slot == "green":
            uma.green_name = item.reviewed_name
            uma.green_star = star
        elif item.slot == "white":
            if 0 <= item.white_index < len(uma.skills):
                uma.skills[item.white_index].name = item.reviewed_name
                uma.skills[item.white_index].star = star
