"""Per-factor-box recognition orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .assignment import apply_factor_result, build_review_item
from .candidate_generation import (
    FactorOCRLike,
    FactorPredictorLike,
    recognize_factor_candidates,
)
from .image_crops import display_crop_for_slot
from .slots import SlotFlags, classify_factor_slot, should_adopt_green_box
from .star_rank import RankPredictorLike, predict_factor_star
from ..review import ReviewItem
from ..schema import UmaFactors


class FactorBoxLike(Protocol):
    uma_index: int
    row_index: int
    col_index: int
    color: str
    bbox: tuple[int, int, int, int]
    rank_bbox: tuple[int, int, int, int] | None
    gold_star_count: int | None


@dataclass(frozen=True)
class RecognizedFactorBox:
    review_item: ReviewItem
    slot_flags: SlotFlags
    green_adoptable: bool


def recognize_factor_box(
    umas: Sequence[UmaFactors],
    white_counters: dict[int, int],
    factor_pred: FactorPredictorLike,
    rank_pred: RankPredictorLike,
    ocr: FactorOCRLike | None,
    img_orig: np.ndarray,
    norm_img: np.ndarray,
    box: FactorBoxLike,
    boxes: Sequence[FactorBoxLike],
    scale: float,
    green_name_set: set[str],
    best_green_box: dict[int, FactorBoxLike],
    best_green_score: dict[int, float],
) -> RecognizedFactorBox:
    x0, y0, x1, y1 = box.bbox
    text_crop_norm = norm_img[y0:y1, x0:x1]

    slot_flags = classify_factor_slot(box)
    is_blue_slot = slot_flags.is_blue
    is_red_slot = slot_flags.is_red
    is_green_slot = slot_flags.is_green

    uma = umas[box.uma_index]
    green_adoptable = should_adopt_green_box(
        box,
        boxes,
        is_green_slot=is_green_slot,
        current_green_name=uma.green_name,
        best_green_box=best_green_box.get(box.uma_index),
        best_green_score=best_green_score.get(box.uma_index, 0.0),
    )

    display_crop = display_crop_for_slot(
        img_orig,
        norm_img.shape,
        box.bbox,
        scale,
        is_blue_slot=is_blue_slot,
        is_red_slot=is_red_slot,
    )
    candidate_recognition = recognize_factor_candidates(
        factor_pred,
        ocr,
        img_orig,
        text_crop_norm,
        display_crop,
        box.bbox,
        scale,
        is_blue_slot=is_blue_slot,
        is_red_slot=is_red_slot,
        green_adoptable=green_adoptable,
        green_name_set=green_name_set,
    )

    star = predict_factor_star(rank_pred, img_orig, box, scale)
    assignment = apply_factor_result(
        uma,
        white_counters,
        img_orig,
        box,
        boxes,
        scale,
        candidate_recognition.top_name,
        star,
        is_blue_slot=is_blue_slot,
        is_red_slot=is_red_slot,
        green_adoptable=green_adoptable,
    )
    review_item = build_review_item(
        box,
        assignment,
        display_crop,
        candidate_recognition.candidates,
        candidate_recognition.sources,
        candidate_recognition.ocr_raw,
        candidate_recognition.top_name,
        star,
    )
    return RecognizedFactorBox(review_item, slot_flags, green_adoptable)
