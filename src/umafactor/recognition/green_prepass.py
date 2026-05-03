"""Green-slot pre-pass helpers for factor recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .image_crops import display_crop_from_original
from .slots import is_green_candidate_box
from ..templates import match_green_name


class FactorBoxLike(Protocol):
    uma_index: int
    row_index: int
    col_index: int
    color: str
    bbox: tuple[int, int, int, int]
    gold_star_count: int | None


class GreenOCRLike(Protocol):
    def recognize_with_parts(self, img: np.ndarray) -> tuple[str, list[str]]:
        ...

    def match_to_green_factor_multi(
        self, text: str, fragments: list[str], top_k: int = 1
    ) -> list[tuple[str, float]]:
        ...


@dataclass(frozen=True)
class GreenPrepassResult:
    best_green_box: dict[int, FactorBoxLike]
    best_green_score: dict[int, float]
    best_green_gold: dict[int, int]
    any_green_gold: dict[int, int]


def compute_green_prepass(
    boxes: Sequence[FactorBoxLike],
    img_orig: np.ndarray,
    scale: float,
    ocr: GreenOCRLike | None,
) -> GreenPrepassResult:
    best_green_box: dict[int, FactorBoxLike] = {}
    best_green_score: dict[int, float] = {}
    best_green_gold: dict[int, int] = {}
    any_green_gold: dict[int, int] = {}

    for box in boxes:
        if not is_green_candidate_box(box):
            continue

        gold = box.gold_star_count or 0
        if gold > any_green_gold.get(box.uma_index, 0):
            any_green_gold[box.uma_index] = gold

        if box.col_index != 0:
            continue

        display_crop = display_crop_from_original(img_orig, box.bbox, scale)
        if ocr is None:
            ocr_candidates = []
        else:
            raw, fragments = ocr.recognize_with_parts(display_crop)
            ocr_candidates = ocr.match_to_green_factor_multi(raw, fragments, top_k=1)
        top_conf = ocr_candidates[0][1] if ocr_candidates else 0.0

        x0, y0, x1, y1 = box.bbox
        name_x1 = x0 + int((x1 - x0) * 0.85)
        name_crop = display_crop_from_original(
            img_orig, (x0, y0, name_x1, y1), scale, pad_y_norm=2
        )
        green_name_matches = match_green_name(name_crop)
        green_name_conf = green_name_matches[0][1] if green_name_matches else 0.0
        combined_conf = max(top_conf, green_name_conf)

        uma_index = box.uma_index
        if combined_conf > best_green_score.get(uma_index, 0.0):
            best_green_score[uma_index] = combined_conf
            best_green_box[uma_index] = box

        if gold > best_green_gold.get(uma_index, 0):
            best_green_gold[uma_index] = gold

    return GreenPrepassResult(
        best_green_box=best_green_box,
        best_green_score=best_green_score,
        best_green_gold=best_green_gold,
        any_green_gold=any_green_gold,
    )
