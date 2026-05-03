"""Star-rank resolution helpers for factor recognition."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .constants import PERTURBATIONS_RANK
from .image_crops import crop_rank_from_original, display_crop_from_original
from ..templates import match_green_star, match_star


class FactorBoxLike(Protocol):
    uma_index: int
    row_index: int
    col_index: int
    color: str
    bbox: tuple[int, int, int, int]
    rank_bbox: tuple[int, int, int, int] | None
    gold_star_count: int | None


class RankPredictionLike(Protocol):
    label: str
    confidence: float


class RankPredictorLike(Protocol):
    def predict_with_perturbation(
        self,
        img_hwc_bgr: np.ndarray,
        perturbations: list[tuple[int, int]],
    ) -> RankPredictionLike:
        ...


class UmaFactorsLike(Protocol):
    green_name: str
    green_star: int


def predict_factor_star(
    rank_pred: RankPredictorLike,
    img_orig: np.ndarray,
    box: FactorBoxLike,
    scale: float,
) -> int:
    if box.gold_star_count is not None and box.gold_star_count > 0:
        return box.gold_star_count

    rank_crop_orig = crop_rank_from_original(img_orig, box.bbox, scale, box.rank_bbox)
    rpred = rank_pred.predict_with_perturbation(rank_crop_orig, PERTURBATIONS_RANK)
    try:
        star = int(rpred.label)
    except ValueError:
        star = 0

    if (
        star == 0
        and box.row_index == 0
        and box.col_index in (0, 1)
        and rpred.confidence < 0.6
    ):
        return 1
    return star


def resolve_colored_star(
    img_orig: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    color: str,
    fallback_star: int,
) -> int:
    x0, y0, x1, y1 = bbox
    right_x0 = x0 + int((x1 - x0) * 0.5)
    star_crop = display_crop_from_original(
        img_orig, (right_x0, y0, x1, y1), scale, pad_y_norm=2
    )
    star_matches = match_star(star_crop, color)
    if star_matches and star_matches[0][1] >= 0.92:
        return star_matches[0][0]
    return fallback_star


def nearest_green_gold_star(box: FactorBoxLike, boxes: Sequence[FactorBoxLike]) -> int:
    nearest_star = 0
    best_dist = None
    for candidate in boxes:
        if candidate.uma_index != box.uma_index or candidate.color != "green":
            continue
        gold = candidate.gold_star_count or 0
        if gold <= 0:
            continue
        dist = abs(candidate.row_index - box.row_index)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            nearest_star = gold
    return nearest_star


def resolve_green_star(
    img_orig: np.ndarray,
    box: FactorBoxLike,
    boxes: Sequence[FactorBoxLike],
    scale: float,
    fallback_star: int,
) -> int:
    x0, y0, x1, y1 = box.bbox
    right_x0 = x0 + int((x1 - x0) * 0.5)
    star_crop = display_crop_from_original(
        img_orig, (right_x0, y0, x1, y1), scale, pad_y_norm=2
    )
    star_matches = match_green_star(star_crop)
    if star_matches and star_matches[0][1] >= 0.92:
        return star_matches[0][0]

    own_gold = box.gold_star_count or 0
    if own_gold > 0:
        return own_gold

    nearest_star = nearest_green_gold_star(box, boxes)
    if nearest_star > 0:
        return nearest_star

    if fallback_star > 0:
        return fallback_star
    return 1


def apply_missing_green_star_fallbacks(
    umas: Sequence[UmaFactorsLike],
    any_green_gold: dict[int, int],
) -> None:
    for uma_idx, uma in enumerate(umas):
        if not uma.green_name and uma.green_star == 0:
            gold = any_green_gold.get(uma_idx, 0)
            if gold > 0:
                uma.green_star = gold
