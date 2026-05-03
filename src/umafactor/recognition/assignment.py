"""Apply recognized factors to output models and review items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, cast

import numpy as np

from .constants import BLUE_FACTOR_TYPES, RED_FACTOR_TYPES, UMA_ROLES
from .star_rank import resolve_colored_star, resolve_green_star
from ..review import ReviewItem, SlotKind
from ..schema import FactorEntry, UmaFactors


class FactorBoxLike(Protocol):
    uma_index: int
    row_index: int
    col_index: int
    color: str
    bbox: tuple[int, int, int, int]
    rank_bbox: tuple[int, int, int, int] | None
    gold_star_count: int | None


@dataclass(frozen=True)
class AssignmentResult:
    slot_kind: SlotKind
    white_index: int


def apply_factor_result(
    uma: UmaFactors,
    white_counters: dict[int, int],
    img_orig: np.ndarray,
    box: FactorBoxLike,
    boxes: Sequence[FactorBoxLike],
    scale: float,
    top_name: str,
    star: int,
    *,
    is_blue_slot: bool,
    is_red_slot: bool,
    green_adoptable: bool,
) -> AssignmentResult:
    white_idx = 0
    if is_blue_slot and top_name in BLUE_FACTOR_TYPES and not uma.blue_type:
        uma.blue_type = top_name
        uma.blue_star = resolve_colored_star(img_orig, box.bbox, scale, "blue", star)
        return AssignmentResult(slot_kind="blue", white_index=white_idx)

    if is_red_slot and top_name in RED_FACTOR_TYPES and not uma.red_type:
        uma.red_type = top_name
        uma.red_star = resolve_colored_star(img_orig, box.bbox, scale, "red", star)
        return AssignmentResult(slot_kind="red", white_index=white_idx)

    if green_adoptable:
        uma.green_name = top_name
        uma.green_star = resolve_green_star(img_orig, box, boxes, scale, star)
        return AssignmentResult(slot_kind="green", white_index=white_idx)

    uma.skills.append(FactorEntry(color=box.color, name=top_name, star=star))
    white_idx = white_counters[box.uma_index]
    white_counters[box.uma_index] += 1
    return AssignmentResult(slot_kind="white", white_index=white_idx)


def build_review_item(
    box: FactorBoxLike,
    assignment: AssignmentResult,
    display_crop: np.ndarray,
    candidates: list[tuple[str, float]],
    candidate_sources: dict[str, str],
    ocr_raw: str,
    current_name: str,
    current_star: int,
) -> ReviewItem:
    return ReviewItem(
        uma_index=box.uma_index,
        uma_role=UMA_ROLES[box.uma_index],
        slot=cast(SlotKind, assignment.slot_kind),
        white_index=assignment.white_index,
        image=display_crop.copy(),
        candidates=candidates,
        candidate_sources=candidate_sources,
        ocr_raw=ocr_raw,
        current_name=current_name,
        current_star=current_star,
    )
