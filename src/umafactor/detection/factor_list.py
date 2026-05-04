"""Detection helpers for stitched vertical factor lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .boxes import detect_factor_color
from .constants import (
    BASE_WIDTH,
    PARENT_ROW0_LOOKBACK,
    STAR_Y_IN_TILE,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from .sections import detect_chara_sections
from .stars import (
    _cluster_stars_into_rows,
    _detect_empty_stars,
    _detect_golden_stars,
    _estimate_tile_right_edges,
)
from .types import FactorColor, normalize_width

FactorListRole = Literal["parent", "ancestor1", "ancestor2"]


@dataclass(frozen=True)
class FactorListTile:
    order: int
    section_index: int
    role: FactorListRole
    row_index: int
    col_index: int
    color: FactorColor
    star: int
    bbox: tuple[int, int, int, int]
    bbox_norm: tuple[int, int, int, int]
    raw_name: str = ""
    final_name: str = ""
    known_skill: bool = False
    needs_review: bool = False
    confidence: float = 1.0


@dataclass(frozen=True)
class FactorListDetection:
    image_width: int
    image_height: int
    scale: float
    section_index: int
    role: FactorListRole
    tiles: list[FactorListTile]


@dataclass(frozen=True)
class _ColumnRowCandidate:
    row_index: int
    col_index: int
    y_center: int
    star: int
    bbox_norm: tuple[int, int, int, int]
    color: FactorColor


def detect_stitched_factor_list(
    image: np.ndarray,
    *,
    section_index: int = 0,
) -> FactorListDetection:
    """Detect factor tiles from a stitched list image.

    This detector is intentionally list-oriented.  It treats left and right
    columns independently because green tiles can place their stars at a
    slightly different vertical position from the paired right-column tile.
    """

    if image is None or image.size == 0:
        raise ValueError("image is empty")

    norm, scale = normalize_width(image, BASE_WIDTH)
    sections = detect_chara_sections(norm)
    if section_index < 0 or section_index >= len(sections):
        raise IndexError(f"section_index out of range: {section_index}")

    gold = _detect_golden_stars(norm)
    empty = _detect_empty_stars(norm)
    rows = _cluster_stars_into_rows(gold, empty, norm.shape[1])
    x_left1, x_right1 = _estimate_tile_right_edges(rows)
    if x_left1 is None or x_right1 is None:
        raise RuntimeError("failed to estimate factor tile columns")

    section = sections[section_index]
    y_min = max(0, section.factor_y_start - PARENT_ROW0_LOOKBACK - 30)
    y_max = min(norm.shape[0], section.factor_y_end + TILE_HEIGHT + 30)
    left_rows: list[_ColumnRowCandidate] = []
    right_rows: list[_ColumnRowCandidate] = []
    for y_center, left_gold, right_gold, _left_empty, _right_empty in rows:
        if not (y_min <= y_center <= y_max):
            continue
        if left_gold:
            left_rows.append(
                _build_column_candidate(
                    norm,
                    row_index=len(left_rows),
                    col_index=0,
                    y_center=y_center,
                    x_right=x_left1,
                    gold_count=len(left_gold),
                )
            )
        if right_gold:
            right_rows.append(
                _build_column_candidate(
                    norm,
                    row_index=len(right_rows),
                    col_index=1,
                    y_center=y_center,
                    x_right=x_right1,
                    gold_count=len(right_gold),
                )
            )

    role = _role_for_section(section_index)
    tiles: list[FactorListTile] = []
    order = 0
    inv = 1.0 / scale if scale else 1.0
    for row_index in range(max(len(left_rows), len(right_rows))):
        for candidate_rows in (left_rows, right_rows):
            if row_index >= len(candidate_rows):
                continue
            candidate = candidate_rows[row_index]
            tiles.append(
                FactorListTile(
                    order=order,
                    section_index=section_index,
                    role=role,
                    row_index=row_index,
                    col_index=candidate.col_index,
                    color=candidate.color,
                    star=candidate.star,
                    bbox=_scale_bbox(candidate.bbox_norm, inv, image.shape),
                    bbox_norm=candidate.bbox_norm,
                )
            )
            order += 1

    return FactorListDetection(
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        scale=scale,
        section_index=section_index,
        role=role,
        tiles=tiles,
    )


def _build_column_candidate(
    norm_img: np.ndarray,
    *,
    row_index: int,
    col_index: int,
    y_center: int,
    x_right: int,
    gold_count: int,
) -> _ColumnRowCandidate:
    x0 = max(0, x_right - TILE_WIDTH)
    x1 = min(norm_img.shape[1], x_right)
    y0 = max(0, y_center - STAR_Y_IN_TILE)
    y1 = min(norm_img.shape[0], y0 + TILE_HEIGHT)
    bbox_norm = (x0, y0, x1, y1)
    tile = norm_img[y0:y1, x0:x1]
    color = detect_factor_color(tile) if tile.size else "white"
    return _ColumnRowCandidate(
        row_index=row_index,
        col_index=col_index,
        y_center=y_center,
        star=min(3, max(0, gold_count)),
        bbox_norm=bbox_norm,
        color=color,
    )


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    scale: float,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    height, width = image_shape[:2]
    return (
        max(0, min(width, int(round(x0 * scale)))),
        max(0, min(height, int(round(y0 * scale)))),
        max(0, min(width, int(round(x1 * scale)))),
        max(0, min(height, int(round(y1 * scale)))),
    )


def _role_for_section(section_index: int) -> FactorListRole:
    if section_index == 0:
        return "parent"
    if section_index == 1:
        return "ancestor1"
    return "ancestor2"
