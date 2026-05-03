"""Compatibility wrapper for factor image detection helpers."""

from __future__ import annotations

from .detection.boxes import (
    _build_boxes_for_row,
    _crop_rank_region,
    _extract_factor_boxes_legacy,
    _is_blank_row,
    _strip_leading_empty_rows,
    detect_factor_color,
    extract_factor_boxes,
)
from .detection.constants import *  # noqa: F403
from .detection.rows import (
    _assign_row_to_section,
    _detect_factor_rows,
    assign_row_to_section,
    detect_factor_rows,
)
from .detection.sections import (
    _detect_chara_sections_by_stars,
    _find_low_sat_runs,
    _row_saturation,
    detect_chara_sections,
)
from .detection.stars import (
    _cluster_stars_into_rows,
    _detect_empty_stars,
    _detect_golden_stars,
    _detect_green_tile_stars_relaxed,
    _detect_stars_by_hsv,
    _detect_stars_by_hsv_closed,
    _estimate_tile_right_edges,
)
from .detection.types import (
    BASE_HEIGHT,
    BASE_WIDTH,
    CharaSection,
    FactorBox,
    FactorColor,
    normalize_width,
)

__all__ = [
    "BASE_HEIGHT",
    "BASE_WIDTH",
    "CharaSection",
    "FactorBox",
    "FactorColor",
    "assign_row_to_section",
    "detect_chara_sections",
    "detect_factor_color",
    "detect_factor_rows",
    "extract_factor_boxes",
    "normalize_width",
]
