"""Detection package for factor image structure extraction."""

from .boxes import detect_factor_color, extract_factor_boxes
from .sections import detect_chara_sections
from .types import (
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
    "detect_chara_sections",
    "detect_factor_color",
    "extract_factor_boxes",
    "normalize_width",
]
