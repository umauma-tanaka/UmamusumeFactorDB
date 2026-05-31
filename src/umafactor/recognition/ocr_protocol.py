"""Common OCR interfaces used by the factor-list pipeline."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

FactorListCardDetectorMode = Literal["card-body"]
FactorListOcrCropTarget = Literal["name", "card"]
FactorListOcrExecutionMode = Literal["sequential", "canvas", "batch", "role_sheet"]
FactorListOcrRoiProfile = Literal["body_name", "card_upper_band", "text_band_with_margin", "tight_text_roi"]
FactorListOcrPreprocessMode = Literal[
    "raw_upscaled",
    "gray_sharpen",
    "color_text_safe",
    "light_text_safe",
    "dark_text",
    "white_text",
]

DEFAULT_OCR_MIN_WIDTH = 640
DEFAULT_OCR_MIN_HEIGHT = 160
DEFAULT_OCR_MAX_UPSCALE = 4.0
DEFAULT_OCR_SHARPEN_STRENGTH = 0.45
DEFAULT_OCR_CONTRAST_CLIP_LIMIT = 1.6
DEFAULT_OCR_SHEET_MAX_SIDE = 3600


class FactorListOCRLike(Protocol):
    def recognize(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_blue(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_red(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_with_parts(self, img_bgr: np.ndarray) -> tuple[str, list[str]]:
        ...

    def recognize_many(self, images_bgr: list[np.ndarray]) -> list[str]:
        ...
