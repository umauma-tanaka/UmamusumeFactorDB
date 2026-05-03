"""Character recognition and post-processing helpers."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .image_crops import extract_character_icon_bgr
from ..schema import UmaFactors


class CharaSectionLike(Protocol):
    uma_index: int
    portrait_bbox: tuple[int, int, int, int]


class CharacterPredictionLike(Protocol):
    label: str


class CharacterPredictorLike(Protocol):
    def predict(self, img_hwc_bgr: np.ndarray) -> CharacterPredictionLike:
        ...


def recognize_characters(
    umas: Sequence[UmaFactors],
    sections: Sequence[CharaSectionLike],
    norm_img: np.ndarray,
    char_pred: CharacterPredictorLike,
) -> None:
    for section in sections:
        icon = extract_character_icon_bgr(norm_img, section)
        pred = char_pred.predict(icon)
        umas[section.uma_index].character = pred.label


def apply_unique_skill_character_overrides(
    umas: Sequence[UmaFactors],
    unique_skill_to_character: dict[str, str],
) -> None:
    if not unique_skill_to_character:
        return

    for uma in umas:
        if not uma.green_name:
            continue
        key_candidates = [uma.green_name, uma.green_name.strip()]
        for key in key_candidates:
            if key in unique_skill_to_character:
                uma.character = unique_skill_to_character[key]
                break
