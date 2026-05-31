"""Formal factor-list card detection entry point."""

from __future__ import annotations

import numpy as np

from .factor_list import FactorListDetection, detect_stitched_factor_list_card_body


def detect_factor_list_cards(
    image: np.ndarray,
    *,
    section_index: int = 0,
) -> FactorListDetection:
    """Detect factor-list cards using the production card-body detector."""

    return detect_stitched_factor_list_card_body(image, section_index=section_index)
