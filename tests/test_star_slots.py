from __future__ import annotations

import cv2
import numpy as np
import pytest

from umafactor.detection.star_slots import detect_star_slots_from_card


@pytest.mark.parametrize(
    "background",
    [
        (220, 180, 70),
        (180, 90, 240),
        (80, 190, 110),
        (230, 230, 230),
    ],
)
def test_star_slots_count_filled_slots_independent_of_card_color(background) -> None:
    image = _card(background=background, filled_slots=2, icon=True)

    debug = detect_star_slots_from_card(image, (0, 0, image.shape[1], image.shape[0]))

    assert debug.star_count == 2
    assert len(debug.slot_bboxes) == 3
    assert debug.yellow_ratios[0] > debug.yellow_ratios[2]
    assert debug.icon_exclusion_bbox[2] <= debug.star_roi_bbox[0]


def test_star_slots_ignore_yellow_left_icon_when_no_star_is_filled() -> None:
    image = _card(background=(80, 190, 110), filled_slots=0, icon=True)

    debug = detect_star_slots_from_card(image, (0, 0, image.shape[1], image.shape[0]))

    assert debug.star_count == 0
    assert all(ratio < 0.02 for ratio in debug.yellow_ratios)


def test_star_slots_ignore_yellow_green_icon_near_old_roi_boundary() -> None:
    image = _card(background=(80, 190, 110), filled_slots=3, icon=True)
    cv2.circle(image, (86, 24), 10, (0, 220, 255), -1)

    debug = detect_star_slots_from_card(image, (0, 0, image.shape[1], image.shape[0]))

    assert debug.star_count == 3
    assert debug.icon_exclusion_bbox[2] < debug.star_roi_bbox[0]


def test_star_slots_use_right_column_fixed_positions() -> None:
    image = np.full((90, 440, 3), 230, dtype=np.uint8)
    card_bbox = (220, 5, 420, 85)
    cv2.rectangle(image, (card_bbox[0], card_bbox[1]), (card_bbox[2] - 1, card_bbox[3] - 1), (220, 220, 220), 1)
    for center in [(329, 61), (350, 61)]:
        cv2.circle(image, center, 9, (0, 220, 255), -1)

    debug = detect_star_slots_from_card(image, card_bbox)

    assert debug.star_count == 2
    assert debug.yellow_ratios[0] > 0.05
    assert debug.yellow_ratios[1] > 0.05
    assert debug.yellow_ratios[2] < 0.05


def test_star_slots_clamp_to_three() -> None:
    image = _card(background=(230, 230, 230), filled_slots=3, icon=True)
    cv2.circle(image, (188, 56), 9, (0, 220, 255), -1)

    debug = detect_star_slots_from_card(image, (0, 0, image.shape[1], image.shape[0]))

    assert debug.star_count == 3


def _card(
    *,
    background: tuple[int, int, int],
    filled_slots: int,
    icon: bool,
) -> np.ndarray:
    image = np.full((80, 200, 3), background, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (199, 79), (220, 220, 220), 1)
    if icon:
        cv2.circle(image, (28, 24), 13, (0, 220, 255), -1)

    centers = [(129, 56), (150, 56), (171, 56)]
    for index, center in enumerate(centers):
        color = (0, 220, 255) if index < filled_slots else (225, 225, 225)
        cv2.circle(image, center, 9, color, -1)
    return image
