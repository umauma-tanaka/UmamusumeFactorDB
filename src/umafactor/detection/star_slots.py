"""Card-local star slot detection for stitched factor-list tiles."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import (
    EMPTY_STAR_HSV_HI,
    EMPTY_STAR_HSV_LO,
    GOLD_STAR_HSV_HI,
    GOLD_STAR_HSV_LO,
)


@dataclass(frozen=True)
class StarSlotConfig:
    icon_exclusion_right_rel: float = 0.45
    roi_x0_rel: float = 0.45
    roi_x1_rel: float = 1.00
    roi_y0_rel: float = 0.40
    roi_y1_rel: float = 0.98
    yellow_ratio_threshold: float = 0.055
    left_column_star_center_x_rel: float = 0.75
    right_column_star_center_x_rel: float = 0.65
    star_center_y_rel: float = 0.70
    star_search_y0_rel: float = 0.45
    star_search_y1_rel: float = 1.70
    star_pitch_card_rel: float = 0.105
    fixed_slot_width_card_rel: float = 0.13
    fixed_slot_height_card_rel: float = 0.70
    slot_width_roi_rel: float = 0.28
    slot_height_roi_rel: float = 0.90
    min_component_area_card_rel: float = 0.006
    max_component_area_card_rel: float = 0.18
    min_component_width_card_rel: float = 0.025
    max_component_width_card_rel: float = 0.20
    min_component_height_card_rel: float = 0.18
    max_component_height_card_rel: float = 0.95


@dataclass(frozen=True)
class StarSlotDebug:
    card_bbox: tuple[int, int, int, int]
    icon_exclusion_bbox: tuple[int, int, int, int]
    star_roi_bbox: tuple[int, int, int, int]
    slot_bboxes: tuple[tuple[int, int, int, int], ...]
    yellow_ratios: tuple[float, ...]
    star_count: int


DEFAULT_STAR_SLOT_CONFIG = StarSlotConfig()


def detect_star_slots_from_card(
    image: np.ndarray,
    card_bbox: tuple[int, int, int, int],
    *,
    config: StarSlotConfig = DEFAULT_STAR_SLOT_CONFIG,
) -> StarSlotDebug:
    """Detect filled star count from three card-local star slots.

    The left icon area is excluded before any star decision.  The final count is
    based on yellow fill ratio in three estimated slot boxes, not on the number
    of yellow connected components in the whole card.
    """

    card = _clip_bbox(card_bbox, image.shape)
    x0, y0, x1, y1 = card
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    icon = _relative_bbox(card, 0.0, 0.0, config.icon_exclusion_right_rel, 1.0, image.shape)
    slot_bboxes = _fixed_slot_bboxes(card, image, config=config)
    roi = _union_bbox(slot_bboxes)

    if _bbox_is_empty(roi):
        return StarSlotDebug(card, icon, roi, tuple(), tuple(), 0)

    yellow_mask, _slot_mask = _star_masks(image, roi)
    ratios = tuple(_yellow_ratio(yellow_mask, roi, slot_bbox) for slot_bbox in slot_bboxes)
    star_count = min(3, max(0, sum(1 for ratio in ratios if ratio > config.yellow_ratio_threshold)))
    return StarSlotDebug(
        card_bbox=card,
        icon_exclusion_bbox=icon,
        star_roi_bbox=roi,
        slot_bboxes=tuple(slot_bboxes),
        yellow_ratios=ratios,
        star_count=star_count,
    )


def _fixed_slot_bboxes(
    card_bbox: tuple[int, int, int, int],
    image: np.ndarray,
    *,
    config: StarSlotConfig,
) -> list[tuple[int, int, int, int]]:
    image_shape = image.shape
    x0, y0, x1, y1 = card_bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    is_right_column = (x0 + x1) / 2.0 > image_shape[1] / 2.0
    center_x_rel = (
        config.right_column_star_center_x_rel
        if is_right_column
        else config.left_column_star_center_x_rel
    )
    group_center_x = x0 + width * center_x_rel
    pitch = max(2.0, width * config.star_pitch_card_rel)
    slot_width = max(3, int(round(width * config.fixed_slot_width_card_rel)))
    slot_height = max(3, int(round(height * config.fixed_slot_height_card_rel)))
    centers_x = [group_center_x + pitch * offset for offset in (-1, 0, 1)]
    center_y = _estimate_fixed_star_center_y(
        image=image,
        card_bbox=card_bbox,
        centers_x=centers_x,
        slot_width=slot_width,
        slot_height=slot_height,
        config=config,
    )

    slots: list[tuple[int, int, int, int]] = []
    for center_x in centers_x:
        slot = (
            int(round(center_x - slot_width / 2.0)),
            int(round(center_y - slot_height / 2.0)),
            int(round(center_x + slot_width / 2.0)),
            int(round(center_y + slot_height / 2.0)),
        )
        slots.append(_clip_bbox(slot, image_shape))
    return slots


def _union_bbox(
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    if not bboxes:
        return 0, 0, 0, 0
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _estimate_fixed_star_center_y(
    *,
    image: np.ndarray,
    card_bbox: tuple[int, int, int, int],
    centers_x: list[float],
    slot_width: int,
    slot_height: int,
    config: StarSlotConfig,
) -> float:
    x0, y0, x1, y1 = card_bbox
    height = max(1, y1 - y0)
    default_y = y0 + height * config.star_center_y_rel
    search_y0 = int(round(y0 + height * config.star_search_y0_rel))
    search_y1 = int(round(y0 + height * config.star_search_y1_rel))
    search_y0 = max(0, min(image.shape[0] - 1, search_y0))
    search_y1 = max(search_y0 + 1, min(image.shape[0], search_y1))

    x_min = max(0, int(round(min(centers_x) - slot_width / 2.0)))
    x_max = min(image.shape[1], int(round(max(centers_x) + slot_width / 2.0)))
    y_min = max(0, int(round(search_y0 - slot_height / 2.0)))
    y_max = min(image.shape[0], int(round(search_y1 + slot_height / 2.0)))
    if x_max <= x_min or y_max <= y_min:
        return default_y

    crop = image[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array(GOLD_STAR_HSV_LO, dtype=np.uint8),
        np.array(GOLD_STAR_HSV_HI, dtype=np.uint8),
    )

    best_y = default_y
    best_score = 0
    for center_y in range(search_y0, search_y1):
        score = 0
        for center_x in centers_x:
            sx0 = max(0, int(round(center_x - slot_width / 2.0)) - x_min)
            sx1 = min(yellow.shape[1], int(round(center_x + slot_width / 2.0)) - x_min)
            sy0 = max(0, int(round(center_y - slot_height / 2.0)) - y_min)
            sy1 = min(yellow.shape[0], int(round(center_y + slot_height / 2.0)) - y_min)
            if sx1 > sx0 and sy1 > sy0:
                score += cv2.countNonZero(yellow[sy0:sy1, sx0:sx1])
        if score > best_score:
            best_score = score
            best_y = float(center_y)
    return best_y


def _relative_bbox(
    card_bbox: tuple[int, int, int, int],
    x0_rel: float,
    y0_rel: float,
    x1_rel: float,
    y1_rel: float,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = card_bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    return _clip_bbox(
        (
            x0 + int(round(width * x0_rel)),
            y0 + int(round(height * y0_rel)),
            x0 + int(round(width * x1_rel)),
            y0 + int(round(height * y1_rel)),
        ),
        image_shape,
    )


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(image_width, int(x0)))
    x1 = max(0, min(image_width, int(x1)))
    y0 = max(0, min(image_height, int(y0)))
    y1 = max(0, min(image_height, int(y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _bbox_is_empty(bbox: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = bbox
    return x1 <= x0 or y1 <= y0


def _star_masks(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = roi
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array(GOLD_STAR_HSV_LO, dtype=np.uint8),
        np.array(GOLD_STAR_HSV_HI, dtype=np.uint8),
    )
    empty = cv2.inRange(
        hsv,
        np.array(EMPTY_STAR_HSV_LO, dtype=np.uint8),
        np.array(EMPTY_STAR_HSV_HI, dtype=np.uint8),
    )
    slot_mask = cv2.bitwise_or(yellow, empty)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    slot_mask = cv2.morphologyEx(slot_mask, cv2.MORPH_CLOSE, kernel)
    return yellow, slot_mask


def _slot_component_candidates(
    slot_mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    card_width: int,
    card_height: int,
    config: StarSlotConfig,
) -> list[tuple[int, int, int, int]]:
    card_area = max(1, card_width * card_height)
    min_area = max(3, int(round(card_area * config.min_component_area_card_rel)))
    max_area = max(min_area, int(round(card_area * config.max_component_area_card_rel)))
    min_width = max(2, int(round(card_width * config.min_component_width_card_rel)))
    max_width = max(min_width, int(round(card_width * config.max_component_width_card_rel)))
    min_height = max(2, int(round(card_height * config.min_component_height_card_rel)))
    max_height = max(min_height, int(round(card_height * config.max_component_height_card_rel)))

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(slot_mask, connectivity=8)
    roi_x0, roi_y0, _roi_x1, _roi_y1 = roi
    candidates: list[tuple[int, int, int, int]] = []
    for index in range(1, n):
        x, y, width, height, area = stats[index]
        if not (
            min_area <= area <= max_area
            and min_width <= width <= max_width
            and min_height <= height <= max_height
        ):
            continue
        candidates.append(
            (
                int(roi_x0 + x),
                int(roi_y0 + y),
                int(roi_x0 + x + width),
                int(roi_y0 + y + height),
            )
        )

    if len(candidates) > 3:
        candidates = sorted(candidates, key=_bbox_area, reverse=True)[:3]
    return sorted(candidates, key=lambda bbox: (bbox[0] + bbox[2]) / 2.0)


def _estimate_slot_bboxes(
    filled_candidates: list[tuple[int, int, int, int]],
    slot_candidates: list[tuple[int, int, int, int]],
    roi: tuple[int, int, int, int],
    *,
    card_width: int,
    card_height: int,
    config: StarSlotConfig,
) -> list[tuple[int, int, int, int]]:
    roi_x0, roi_y0, roi_x1, roi_y1 = roi
    roi_width = max(1, roi_x1 - roi_x0)
    roi_height = max(1, roi_y1 - roi_y0)
    source_candidates: list[tuple[int, int, int, int]] = []
    if filled_candidates:
        centers = _infer_slot_centers_from_filled(filled_candidates, roi, card_height)
        source_candidates = filled_candidates
    elif len(slot_candidates) >= 3:
        centers = [
            ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            for bbox in slot_candidates[:3]
        ]
        source_candidates = slot_candidates[:3]
    else:
        centers = [
            (roi_x0 + roi_width * (index + 0.5) / 3.0, roi_y0 + roi_height * 0.5)
            for index in range(3)
        ]

    if source_candidates:
        source_widths = [max(1, bbox[2] - bbox[0]) for bbox in source_candidates]
        source_heights = [max(1, bbox[3] - bbox[1]) for bbox in source_candidates]
        slot_width = max(3, int(round(float(np.median(source_widths)) * 1.25)))
        slot_height = max(3, int(round(float(np.median(source_heights)) * 1.20)))
    else:
        slot_width = max(
            3,
            int(round(roi_width * config.slot_width_roi_rel)),
            int(round(card_width * 0.10)),
        )
        slot_height = max(
            3,
            int(round(roi_height * config.slot_height_roi_rel)),
            int(round(card_height * 0.45)),
        )
    slots: list[tuple[int, int, int, int]] = []
    for center_x, center_y in centers[:3]:
        slot = (
            int(round(center_x - slot_width / 2.0)),
            int(round(center_y - slot_height / 2.0)),
            int(round(center_x + slot_width / 2.0)),
            int(round(center_y + slot_height / 2.0)),
        )
        slots.append(_clip_bbox_to_bounds(slot, roi))
    return sorted(slots, key=lambda bbox: (bbox[0] + bbox[2]) / 2.0)


def _infer_slot_centers_from_filled(
    filled_candidates: list[tuple[int, int, int, int]],
    roi: tuple[int, int, int, int],
    card_height: int,
) -> list[tuple[float, float]]:
    candidates = sorted(filled_candidates[:3], key=lambda bbox: (bbox[0] + bbox[2]) / 2.0)
    centers = [((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0) for bbox in candidates]
    widths = [max(1, bbox[2] - bbox[0]) for bbox in candidates]
    roi_x0, roi_y0, roi_x1, roi_y1 = roi
    roi_height = max(1, roi_y1 - roi_y0)
    if len(centers) >= 2:
        pitches = [centers[index + 1][0] - centers[index][0] for index in range(len(centers) - 1)]
        pitch = float(np.median([value for value in pitches if value > 1.0] or pitches))
    else:
        pitch = max(float(np.median(widths) * 1.25), float(card_height * 0.66))

    pitch = max(2.0, pitch)
    y_center = float(np.median([center[1] for center in centers])) if centers else roi_y0 + roi_height * 0.5
    inferred = list(centers)
    while len(inferred) < 3:
        next_x = inferred[-1][0] + pitch
        if next_x > roi_x1 and inferred[0][0] - pitch >= roi_x0:
            inferred.insert(0, (inferred[0][0] - pitch, y_center))
        else:
            inferred.append((next_x, y_center))
    return inferred[:3]


def _clip_bbox_to_bounds(
    bbox: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    bounds_x0, bounds_y0, bounds_x1, bounds_y1 = bounds
    x0, y0, x1, y1 = bbox
    return (
        max(bounds_x0, min(bounds_x1, x0)),
        max(bounds_y0, min(bounds_y1, y0)),
        max(bounds_x0, min(bounds_x1, x1)),
        max(bounds_y0, min(bounds_y1, y1)),
    )


def _yellow_ratio(
    yellow_mask: np.ndarray,
    roi: tuple[int, int, int, int],
    slot_bbox: tuple[int, int, int, int],
) -> float:
    roi_x0, roi_y0, _roi_x1, _roi_y1 = roi
    x0, y0, x1, y1 = slot_bbox
    sx0 = max(0, x0 - roi_x0)
    sx1 = min(yellow_mask.shape[1], x1 - roi_x0)
    sy0 = max(0, y0 - roi_y0)
    sy1 = min(yellow_mask.shape[0], y1 - roi_y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return 0.0
    slot = yellow_mask[sy0:sy1, sx0:sx1]
    return float(cv2.countNonZero(slot)) / float(slot.size)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)
