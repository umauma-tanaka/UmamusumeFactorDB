"""Crop helpers used by the factor recognition pipeline."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class CharaSectionLike(Protocol):
    portrait_bbox: tuple[int, int, int, int]


def extract_character_icon_bgr(img: np.ndarray, section: CharaSectionLike) -> np.ndarray:
    x0, y0, x1, y1 = section.portrait_bbox
    h_sec = y1 - y0
    icon_size = min(h_sec, x1 - x0)
    cy = y0 + icon_size // 2
    cx = x0 + icon_size // 2
    half = icon_size // 2
    crop = img[max(0, cy - half): cy + half, max(0, cx - half): cx + half]
    if crop.size == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return cv2.resize(crop, (32, 32), interpolation=cv2.INTER_LINEAR)


def crop_from_original(
    img_orig: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    dy: int = 0,
    dx: int = 0,
) -> np.ndarray:
    inv = 1.0 / scale if scale != 0 else 1.0
    x0, y0, x1, y1 = bbox
    ox0 = int(round(x0 * inv)) + dx
    oy0 = int(round(y0 * inv)) + dy
    ox1 = int(round(x1 * inv)) + dx
    oy1 = int(round(y1 * inv)) + dy
    ox0 = max(0, min(ox0, img_orig.shape[1]))
    ox1 = max(ox0 + 1, min(ox1, img_orig.shape[1]))
    oy0 = max(0, min(oy0, img_orig.shape[0]))
    oy1 = max(oy0 + 1, min(oy1, img_orig.shape[0]))
    return img_orig[oy0:oy1, ox0:ox1]


def display_crop_from_original(
    img_orig: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    pad_y_norm: int = 2,
) -> np.ndarray:
    """Return a wider crop for review UI and OCR."""
    inv = 1.0 / scale if scale != 0 else 1.0
    x0, y0, x1, y1 = bbox
    pad_left_norm = 32
    pad_right_norm = 8
    ox0 = int(round((x0 - pad_left_norm) * inv))
    oy0 = int(round((y0 - pad_y_norm) * inv))
    ox1 = int(round((x1 + pad_right_norm) * inv))
    oy1 = int(round((y1 + pad_y_norm) * inv))
    ox0 = max(0, ox0)
    oy0 = max(0, oy0)
    ox1 = min(img_orig.shape[1], ox1)
    oy1 = min(img_orig.shape[0], oy1)
    return img_orig[oy0:oy1, ox0:ox1]


def display_crop_for_slot(
    img_orig: np.ndarray,
    norm_img_shape: tuple[int, ...],
    bbox: tuple[int, int, int, int],
    scale: float,
    *,
    is_blue_slot: bool,
    is_red_slot: bool,
) -> np.ndarray:
    if is_blue_slot:
        return display_crop_from_original(img_orig, bbox, scale, pad_y_norm=8)

    if is_red_slot:
        x0, y0, x1, y1 = bbox
        img_h = norm_img_shape[0]
        red_disp_bbox = (x0, y0, x1, min(img_h, y1 + 14))
        return display_crop_from_original(img_orig, red_disp_bbox, scale, pad_y_norm=2)

    return display_crop_from_original(img_orig, bbox, scale)


def crop_rank_from_original(
    img_orig: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    rank_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Crop the star-rank region from the original-resolution image."""
    inv = 1.0 / scale if scale != 0 else 1.0

    if rank_bbox is not None:
        rank_x0_norm, rank_y0_norm, rank_x1_norm, rank_y1_norm = rank_bbox
        rank_y0_norm -= 2
        rank_y1_norm += 2
    else:
        x0, y0, x1, y1 = bbox
        box_w_norm = x1 - x0
        rel_x0 = 0.6786
        rel_x1 = 1.0
        rank_x0_norm = x0 + int(round(box_w_norm * rel_x0))
        rank_x1_norm = x0 + int(round(box_w_norm * rel_x1))
        rank_y0_norm = y0 + 11
        rank_y1_norm = y0 + 27

    rx0 = int(round(rank_x0_norm * inv))
    ry0 = int(round(rank_y0_norm * inv))
    rx1 = int(round(rank_x1_norm * inv))
    ry1 = int(round(rank_y1_norm * inv))
    rx0 = max(0, rx0)
    ry0 = max(0, ry0)
    rx1 = min(img_orig.shape[1], rx1)
    ry1 = min(img_orig.shape[0], ry1)
    return img_orig[ry0:ry1, rx0:rx1]
