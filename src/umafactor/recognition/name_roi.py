"""Geometric factor-name ROI extraction from card-body boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class NameRoiOptions:
    """Relative bounds for the factor-name text line inside a card body."""

    x1_ratio: float = 0.13
    x2_margin_ratio: float = 0.03
    y1_ratio: float = 0.05
    y2_ratio: float = 0.68
    min_height_ratio: float = 0.38


@dataclass(frozen=True)
class NameRoiDebug:
    body_bbox: BBox
    text_bbox: BBox
    icon_exclusion_bbox: BBox


def compute_name_roi_from_body(
    image_shape: tuple[int, ...],
    body_bbox: BBox,
    *,
    options: NameRoiOptions | None = None,
) -> NameRoiDebug:
    """Return a text ROI based only on the card body geometry.

    RapidOCR is used as recognition-only in the current factor-list flow, so
    the OCR crop must be deterministic.  This function deliberately avoids
    text detection, Hough circles, and star edge feedback.
    """

    opts = options or NameRoiOptions()
    x0, y0, x1, y1 = _clip_nonempty_bbox(body_bbox, image_shape)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    text_x0 = x0 + int(round(width * opts.x1_ratio))
    text_x1 = x1 - int(round(width * opts.x2_margin_ratio))
    text_y0 = y0 + int(round(height * opts.y1_ratio))
    text_y1 = y0 + int(round(height * opts.y2_ratio))

    min_height = max(4, int(round(height * opts.min_height_ratio)))
    if text_y1 - text_y0 < min_height:
        text_y1 = min(y1 - 1, text_y0 + min_height)

    text_bbox = _clip_nonempty_bbox((text_x0, text_y0, text_x1, text_y1), image_shape)
    icon_exclusion_bbox = _clip_nonempty_bbox((x0, y0, text_bbox[0], y1), image_shape)
    return NameRoiDebug(
        body_bbox=(x0, y0, x1, y1),
        text_bbox=text_bbox,
        icon_exclusion_bbox=icon_exclusion_bbox,
    )


def crop_name_roi(
    image: np.ndarray,
    body_bbox: BBox,
    *,
    options: NameRoiOptions | None = None,
) -> tuple[np.ndarray, NameRoiDebug]:
    debug = compute_name_roi_from_body(image.shape, body_bbox, options=options)
    x0, y0, x1, y1 = debug.text_bbox
    return image[y0:y1, x0:x1], debug


def _clip_nonempty_bbox(bbox: BBox, image_shape: tuple[int, ...]) -> BBox:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), max(0, width - 1)))
    y0 = max(0, min(int(y0), max(0, height - 1)))
    x1 = max(x0 + 1, min(int(x1), width))
    y1 = max(y0 + 1, min(int(y1), height))
    return x0, y0, x1, y1
