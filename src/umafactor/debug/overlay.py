"""Debug overlay rendering for factor-list OCR results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..factor_list.types import RecognizedFactor


_ROLE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "parent": (0, 80, 255, 255),
    "ancestor1": (255, 0, 0, 255),
    "ancestor2": (0, 180, 0, 255),
}


def write_factor_ocr_overlay(
    path: str | Path,
    image_bgr: np.ndarray,
    factors: Iterable[RecognizedFactor],
) -> None:
    """Write one overlay image containing all OCR results."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = make_factor_ocr_overlay(image_bgr, factors)
    cv2.imwrite(str(target), overlay)


def make_factor_ocr_overlay(
    image_bgr: np.ndarray,
    factors: Iterable[RecognizedFactor],
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image is empty")

    rgb = cv2.cvtColor(_ensure_bgr_u8(image_bgr), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = _load_font(16)
    small_font = _load_font(13)

    for factor in factors:
        draw_bbox = factor.ocr_bbox or factor.bbox
        if draw_bbox is None:
            continue
        color = _ROLE_COLORS.get(factor.role, (120, 120, 120, 255))
        if factor.initial_card_bbox is not None:
            draw.rectangle(factor.initial_card_bbox, outline=(150, 90, 255, 180), width=1)
        if factor.expected_icon_center is not None:
            _draw_cross(draw, factor.expected_icon_center, (0, 220, 0, 230), size=6)
        for rejected_bbox in factor.rejected_icon_bboxes:
            draw.ellipse(rejected_bbox, outline=(150, 150, 150, 130), width=1)
        if factor.icon_bbox is not None:
            draw.ellipse(factor.icon_bbox, outline=(0, 255, 255, 230), width=2)
        if factor.bbox is not None:
            draw.rectangle(factor.bbox, outline=color, width=2)
        if factor.icon_exclusion_bbox is not None:
            draw.rectangle(factor.icon_exclusion_bbox, outline=(0, 210, 255, 180), width=1)
        if factor.star_roi_bbox is not None:
            draw.rectangle(factor.star_roi_bbox, outline=(255, 0, 220, 180), width=1)
        for slot_bbox in factor.star_slot_bboxes:
            draw.rectangle(slot_bbox, outline=(255, 150, 0, 210), width=1)
        if factor.ocr_bbox is not None:
            draw.rectangle(factor.ocr_bbox, outline=(255, 210, 0, 255), width=3)
        _draw_factor_label(draw, factor, draw_bbox, color, font, small_font)

    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _circle_bbox(center: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    cx, cy = center
    return cx - radius, cy - radius, cx + radius, cy + radius


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    color: tuple[int, int, int, int],
    *,
    size: int,
) -> None:
    cx, cy = center
    draw.line((cx - size, cy, cx + size, cy), fill=color, width=2)
    draw.line((cx, cy - size, cx, cy + size), fill=color, width=2)


def _draw_factor_label(
    draw: ImageDraw.ImageDraw,
    factor: RecognizedFactor,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    font,
    small_font,
) -> None:
    x0, y0, _x1, _y1 = bbox
    raw = factor.raw_name.strip() or "(blank)"
    normalized = factor.normalized_name or ""
    title = f"{factor.role} #{factor.order} s{factor.stars}"
    detail = f"raw:{raw}"
    if normalized and normalized != raw:
        detail += f" norm:{normalized}"
    if factor.card_bbox_fallback:
        detail += " fallback"

    title_bbox = draw.textbbox((0, 0), title, font=font)
    detail_bbox = draw.textbbox((0, 0), detail, font=small_font)
    width = max(title_bbox[2], detail_bbox[2]) + 8
    height = (title_bbox[3] - title_bbox[1]) + (detail_bbox[3] - detail_bbox[1]) + 8
    label_y0 = max(0, y0 - height - 2)
    label_y1 = label_y0 + height
    label_x0 = max(0, x0)
    label_x1 = label_x0 + width

    draw.rectangle((label_x0, label_y0, label_x1, label_y1), fill=(255, 255, 255, 210))
    draw.rectangle((label_x0, label_y0, label_x1, label_y1), outline=color, width=1)
    draw.text((label_x0 + 4, label_y0 + 2), title, font=font, fill=color)
    draw.text((label_x0 + 4, label_y0 + 4 + title_bbox[3] - title_bbox[1]), detail, font=small_font, fill=(30, 30, 30, 255))


def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _load_font(size: int):
    font_paths = [
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()
