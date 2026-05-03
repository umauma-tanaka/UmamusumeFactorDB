"""Image loading and factor-box preparation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from ..detection import (
    BASE_WIDTH,
    CharaSection,
    FactorBox,
    detect_chara_sections,
    extract_factor_boxes,
    normalize_width,
)


@dataclass(frozen=True)
class PreparedFactorImage:
    img_orig: np.ndarray
    norm_img: np.ndarray
    scale: float
    sections: list[CharaSection]
    boxes: list[FactorBox]


def prepare_factor_image(
    image_path: str,
    *,
    debug_crops_dir: str | None = None,
    auto_debug: bool = True,
) -> PreparedFactorImage:
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")

    norm_img, scale = normalize_width(img_orig, BASE_WIDTH)
    sections = detect_chara_sections(norm_img)
    if len(sections) < 3:
        raise RuntimeError(
            f"ウマ娘セクションを 3 体分検出できませんでした（検出数={len(sections)}）"
        )
    boxes = extract_factor_boxes(norm_img, sections)

    if debug_crops_dir:
        dump_debug_crops(norm_img, sections, boxes, debug_crops_dir)
    elif auto_debug and len(boxes) < 9:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        auto_dir = os.path.join("tests", "fixtures", "debug_crops", stem)
        dump_debug_crops(norm_img, sections, boxes, auto_dir)

    return PreparedFactorImage(
        img_orig=img_orig,
        norm_img=norm_img,
        scale=scale,
        sections=sections,
        boxes=boxes,
    )


def dump_debug_crops(
    img: np.ndarray,
    sections: list[CharaSection],
    boxes: list[FactorBox],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    overlay = img.copy()
    for section in sections:
        x0, y0, x1, y1 = section.portrait_bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for box in boxes:
        x0, y0, x1, y1 = box.bbox
        color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}.get(
            box.color, (255, 255, 255)
        )
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1)
    cv2.imwrite(os.path.join(out_dir, "_overlay.png"), overlay)
    for box in boxes:
        base = f"uma{box.uma_index}_row{box.row_index:02d}_col{box.col_index}_{box.color}"
        cv2.imwrite(os.path.join(out_dir, f"{base}_text.png"), box.text_img)
        cv2.imwrite(os.path.join(out_dir, f"{base}_rank.png"), box.rank_img)
