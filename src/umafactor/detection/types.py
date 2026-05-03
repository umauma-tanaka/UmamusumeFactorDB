"""Detection data structures and image normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .constants import BASE_HEIGHT, BASE_WIDTH

FactorColor = Literal["blue", "red", "green", "white"]


@dataclass
class FactorBox:
    uma_index: int
    row_index: int
    col_index: int
    color: FactorColor
    text_img: np.ndarray
    rank_img: np.ndarray
    bbox: tuple[int, int, int, int]
    # ★検出が効いた新経路で、実★クラスタの正規化座標 (x0, y0, x1, y1) を保持する。
    # None の場合（legacy 経路）は pipeline 側で layout.rank_x0_in_box_rel から計算。
    rank_bbox: tuple[int, int, int, int] | None = None
    # ★検出駆動で数え上げた金★の個数。rank モデル推論の代替（より高精度な実測値）。
    # None の場合（legacy 経路など）は pipeline 側で rank モデル推論にフォールバック。
    gold_star_count: int | None = None
    # CNN 分類で空★と判定された個数。★全空の緑因子（gold=0 かつ empty>=2）を
    # skill に流さず緑スロットに残す判定に使う。
    empty_star_count: int | None = None


@dataclass
class CharaSection:
    uma_index: int
    factor_y_start: int
    factor_y_end: int
    portrait_bbox: tuple[int, int, int, int]


def normalize_width(img: np.ndarray, target_width: int = BASE_WIDTH) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w == target_width:
        return img, 1.0
    scale = target_width / w
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA), scale
