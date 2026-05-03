"""Factor row detection helpers."""

from __future__ import annotations

import numpy as np

from ..config import FactorLayout
from .constants import (
    MIN_ROW_HEIGHT,
    PARENT_ROW0_LOOKBACK,
    ROW_CONTENT_STD_THRESHOLD,
    ROW_MERGE_GAP,
    SELF_ROW0_LOOKBACK,
    SPLIT_THRESHOLD,
    TARGET_ROW_PITCH,
)
from .types import CharaSection

def _detect_factor_rows(
    img: np.ndarray, section: CharaSection, layout: FactorLayout
) -> list[tuple[int, int]]:
    """セクションの因子行を (y_top, y_bottom) のリストで返す。

    uma0（本人）のみ、低彩度 run より上にある Row 0 を拾うため y 下限を広げる。
    """
    w = img.shape[1]
    left_x0 = int(round(w * layout.left_x0))
    left_x1 = int(round(w * layout.left_x1))

    lookback = SELF_ROW0_LOOKBACK if section.uma_index == 0 else PARENT_ROW0_LOOKBACK
    y_start = max(0, section.factor_y_start - lookback)
    y_end = min(img.shape[0], section.factor_y_end + 5)

    stds = np.array([
        img[y, left_x0:left_x1].std() for y in range(y_start, y_end)
    ])
    mask = stds > ROW_CONTENT_STD_THRESHOLD

    # content runs を抽出
    raw_runs: list[tuple[int, int]] = []
    s = 0
    in_r = False
    for i, f in enumerate(mask):
        if f and not in_r:
            s = i
            in_r = True
        elif not f and in_r:
            if i - s >= 3:
                raw_runs.append((y_start + s, y_start + i))
            in_r = False
    if in_r and len(mask) - s >= 3:
        raw_runs.append((y_start + s, y_start + len(mask)))

    # gap <= ROW_MERGE_GAP の隣接 run をマージ
    merged: list[tuple[int, int]] = []
    for a, b in raw_runs:
        if merged and a - merged[-1][1] <= ROW_MERGE_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    # h > SPLIT_THRESHOLD のブロックは複数行が融合した結果なので、等分割
    split_rows: list[tuple[int, int]] = []
    for a, b in merged:
        h = b - a
        if h > SPLIT_THRESHOLD:
            n = max(2, round(h / TARGET_ROW_PITCH))
            piece = h // n
            for i in range(n - 1):
                split_rows.append((a + i * piece, a + (i + 1) * piece))
            split_rows.append((a + (n - 1) * piece, b))
        else:
            split_rows.append((a, b))

    # 高さ下限フィルタ（バナー h=22 を除外）
    rows = [(a, b) for a, b in split_rows if (b - a) >= MIN_ROW_HEIGHT]
    return rows

detect_factor_rows = _detect_factor_rows

def _assign_row_to_section(
    row_y: int, sections: list[CharaSection]
) -> int | None:
    """因子行 y がどの CharaSection に属するかを判定。

    セクションの y 範囲（lookback 込み）に入る最初のセクション番号を返す。
    どこにも入らなければ None。
    """
    for sec in sections:
        lookback = SELF_ROW0_LOOKBACK if sec.uma_index == 0 else PARENT_ROW0_LOOKBACK
        y_start = max(0, sec.factor_y_start - lookback)
        y_end = sec.factor_y_end + 10
        if y_start <= row_y <= y_end:
            return sec.uma_index
    return None

assign_row_to_section = _assign_row_to_section
