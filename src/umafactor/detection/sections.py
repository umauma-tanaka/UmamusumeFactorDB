"""Character section detection."""

from __future__ import annotations

import cv2
import numpy as np

from .constants import (
    LOW_SAT_THRESHOLD,
    MIN_GRID_RUN_LEN,
    PARENT_HEADER_HEIGHT,
    SELF_HEADER_HEIGHT,
)
from .stars import _cluster_stars_into_rows, _detect_empty_stars, _detect_golden_stars
from .types import CharaSection

def _row_saturation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    band = img[:, 0 : int(w * 0.15)]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].mean(axis=1)


def _find_low_sat_runs(row_sat: np.ndarray, threshold: float, min_len: int) -> list[tuple[int, int]]:
    mask = row_sat < threshold
    runs: list[tuple[int, int]] = []
    start = 0
    in_run = False
    for y, flag in enumerate(mask):
        if flag and not in_run:
            start = y
            in_run = True
        elif not flag and in_run:
            if y - start >= min_len:
                runs.append((start, y))
            in_run = False
    if in_run and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def detect_chara_sections(img: np.ndarray) -> list[CharaSection]:
    h = img.shape[0]
    row_sat = _row_saturation(img)
    runs = _find_low_sat_runs(row_sat, LOW_SAT_THRESHOLD, MIN_GRID_RUN_LEN)
    runs_by_len = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:3]
    grids = sorted(runs_by_len, key=lambda r: r[0])

    if len(grids) != 3:
        # 低彩度 run 検出が 3 セクション分取れなかった場合は、★検出クラスタの y
        # 分布から 3 セクションを推定する fallback を試す。
        fallback = _detect_chara_sections_by_stars(img)
        if fallback is not None:
            return fallback
        raise RuntimeError(f"因子グリッドを 3 領域検出できませんでした（{len(grids)} 件）")

    sections: list[CharaSection] = []
    for i, (g_start, g_end) in enumerate(grids):
        header_h = SELF_HEADER_HEIGHT if i == 0 else PARENT_HEADER_HEIGHT
        portrait_y0 = max(0, g_start - header_h)
        portrait_y1 = max(portrait_y0 + 10, g_start - 10)
        w = img.shape[1]
        sections.append(
            CharaSection(
                uma_index=i,
                factor_y_start=g_start,
                factor_y_end=g_end,
                portrait_bbox=(int(w * 0.01), portrait_y0, int(w * 0.18), portrait_y1),
            )
        )
    return sections


def _detect_chara_sections_by_stars(img: np.ndarray) -> list[CharaSection] | None:
    """★検出クラスタの y 分布から 3 セクションを推定する fallback。

    行ピッチ (~45px) を基準に ★行を密集群としてグループ化し、
    偽陽性（ヘッダー装飾等）を除外したうえで 3 セクションを取り出す。
    - gap > SECTION_SPLIT_GAP (= 65px) で群を分割
    - 群の中で行数 >= MIN_ROWS_PER_SECTION (= 5) かつ 長さ >= MIN_SECTION_SPAN (= 100px)
      のものだけを因子欄セクション候補とみなす
    - 3 つ以上取れたら y 順に先頭 3 つを採用、取れなければ None（legacy を素通り）
    """
    SECTION_SPLIT_GAP = 65
    MIN_ROWS_PER_SECTION = 5
    MIN_SECTION_SPAN = 100

    gold = _detect_golden_stars(img)
    empty = _detect_empty_stars(img)
    classified = _cluster_stars_into_rows(gold, empty, img.shape[1])
    if len(classified) < MIN_ROWS_PER_SECTION * 3:
        return None
    ys = sorted(r[0] for r in classified)

    groups: list[list[int]] = [[ys[0]]]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] > SECTION_SPLIT_GAP:
            groups.append([ys[i]])
        else:
            groups[-1].append(ys[i])

    big_groups = [
        g for g in groups
        if len(g) >= MIN_ROWS_PER_SECTION and (g[-1] - g[0]) >= MIN_SECTION_SPAN
    ]
    if len(big_groups) < 3:
        return None

    sections_y = sorted(big_groups, key=lambda g: g[0])[:3]
    w = img.shape[1]
    sections: list[CharaSection] = []
    for i, grp in enumerate(sections_y):
        y_s, y_e = grp[0], grp[-1]
        header_h = SELF_HEADER_HEIGHT if i == 0 else PARENT_HEADER_HEIGHT
        portrait_y0 = max(0, y_s - header_h)
        portrait_y1 = max(portrait_y0 + 10, y_s - 10)
        sections.append(
            CharaSection(
                uma_index=i,
                factor_y_start=y_s,
                factor_y_end=y_e,
                portrait_bbox=(int(w * 0.01), portrait_y0, int(w * 0.18), portrait_y1),
            )
        )
    return sections
