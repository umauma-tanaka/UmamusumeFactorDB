"""Star detection helpers for factor boxes."""

from __future__ import annotations

import cv2
import numpy as np

from .constants import (
    EMPTY_STAR_HSV_HI,
    EMPTY_STAR_HSV_LO,
    EMPTY_STAR_MIN_AREA,
    GOLD_STAR_HSV_HI,
    GOLD_STAR_HSV_LO,
    GREEN_RELAX_EMPTY_HSV_HI,
    GREEN_RELAX_EMPTY_HSV_LO,
    GREEN_RELAX_GOLD_HSV_HI,
    GREEN_RELAX_GOLD_HSV_LO,
    MIN_STARS_PER_COLUMN,
    STAR_MAX_AREA,
    STAR_MAX_H,
    STAR_MAX_W,
    STAR_MIN_AREA,
    STAR_MIN_H,
    STAR_MIN_W,
    STAR_PITCH_MAX,
    STAR_PITCH_MIN,
    STAR_ROW_Y_TOL,
    TILE_RIGHT_PADDING,
    TILE_RIGHT_PERCENTILE,
)

def _detect_stars_by_hsv(
    img: np.ndarray,
    lo: tuple[int, int, int],
    hi: tuple[int, int, int],
) -> list[tuple[int, int, int, int]]:
    """指定 HSV 範囲のマスクから★形サイズの連結成分を (x, y, w, h) で返す汎用関数。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stars: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (
            STAR_MIN_W <= w <= STAR_MAX_W
            and STAR_MIN_H <= h <= STAR_MAX_H
            and STAR_MIN_AREA <= area <= STAR_MAX_AREA
        ):
            stars.append((int(x), int(y), int(w), int(h)))
    return stars


# 緑タイル内の★再検出用・緩和 HSV 閾値。
# 背景が薄緑で★の S/V が弱まる画像（umamusume_*）向け。
# 画像全体に適用すると偽陽性が増えるため、緑タイル内限定で使う。
# さらに緑背景の薄金色★は V=80〜130 付近まで落ちるため、V 下限は大幅に緩める。


def _detect_stars_by_hsv_closed(
    img: np.ndarray,
    lo: tuple[int, int, int],
    hi: tuple[int, int, int],
    close_kernel: int = 3,
) -> list[tuple[int, int, int, int]]:
    """_detect_stars_by_hsv の閉じ処理版。★形の穴や隙間を morphological closing で埋める。

    緑タイル内の★マークは塗りがグラデーションになって HSV マスクが穴あきに
    なりやすく、★1 個が複数連結成分に割れる。CLOSING で隙間を埋めてから
    連結成分化する。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stars: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (
            STAR_MIN_W <= w <= STAR_MAX_W
            and STAR_MIN_H <= h <= STAR_MAX_H
            and STAR_MIN_AREA <= area <= STAR_MAX_AREA
        ):
            stars.append((int(x), int(y), int(w), int(h)))
    return stars


def _detect_green_tile_stars_relaxed(tile_bgr: np.ndarray) -> tuple[int, int]:
    """緑タイル内限定の緩和★検出。gold_count / empty_count を返す。

    画像全体の HSV 金★検出が背景色との関係で失敗する画像のための補助経路。
    タイル右側 60% 領域（＝ 呼び出し側で既にクロップ済み）に限定することで
    左端の黄色●アイコンを含めず、偽陽性を抑える。

    判定ロジック:
      1. 緩和 gold/empty HSV マスクを MORPH_CLOSE で穴埋めしてから★形連結成分
         検出（★マークのグラデーションで mask が割れるのを防ぐ）
      2. 両マスクに同じ位置で拾われた★は「金★」扱い
      3. 等間隔ピッチ検証で偽陽性を排除
      4. HSV マスクの結果をそのまま gold/empty カウントに使う（CNN は緑背景上
         の★に弱いので通さない）
    """
    gold = _detect_stars_by_hsv_closed(
        tile_bgr, GREEN_RELAX_GOLD_HSV_LO, GREEN_RELAX_GOLD_HSV_HI,
    )
    empty = _detect_stars_by_hsv_closed(
        tile_bgr, GREEN_RELAX_EMPTY_HSV_LO, GREEN_RELAX_EMPTY_HSV_HI,
    )
    # 金・空の両方で拾われたスロットは 金 扱い（S が高めで両マスクにヒット）
    gold_centers = {(x + w // 2, y + h // 2) for x, y, w, h in gold}
    dedup_empty = [
        (x, y, w, h) for x, y, w, h in empty
        if (x + w // 2, y + h // 2) not in gold_centers
    ]
    # 等間隔ピッチ検証で偽陽性抑制
    all_stars = sorted(
        [("g", *s) for s in gold] + [("e", *s) for s in dedup_empty],
        key=lambda t: t[1],
    )
    if len(all_stars) >= 2:
        kept = [all_stars[0]]
        for prev, cur in zip(all_stars, all_stars[1:]):
            pitch = (cur[1] + cur[3] // 2) - (prev[1] + prev[3] // 2)
            if STAR_PITCH_MIN <= pitch <= STAR_PITCH_MAX:
                kept.append(cur)
        all_stars = kept
    gold_count = sum(1 for s in all_stars if s[0] == "g")
    empty_count = sum(1 for s in all_stars if s[0] == "e")
    return min(gold_count, 3), min(empty_count, 3)


def _detect_golden_stars(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """画像全体から金★（点灯★）候補を (x, y, w, h) で返す。"""
    return _detect_stars_by_hsv(img, GOLD_STAR_HSV_LO, GOLD_STAR_HSV_HI)


def _detect_empty_stars(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """画像全体から空★（未点灯★）候補を (x, y, w, h) で返す。

    空★は薄ピンク縁＋白塗りで描画され、金★と彩度で明確に分離できる（S<90）。
    空★位置を併用することで ★3 スロット全体の位置を確定でき、rank 領域切り出しや
    「金★取りこぼし検知」が可能になる。

    最小 area を EMPTY_STAR_MIN_AREA(=80) まで上げ、細切れのテキスト装飾等を除外する。
    """
    raw = _detect_stars_by_hsv(img, EMPTY_STAR_HSV_LO, EMPTY_STAR_HSV_HI)
    return [(x, y, w, h) for (x, y, w, h) in raw if w * h >= EMPTY_STAR_MIN_AREA]


def _cluster_stars_into_rows(
    gold_stars: list[tuple[int, int, int, int]],
    empty_stars: list[tuple[int, int, int, int]] | None = None,
    img_width: int = 540,
) -> list[tuple[
    int,
    list[tuple[int, int, int, int]],  # left gold
    list[tuple[int, int, int, int]],  # right gold
    list[tuple[int, int, int, int]],  # left empty
    list[tuple[int, int, int, int]],  # right empty
]]:
    """金★主導で y 近接クラスタリングし、各行の近傍 y に存在する空★を取得する。

    空★は偽陽性が混入しやすい（タイル内の薄ピンク/白要素）ため、y クラスタリングの
    基準にはせず「既に確定した金★行の近傍」に位置する空★だけを★3スロット候補とする。
    これで空★のノイズに行構造が崩れず、かつ「★1個+空2個」の行でも空★2個を
    正しく拾える。

    Returns:
        各行について (y_center, left_gold, right_gold, left_empty, right_empty)。
    """
    empty_stars = empty_stars or []
    if not gold_stars:
        return []
    mid_x = img_width // 2

    gold_sorted = sorted(gold_stars, key=lambda s: s[1] + s[3] // 2)
    rows: list[list[tuple[int, int, int, int]]] = []
    for s in gold_sorted:
        cy = s[1] + s[3] // 2
        if rows:
            ref_cy = int(np.mean([r[1] + r[3] // 2 for r in rows[-1]]))
            if abs(cy - ref_cy) <= STAR_ROW_Y_TOL:
                rows[-1].append(s)
                continue
        rows.append([s])

    classified: list[tuple[
        int, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]],
        list[tuple[int, int, int, int]], list[tuple[int, int, int, int]],
    ]] = []
    for row in rows:
        lg = [s for s in row if (s[0] + s[2] // 2) < mid_x]
        rg = [s for s in row if (s[0] + s[2] // 2) >= mid_x]
        y_center = int(np.mean([s[1] + s[3] // 2 for s in row]))
        # この y 近傍にある空★を取得（STAR_ROW_Y_TOL 以内）
        le: list[tuple[int, int, int, int]] = []
        re_: list[tuple[int, int, int, int]] = []
        for s in empty_stars:
            if abs((s[1] + s[3] // 2) - y_center) <= STAR_ROW_Y_TOL:
                cx = s[0] + s[2] // 2
                (le if cx < mid_x else re_).append(s)
        if not (lg or rg or le or re_):
            continue
        classified.append((y_center, lg, rg, le, re_))
    return classified


def _estimate_tile_right_edges(
    classified_rows: list[tuple[
        int, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]],
        list[tuple[int, int, int, int]], list[tuple[int, int, int, int]],
    ]],
) -> tuple[int | None, int | None]:
    """左列/右列それぞれの「タイル右端 x」を金★の x_right 中央値から推定。

    空★は偽陽性（タイル内の薄ピンク UI 要素）が混入しやすいため推定には使わない。
    ★3個並んだ行の金★最右端 + TILE_RIGHT_PADDING(36) でタイル右端に相当する。
    """
    left_maxes: list[int] = []
    right_maxes: list[int] = []
    for _y, lg, rg, _le, _re in classified_rows:
        if lg:
            left_maxes.append(max(s[0] + s[2] for s in lg))
        if rg:
            right_maxes.append(max(s[0] + s[2] for s in rg))
    # ★は行ごとに 1-3 個と可変なので、中央値では★1個行に引っ張られる。
    # ★3個行の★右端を基準にするため上位 percentile を使う。
    # 更に ★右端の右側にあるタイル余白（TILE_RIGHT_PADDING）を足してタイル右端とする。
    x_L1 = (
        int(np.percentile(left_maxes, TILE_RIGHT_PERCENTILE)) + TILE_RIGHT_PADDING
        if len(left_maxes) >= MIN_STARS_PER_COLUMN
        else None
    )
    x_R1 = (
        int(np.percentile(right_maxes, TILE_RIGHT_PERCENTILE)) + TILE_RIGHT_PADDING
        if len(right_maxes) >= MIN_STARS_PER_COLUMN
        else None
    )
    return x_L1, x_R1
