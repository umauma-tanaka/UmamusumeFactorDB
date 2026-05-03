"""Factor box extraction."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import FACTOR_COLOR_HSV_RANGES, FactorLayout
from .constants import (
    ENABLE_GOLD_LAYOUT_FILTER,
    MIN_DETECTED_ROWS,
    STAR_PITCH_MAX,
    STAR_PITCH_MIN,
    STAR_Y_IN_TILE,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from .rows import _assign_row_to_section, _detect_factor_rows
from .stars import (
    _cluster_stars_into_rows,
    _detect_empty_stars,
    _detect_golden_stars,
    _detect_green_tile_stars_relaxed,
    _estimate_tile_right_edges,
)
from .types import CharaSection, FactorBox, FactorColor

def detect_factor_color(box_bgr: np.ndarray) -> FactorColor:
    """因子ボックスの左端色チップから青/赤/緑/白を判定する。

    chip 幅 15% と 22% の両方でスコアを計算し、色ごとに最大値を採用する。
    緑アイコンの位置/サイズが tile により異なり、
    - 小さい緑アイコン（1558/1814 の緑因子 tile）は 22% 幅で拾える
    - 大きい緑アイコン（2331 のレース名スキル等）は 15% 幅で濃く出る（広げると薄まる）
    両方で max を取ることで両パターンを救う（Exp 13）。
    """
    h, w = box_bgr.shape[:2]

    def score_chip(pct: float) -> dict[str, float]:
        cw = max(4, int(w * pct))
        chip = box_bgr[:, 0:cw]
        hsv = cv2.cvtColor(chip, cv2.COLOR_BGR2HSV)

        def ratio(lo, hi):
            mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            return float(mask.mean()) / 255.0

        return {
            "blue": ratio(*FACTOR_COLOR_HSV_RANGES["blue"]),
            "green": ratio(*FACTOR_COLOR_HSV_RANGES["green"]),
            "red": ratio(*FACTOR_COLOR_HSV_RANGES["red"]),
        }

    s15 = score_chip(0.15)
    s22 = score_chip(0.22)
    scores = {k: max(s15[k], s22[k]) for k in s15}
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    # 閾値 0.18: 1558/1814 の緑 tile が chip 22% で green=0.191 を返すため、
    # 0.20 から 0.18 に下げて拾えるようにする。row 0 は位置絶対化で color を
    # 使わないため影響せず、row >= 1 で従来 "white" → 該当色に変わる行は
    # skill → blue/red/green に流れるが、is_*_slot は位置と color の両方を
    # 見るので誤昇格は起きにくい。
    if scores[best] > 0.18:
        return best  # type: ignore[return-value]
    return "white"

def extract_factor_boxes(
    img: np.ndarray,
    sections: list[CharaSection],
    layout: FactorLayout | None = None,
) -> list[FactorBox]:
    """因子ボックスを ★検出駆動で抽出（新経路）。失敗時は legacy にフォールバック。

    新経路は UmamusumeReceiptMaker の合成画像や解像度の異なるスクショでも
    layout 比率に依存せず動作する。旧経路は legacy 関数として保存している。
    """
    layout = layout or FactorLayout()
    gold = _detect_golden_stars(img)
    empty = _detect_empty_stars(img)
    classified_rows = _cluster_stars_into_rows(gold, empty, img.shape[1])

    if len(classified_rows) < MIN_DETECTED_ROWS:
        return _extract_factor_boxes_legacy(img, sections, layout)

    x_L1, x_R1 = _estimate_tile_right_edges(classified_rows)
    if x_L1 is None or x_R1 is None:
        return _extract_factor_boxes_legacy(img, sections, layout)

    x_L0 = max(0, x_L1 - TILE_WIDTH)
    x_R0 = max(0, x_R1 - TILE_WIDTH)

    # row_index はセクション内で 0 から数える
    per_section_row_idx: dict[int, int] = {}
    boxes: list[FactorBox] = []

    for y_center, lg, rg, le, re_ in classified_rows:
        uma_idx = _assign_row_to_section(y_center, sections)
        if uma_idx is None:
            continue
        row_idx = per_section_row_idx.get(uma_idx, 0)
        per_section_row_idx[uma_idx] = row_idx + 1
        boxes.extend(
            _build_boxes_for_row(
                img, uma_idx, row_idx, y_center,
                lg, rg, le, re_,
                x_L0, x_L1, x_R0, x_R1,
            )
        )
    if len(boxes) < MIN_DETECTED_ROWS:
        return _extract_factor_boxes_legacy(img, sections, layout)
    boxes = _strip_leading_empty_rows(boxes)
    return boxes


def _strip_leading_empty_rows(boxes: list[FactorBox]) -> list[FactorBox]:
    """各 uma の先頭行が「全 box で gold_star_count==0」なら偽陽性行として削除し、
    後続の row_index を 1 ずつ繰り上げる。

    背景: 一部の画像（UmamusumeReceiptMaker 合成 + 特定の背景色）では、因子タイル
    上部の青/赤アイコン（色チップの丸）が HSV 金★マスクに偽陽性として引っかかり、
    CNN で empty 判定された「空 row」がタイル本体より上の行として classified_rows に
    追加される。結果として row_idx が 1 つズレ、本物の青/赤因子が row=1、緑因子が
    row=2 になってしまう（位置絶対化 row=0 col=0/1=青/赤、row=1 col=0=緑 が崩壊）。

    本関数は「先頭行の全 box が gold=0」かつ「後続行に gold>=1 がある」という条件で
    先頭行を削除し、row_idx を詰め直す。gold=0 でも後続が無ければ本物の空因子行と
    見做して残す（フェイルセーフ）。
    """
    from collections import defaultdict

    per_uma: dict[int, list[FactorBox]] = defaultdict(list)
    for b in boxes:
        per_uma[b.uma_index].append(b)

    filtered: list[FactorBox] = []
    for uma_idx, uboxes in per_uma.items():
        rows: dict[int, list[FactorBox]] = defaultdict(list)
        for b in uboxes:
            rows[b.row_index].append(b)
        sorted_row_keys = sorted(rows.keys())
        # 先頭が全 gold=0 なら削除（後続に gold>=1 が残っていることが条件）
        while len(sorted_row_keys) > 1:
            first_row = sorted_row_keys[0]
            first_all_zero = all(
                (b.gold_star_count or 0) == 0 for b in rows[first_row]
            )
            rest_has_gold = any(
                (b.gold_star_count or 0) >= 1
                for rk in sorted_row_keys[1:]
                for b in rows[rk]
            )
            if first_all_zero and rest_has_gold:
                sorted_row_keys.pop(0)
            else:
                break
        # 残りを新 row_idx で再振り
        for new_idx, old_row in enumerate(sorted_row_keys):
            for b in rows[old_row]:
                b.row_index = new_idx
                filtered.append(b)
    return filtered


def _build_boxes_for_row(
    img: np.ndarray,
    uma_idx: int,
    row_idx: int,
    y_center: int,
    left_gold: list,
    right_gold: list,
    left_empty: list,
    right_empty: list,
    x_L0: int,
    x_L1: int,
    x_R0: int,
    x_R1: int,
) -> list[FactorBox]:
    """1 つの★行について左右列の FactorBox を生成する。

    空★まで併せて検出することで ★3 スロット全体の位置が確定し、
    rank 領域 (rank_bbox) は金+空の実位置から正確に算出される。
    金★個数そのものが★数になるため rank モデル推論は不要（gold_star_count を使う）。
    """
    boxes: list[FactorBox] = []
    # ★中心 y が bbox 内 y=STAR_Y_IN_TILE に来るよう bbox y0 を決める
    y_top = max(0, y_center - STAR_Y_IN_TILE)
    y_bot = min(img.shape[0], y_top + TILE_HEIGHT)
    if y_bot - y_top < TILE_HEIGHT // 2:
        return boxes

    for col_idx, (xa, xb, col_gold, col_empty) in enumerate([
        (x_L0, x_L1, left_gold, left_empty),
        (x_R0, x_R1, right_gold, right_empty),
    ]):
        col_stars_all = col_gold + col_empty
        # row 0 は各ウマ娘の青(col=0)/赤(col=1)スロット。このペアが常に存在する
        # ゲーム UI 構造を使い、★全く検出されないケースでも row 0 に限って
        # bbox を作る（位置ベース救済で pipeline 側が赤/青に割り当てる）。
        # row >= 1 はノイズを拾わないよう従来通り★未検出はスキップ。
        if not col_stars_all and row_idx > 0:
            continue
        xa_c = max(0, xa)
        xb_c = min(img.shape[1], xb)
        box_bgr = img[y_top:y_bot, xa_c:xb_c]
        if box_bgr.size == 0 or _is_blank_row(box_bgr):
            continue
        color = detect_factor_color(box_bgr)
        text_img = cv2.resize(box_bgr, (168, 16), interpolation=cv2.INTER_AREA)

        # 緑因子タイルは左端の黄色●アイコンが金★マスクに偽陽性として引っかかる。
        # bbox 右 60% に位置する★のみを ★スロット候補として採用する（金/空 共通）。
        eff_gold = col_gold
        eff_empty = col_empty
        if color == "green":
            x_threshold = xa_c + int((xb_c - xa_c) * 0.4)
            eff_gold = [s for s in col_gold if (s[0] + s[2] // 2) >= x_threshold]
            eff_empty = [s for s in col_empty if (s[0] + s[2] // 2) >= x_threshold]

        eff_all = eff_gold + eff_empty
        if eff_all:
            # ★3 スロット全体（金+空）の bbox を rank 領域とする。
            # これで金★のみの狭い crop や空★のみの crop にならず、
            # rank モデル不要（金★個数を直接★数として採用できる）になる。
            sr_x0 = min(s[0] for s in eff_all)
            sr_x1 = max(s[0] + s[2] for s in eff_all)
            sr_y0 = min(s[1] for s in eff_all)
            sr_y1 = max(s[1] + s[3] for s in eff_all)
            rx0 = max(0, sr_x0 - 2)
            rx1 = min(img.shape[1], sr_x1 + 2)
            ry0 = max(0, sr_y0 - 2)
            ry1 = min(img.shape[0], sr_y1 + 2)
            rank_bbox: tuple[int, int, int, int] | None = (rx0, ry0, rx1, ry1)
        else:
            # 全く★検出できない行（row 0 の空 bbox 用）:
            # bbox 右端の layout 比率から rank 領域を推定（rank モデル fallback 用）
            rel_x0 = 0.6786  # = FactorLayout.rank_x0_in_box_rel
            box_w = xb_c - xa_c
            rx0 = xa_c + int(round(box_w * rel_x0))
            rx1 = xb_c
            ry0 = y_top + 11
            ry1 = min(img.shape[0], y_top + 27)
            rank_bbox = None  # pipeline 側で legacy 計算

        rank_raw = img[ry0:ry1, rx0:rx1]
        if rank_raw.size == 0:
            continue
        rank_img = cv2.resize(rank_raw, (52, 16), interpolation=cv2.INTER_AREA)

        # CNN 分類器で ★スロットを金/空 再判定してから金★数を数える。
        # HSV だけでは拾えない暗め金★や、偽陽性の緑●/白UI要素を吸収できる。
        # eff_gold + eff_empty は右60%フィルタ適用済みなので、緑タイル左端の
        # 黄色●は元から候補に含まれない。
        star_candidates = eff_gold + eff_empty
        if star_candidates:
            slot_imgs: list[np.ndarray] = []
            for (sx, sy, sw, sh) in star_candidates:
                scx, scy = sx + sw // 2, sy + sh // 2
                shalf = max(sw, sh) // 2 + 4
                ssx0 = max(0, scx - shalf)
                ssx1 = min(img.shape[1], scx + shalf)
                ssy0 = max(0, scy - shalf)
                ssy1 = min(img.shape[0], scy + shalf)
                slot_imgs.append(img[ssy0:ssy1, ssx0:ssx1])
            from ..recognition.stars import predict_stars_batch

            classifications = predict_stars_batch(slot_imgs)
            # CNN で gold 判定された候補の (x_center, bbox) を保持
            gold_cand = [
                (star_candidates[i][0] + star_candidates[i][2] // 2, star_candidates[i])
                for i, (label, _) in enumerate(classifications)
                if label == "gold"
            ]
            if ENABLE_GOLD_LAYOUT_FILTER and len(gold_cand) >= 2:
                # 左詰め + 等間隔ピッチで偽陽性を除外
                gold_cand.sort(key=lambda t: t[0])
                kept = [gold_cand[0]]
                for prev, cur in zip(gold_cand, gold_cand[1:]):
                    pitch = cur[0] - prev[0]
                    if STAR_PITCH_MIN <= pitch <= STAR_PITCH_MAX:
                        kept.append(cur)
                    else:
                        break  # 連続性が切れたら以降は偽陽性扱い
                gold_cand = kept
            gold_count = min(len(gold_cand), 3)
            empty_count = min(
                sum(1 for (label, _) in classifications if label == "empty"), 3
            )
        else:
            gold_count = 0
            empty_count = 0

        # 緑タイルの★が全く拾えなかった場合、タイル右半分に HSV 閾値を
        # 緩めて再検出する（umamusume_* 系で緑背景＋★の輝度差が小さく、
        # 画像全体の HSV マスクから外れる画像が多い）。
        # 右側 60% に限定してタイル左端の黄色●アイコンを除外する。
        if color == "green" and gold_count == 0 and empty_count == 0:
            right_start = xa_c + int((xb_c - xa_c) * 0.4)
            tile_stars = img[y_top:y_bot, right_start:xb_c]
            if tile_stars.size > 0:
                gold_count, empty_count = _detect_green_tile_stars_relaxed(tile_stars)

        boxes.append(
            FactorBox(
                uma_index=uma_idx,
                row_index=row_idx,
                col_index=col_idx,
                color=color,
                text_img=text_img,
                rank_img=rank_img,
                bbox=(xa_c, y_top, xb_c, y_bot),
                rank_bbox=rank_bbox,
                gold_star_count=gold_count,
                empty_star_count=empty_count,
            )
        )
    return boxes


def _extract_factor_boxes_legacy(
    img: np.ndarray,
    sections: list[CharaSection],
    layout: FactorLayout,
) -> list[FactorBox]:
    """layout 比率依存の旧ロジック（ゲーム直撮り画像の fallback 用）。

    umacapture の実測に合わせ、因子ボックスは行 top から 27 px 高で固定クロップする
    （recognizer.json の box_height_rel = 0.0278 * 960 = 27）。
    rank 領域はボックス内 y=11..27（下部 16 px）、x=48..99（0.29..0.59 相対）。
    """
    w = img.shape[1]

    left_x0 = int(round(w * layout.left_x0))
    left_x1 = int(round(w * layout.left_x1))
    right_x0 = int(round(w * layout.right_x0))
    right_x1 = int(round(w * layout.right_x1))

    box_h = 27
    rank_y_offset = 11
    rank_h = 16

    boxes: list[FactorBox] = []
    for section in sections:
        rows = _detect_factor_rows(img, section, layout)
        for row_idx, (y_top, _y_bot) in enumerate(rows):
            box_y1 = min(img.shape[0], y_top + box_h)
            for col_idx, (xa, xb) in enumerate([(left_x0, left_x1), (right_x0, right_x1)]):
                box_bgr = img[y_top:box_y1, xa:xb]
                if box_bgr.size == 0 or _is_blank_row(box_bgr):
                    continue
                color = detect_factor_color(box_bgr)
                text_img = cv2.resize(box_bgr, (168, 16), interpolation=cv2.INTER_AREA)

                box_w = xb - xa
                rank_y0 = y_top + rank_y_offset
                rank_y1 = min(img.shape[0], rank_y0 + rank_h)
                rank_x0 = xa + int(round(box_w * layout.rank_x0_in_box_rel))
                rank_x1 = min(img.shape[1], xa + int(round(box_w * layout.rank_x1_in_box_rel)))
                rank_raw = img[rank_y0:rank_y1, rank_x0:rank_x1]
                if rank_raw.size == 0:
                    continue
                rank_img = cv2.resize(rank_raw, (52, 16), interpolation=cv2.INTER_AREA)

                boxes.append(
                    FactorBox(
                        uma_index=section.uma_index,
                        row_index=row_idx,
                        col_index=col_idx,
                        color=color,
                        text_img=text_img,
                        rank_img=rank_img,
                        bbox=(xa, y_top, xb, box_y1),
                    )
                )
    return boxes


def _is_blank_row(box_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY)
    return bool(gray.std() < 10.0)


def _crop_rank_region(box_bgr: np.ndarray, layout: FactorLayout) -> np.ndarray:
    h, w = box_bgr.shape[:2]
    x0 = int(round(w * layout.rank_x0_in_box_rel))
    x1 = int(round(w * layout.rank_x1_in_box_rel))
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    rank_region = box_bgr[:, x0:x1]
    return cv2.resize(rank_region, (52, 16), interpolation=cv2.INTER_AREA)
