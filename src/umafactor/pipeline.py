"""画像 → (Submission, ReviewQueue) への統合パイプライン。

ReviewQueue は低信頼度の因子をユーザにレビューしてもらうための候補リスト。
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import green_factor_names, load_unique_skill_to_character
from .cropper import (
    BASE_WIDTH,
    CharaSection,
    FactorBox,
    detect_chara_sections,
    extract_factor_boxes,
    normalize_width,
)
from .infer import get_predictor
from .ocr import get_ocr
from .recognition.assignment import apply_factor_result, build_review_item
from .recognition.candidate_generation import (
    filter_slot_candidates,
    match_template_candidates,
    predict_onnx_candidates,
    recognize_ocr_candidates,
)
from .recognition.candidate_fusion import (
    merge_candidates as _merge_candidates,
    merge_candidates_v2 as _merge_candidates_v2,
)
from .recognition.image_crops import (
    crop_from_original as _crop_from_original,
    crop_rank_from_original as _crop_rank_from_original,
    display_crop_from_original as _display_crop_from_original,
    extract_character_icon_bgr as _extract_character_icon_bgr,
)
from .recognition.slots import (
    classify_factor_slot,
    is_green_candidate_box,
    should_adopt_green_box,
)
from .recognition.star_rank import (
    apply_missing_green_star_fallbacks,
    predict_factor_star,
)
from .review import ReviewQueue
from .schema import Submission, UmaFactors
from .templates import match_green_name


def analyze_image(
    image_path: str,
    submitter_id: str,
    debug_crops_dir: str | None = None,
    auto_debug: bool = True,
    skip_ocr: bool = False,
) -> tuple[Submission, ReviewQueue]:
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
        _dump_debug_crops(norm_img, sections, boxes, debug_crops_dir)
    elif auto_debug and len(boxes) < 9:
        # 因子 box が 3 ウマ × 3 行 = 9 個未満になった画像は cropper 失敗の疑い。
        # 後から目視で原因究明できるよう自動で debug crops を dump する。
        import os
        stem = os.path.splitext(os.path.basename(image_path))[0]
        auto_dir = os.path.join("tests", "fixtures", "debug_crops", stem)
        _dump_debug_crops(norm_img, sections, boxes, auto_dir)

    factor_pred = get_predictor("factor")
    rank_pred = get_predictor("factor_rank")
    char_pred = get_predictor("character")
    ocr = None if skip_ocr else get_ocr()
    # 緑因子（固有スキル 249 件）の名前セット。非緑スロットの ONNX/OCR 候補から
    # 除外するフィルタに使う。
    green_name_set: set[str] = set(
        green_factor_names() if skip_ocr else ocr._green_factor_names
    )

    umas = [UmaFactors(), UmaFactors(), UmaFactors()]
    review = ReviewQueue()
    white_counters = {0: 0, 1: 0, 2: 0}

    for section in sections:
        icon = _extract_character_icon_bgr(norm_img, section)
        pred = char_pred.predict(icon)
        umas[section.uma_index].character = pred.label

    # Pass 0: 各 uma の緑 box 候補について OCR top1 conf と最大 gold_star_count を
    # 別々に事前計算する。従来の「gold_star_count>0 の先着 box を採用」だと、
    # ★全空の行（row=1 col=0、実はテキストが正解）が skip されて row=2 の
    # 別 box（OCR は空や雑音）が採用される事故が多発していた。
    # 因子名は「OCR top1 conf 最大の box」、★は「同 uma 内の緑 box の最大
    # gold_star_count」と別軸で採用することで、rank fallback が誤★を返す問題も回避。
    # 緑候補には UI 仕様上 row=1 col=0（青因子の下）の絶対位置 box も必ず含める。
    # detect_factor_color が緑→white 等に誤判定しても、位置ベースで緑候補に入れる
    # ことで「緑因子が白スキルに流入する」事故を防ぐ。
    best_green_box: dict[int, FactorBox] = {}
    best_green_score: dict[int, float] = {}
    best_green_gold: dict[int, int] = {}  # 名前採用時の最大 gold（col=0 のみ）
    any_green_gold: dict[int, int] = {}  # ★補填用の最大 gold（col 問わず）
    for box in boxes:
        # 緑候補: 色判定 green の box、または UI 仕様上の絶対位置（row=1 col=0）。
        # 位置ベース候補を加えることで、色チップ検出が white 等に失敗しても
        # 緑因子が拾える（ユーザー報告: 緑因子が白因子扱いで混入した）。
        if not is_green_candidate_box(box):
            continue
        g = box.gold_star_count or 0
        if g > any_green_gold.get(box.uma_index, 0):
            any_green_gold[box.uma_index] = g
        # 緑因子は UI 仕様上 col=0（左側）のみ。col=1（右側＝白/スキル列）で
        # detect_factor_color が緑誤判定するケース（受領 1558 parent1/parent2、
        # 1814 main/parent1 等）があり、そこを採用するとレース名行の OCR 結果を
        # 緑 name にしてしまう事故が発生する。col=1 は名前採用候補から除外。
        # ただし★数は col=1 でも偽陽性タイルとして★が正しく取れていることが
        # 多いため、any_green_gold で保持して後段 Pass 2 の★補填に使う。
        if box.col_index != 0:
            continue
        dc = _display_crop_from_original(img_orig, box.bbox, scale)
        if ocr is None:
            cands = []
        else:
            raw, frags = ocr.recognize_with_parts(dc)
            cands = ocr.match_to_green_factor_multi(raw, frags, top_k=1)
        top_conf = cands[0][1] if cands else 0.0
        # OCR が空 / 低スコアでも、テンプレマッチで box を比較できるよう
        # 緑名前テンプレ top1 のスコアも加味する。同 uma 内に row=1 col=0 と
        # row=2 col=0 の両方が color=green と判定されるケース（umamusume_182056 等）
        # で、テンプレマッチが高スコアの方を best_green_box に選べる。
        _gnx0, _gny0, _gnx1, _gny1 = box.bbox
        _gn_x1 = _gnx0 + int((_gnx1 - _gnx0) * 0.85)
        _gn_crop = _display_crop_from_original(
            img_orig, (_gnx0, _gny0, _gn_x1, _gny1), scale, pad_y_norm=2,
        )
        _gn_matches = match_green_name(_gn_crop)
        _gn_conf = _gn_matches[0][1] if _gn_matches else 0.0
        combined_conf = max(top_conf, _gn_conf)
        uidx = box.uma_index
        if combined_conf > best_green_score.get(uidx, 0.0):
            best_green_score[uidx] = combined_conf
            best_green_box[uidx] = box
        g = box.gold_star_count or 0
        if g > best_green_gold.get(uidx, 0):
            best_green_gold[uidx] = g

    for box in boxes:
        x0, y0, x1, y1 = box.bbox
        text_crop_norm = norm_img[y0:y1, x0:x1]

        # 色チップ検出が弱い合成画像で box.color が "white" / 逆の青赤 に落ちても、
        # 因子が常に決まった位置に並ぶゲーム UI 構造を使って位置で補正する。
        #   row 0 col 0 → 青因子（左上）
        #   row 0 col 1 → 赤因子（青の右）
        #   row 1 col 0 → 緑因子（青の下）
        # は必ず存在するため、この 3 セルは位置で絶対確定し color 判定は無視する。
        # これで「緑因子が色判定 white に落ちて白スキル行に混入」「row 0 col 1 が
        # 青と誤判定され blue slot に先取りされる」等の事故を防ぐ。
        # row>=2 col=0 で色判定 green になる box（稀だが本物緑が cropper の
        # 都合で row=2 側に検出される画像あり）は fallback で緑スロット候補に残す。
        slot_flags = classify_factor_slot(box)
        is_blue_slot = slot_flags.is_blue
        is_red_slot = slot_flags.is_red
        is_green_slot = slot_flags.is_green

        # この box が本当に緑スロットとして採用される見込みがあるか。
        # is_green_slot=True でも best_green_box に選ばれなかった box は結局
        # skills 行きになるため、緑辞書（249 種の固有スキル）OCR 処理をすると
        # skills に緑辞書マッチ名が紛れ込む（例: '白い稲妻、見せたるで！'）。
        # 緑として採用される見込みがない box は通常 OCR ルートに流し、
        # 緑辞書は緑スロットに限定する。uma.green_name の逐次状態で先着判定。
        uidx_cur = box.uma_index
        green_adoptable = should_adopt_green_box(
            box,
            boxes,
            is_green_slot=is_green_slot,
            current_green_name=umas[uidx_cur].green_name,
            best_green_box=best_green_box.get(uidx_cur),
            best_green_score=best_green_score.get(uidx_cur, 0.0),
        )

        # 青スロットは box.bbox が★中心基準で算出されており、一部画像で因子名
        # テキストが bbox 下端からはみ出して OCR 入力に映らない問題がある。
        # display_crop の pad_y_norm を 8 に拡大することで「スピード」「スタミナ」等が
        # OCR で拾えるようになり青 +3 件改善を確認。
        # 赤は pad_y_norm 両方向拡張（Exp 3）で「長距離→マイル」悪化が発生したが、
        # 真因は bbox が★中心基準で上にズレてテキストが下端にはみ出すこと。
        # display_crop の元 bbox を y1 のみ +14 に拡張（y0 は維持）で、上の行を
        # 含まずテキストだけを入れる非対称 pad に変更する。
        if is_blue_slot:
            display_crop = _display_crop_from_original(
                img_orig, box.bbox, scale, pad_y_norm=8
            )
        elif is_red_slot:
            img_h = norm_img.shape[0]
            red_disp_bbox = (x0, y0, x1, min(img_h, y1 + 14))
            display_crop = _display_crop_from_original(
                img_orig, red_disp_bbox, scale, pad_y_norm=2
            )
        else:
            display_crop = _display_crop_from_original(img_orig, box.bbox, scale)
        ext_bbox = box.bbox
        ext_text_crop_norm = text_crop_norm

        onnx_candidates = predict_onnx_candidates(
            factor_pred,
            img_orig,
            ext_text_crop_norm,
            box.bbox,
            ext_bbox,
            scale,
            is_blue_slot=is_blue_slot,
            is_red_slot=is_red_slot,
        )
        ocr_raw, ocr_candidates = recognize_ocr_candidates(
            ocr,
            display_crop,
            is_blue_slot=is_blue_slot,
            is_red_slot=is_red_slot,
            green_adoptable=green_adoptable,
        )
        onnx_candidates, ocr_candidates = filter_slot_candidates(
            onnx_candidates,
            ocr_candidates,
            is_blue_slot=is_blue_slot,
            is_red_slot=is_red_slot,
            green_adoptable=green_adoptable,
            green_name_set=green_name_set,
        )
        template_candidates = match_template_candidates(
            display_crop,
            img_orig,
            box.bbox,
            scale,
            is_red_slot=is_red_slot,
            is_blue_slot=is_blue_slot,
            green_adoptable=green_adoptable,
        )

        # マージ（緑スロットは OCR top1 が正解を出すケースでも全 813 辞書の ONNX top1 に
        # 押し負けやすいため、ocr_strong_threshold を 0.5 に緩和して OCR を優先する）
        merge_threshold = 0.5 if green_adoptable else 0.7
        merged, sources = _merge_candidates(
            onnx_candidates,
            ocr_candidates,
            limit=8,
            ocr_strong_threshold=merge_threshold,
        )

        # 赤/青/緑スロットでテンプレマッチ top1 が強い場合、最終 top_name として採用する。
        # merged 側に top_name が存在しない場合もあるため、候補として追加する。
        # 閾値: 赤/青(10/5 カテゴリ)は 0.90、緑名前(46 カテゴリ)はサンプル数が
        # 偏るため 0.95 と厳しめに。
        if template_candidates:
            t_name, t_score = template_candidates[0]
            t_threshold = 0.95 if green_adoptable else 0.90
            if t_score >= t_threshold:
                merged = [(t_name, t_score)] + [(n, s) for n, s in merged if n != t_name]
                sources[t_name] = "template"
        top_name = merged[0][0] if merged else ""

        star = predict_factor_star(rank_pred, img_orig, box, scale)

        assignment = apply_factor_result(
            umas[box.uma_index],
            white_counters,
            img_orig,
            box,
            boxes,
            scale,
            top_name,
            star,
            is_blue_slot=is_blue_slot,
            is_red_slot=is_red_slot,
            green_adoptable=green_adoptable,
        )
        review.add(
            build_review_item(
                box,
                assignment,
                display_crop,
                merged,
                sources,
                ocr_raw,
                top_name,
                star,
            )
        )

    # Pass 2: 緑 col=1 除外により uma.green_name が未採用だが、★は col=1 に
    # 残っているケースがある（受領 1558 parent1/parent2 等）。name は空のまま
    # ★だけ any_green_gold から補填する。評価時に name 誤認件数は変わらないが、
    # ★は正解にできる＝★悪化を防げる。
    apply_missing_green_star_fallbacks(umas, any_green_gold)

    # 緑因子（固有スキル）から character を逆引き：
    # character は ONNX の画像分類だと衣装差などで誤判定しやすいが、
    # 固有スキルは一意に衣装（カード）を決めるため、マッピングが一致する場合は
    # そちらを優先する。マッピングに無い場合は ONNX 結果を残す。
    # 注: 継承タブ画像（親由来の継承スキル）では逆引き先が自分の衣装と一致せず
    # 誤上書きする副作用があるが、育成情報タブ画像での精度向上を優先するため
    # 無条件適用とする。タブ種別が画像から判別できるようになれば再検討する。
    unique_map = load_unique_skill_to_character()
    if unique_map:
        for uma in umas:
            if not uma.green_name:
                continue
            # OCR が前後スペースを含む green_name を返すケースに備え、strip した値でも照合する。
            # uma.green_name 本体は変更せず、逆引きのキーマッチング時のみ正規化する。
            key_candidates = [uma.green_name, uma.green_name.strip()]
            for k in key_candidates:
                if k in unique_map:
                    uma.character = unique_map[k]
                    break

    import os
    submission = Submission(
        submitter_id=submitter_id,
        image_filename=os.path.basename(image_path),
        main=umas[0],
        parent1=umas[1],
        parent2=umas[2],
    )
    return submission, review


def apply_review_results(submission: Submission, review: ReviewQueue) -> None:
    """ユーザレビュー後の ReviewItem の reviewed_name / reviewed_star を Submission に反映。"""
    umas = [submission.main, submission.parent1, submission.parent2]
    for item in review.items:
        if item.reviewed_name is None:
            continue
        uma = umas[item.uma_index]
        star = item.reviewed_star if item.reviewed_star is not None else item.current_star
        if item.slot == "blue":
            uma.blue_type = item.reviewed_name
            uma.blue_star = star
        elif item.slot == "red":
            uma.red_type = item.reviewed_name
            uma.red_star = star
        elif item.slot == "green":
            uma.green_name = item.reviewed_name
            uma.green_star = star
        elif item.slot == "white":
            if 0 <= item.white_index < len(uma.skills):
                uma.skills[item.white_index].name = item.reviewed_name
                uma.skills[item.white_index].star = star


def _dump_debug_crops(
    img: np.ndarray,
    sections: list[CharaSection],
    boxes: list[FactorBox],
    out_dir: str,
) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
    overlay = img.copy()
    for s in sections:
        x0, y0, x1, y1 = s.portrait_bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for b in boxes:
        x0, y0, x1, y1 = b.bbox
        color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}.get(
            b.color, (255, 255, 255)
        )
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1)
    cv2.imwrite(os.path.join(out_dir, "_overlay.png"), overlay)
    for b in boxes:
        base = f"uma{b.uma_index}_row{b.row_index:02d}_col{b.col_index}_{b.color}"
        cv2.imwrite(os.path.join(out_dir, f"{base}_text.png"), b.text_img)
        cv2.imwrite(os.path.join(out_dir, f"{base}_rank.png"), b.rank_img)
