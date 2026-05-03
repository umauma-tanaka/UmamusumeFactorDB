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
    recognize_factor_candidates,
)
from .recognition.characters import (
    apply_unique_skill_character_overrides,
    recognize_characters,
)
from .recognition.green_prepass import compute_green_prepass
from .recognition.image_crops import (
    crop_from_original as _crop_from_original,
    crop_rank_from_original as _crop_rank_from_original,
    display_crop_for_slot as _display_crop_for_slot,
    display_crop_from_original as _display_crop_from_original,
    extract_character_icon_bgr as _extract_character_icon_bgr,
)
from .recognition.slots import (
    classify_factor_slot,
    should_adopt_green_box,
)
from .recognition.star_rank import (
    apply_missing_green_star_fallbacks,
    predict_factor_star,
)
from .review import ReviewQueue
from .schema import Submission, UmaFactors


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

    recognize_characters(umas, sections, norm_img, char_pred)

    # Pass 0: 各 uma の緑 box 候補について OCR top1 conf と最大 gold_star_count を
    # 別々に事前計算する。従来の「gold_star_count>0 の先着 box を採用」だと、
    # ★全空の行（row=1 col=0、実はテキストが正解）が skip されて row=2 の
    # 別 box（OCR は空や雑音）が採用される事故が多発していた。
    # 因子名は「OCR top1 conf 最大の box」、★は「同 uma 内の緑 box の最大
    # gold_star_count」と別軸で採用することで、rank fallback が誤★を返す問題も回避。
    # 緑候補には UI 仕様上 row=1 col=0（青因子の下）の絶対位置 box も必ず含める。
    # detect_factor_color が緑→white 等に誤判定しても、位置ベースで緑候補に入れる
    # ことで「緑因子が白スキルに流入する」事故を防ぐ。
    green_prepass = compute_green_prepass(boxes, img_orig, scale, ocr)
    best_green_box = green_prepass.best_green_box
    best_green_score = green_prepass.best_green_score
    any_green_gold = green_prepass.any_green_gold

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
        display_crop = _display_crop_for_slot(
            img_orig,
            norm_img.shape,
            box.bbox,
            scale,
            is_blue_slot=is_blue_slot,
            is_red_slot=is_red_slot,
        )
        candidate_recognition = recognize_factor_candidates(
            factor_pred,
            ocr,
            img_orig,
            text_crop_norm,
            display_crop,
            box.bbox,
            scale,
            is_blue_slot=is_blue_slot,
            is_red_slot=is_red_slot,
            green_adoptable=green_adoptable,
            green_name_set=green_name_set,
        )
        merged = candidate_recognition.candidates
        sources = candidate_recognition.sources
        top_name = candidate_recognition.top_name
        ocr_raw = candidate_recognition.ocr_raw

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
    apply_unique_skill_character_overrides(umas, load_unique_skill_to_character())

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
