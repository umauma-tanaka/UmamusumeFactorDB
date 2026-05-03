"""画像 → (Submission, ReviewQueue) への統合パイプライン。

ReviewQueue は低信頼度の因子をユーザにレビューしてもらうための候補リスト。
"""

from __future__ import annotations

from .config import load_unique_skill_to_character
from .recognition.characters import (
    apply_unique_skill_character_overrides,
    recognize_characters,
)
from .recognition.context import build_recognition_context
from .recognition.factor_recognition import recognize_factor_box
from .recognition.green_prepass import compute_green_prepass
from .recognition.image_crops import (
    crop_from_original as _crop_from_original,
    crop_rank_from_original as _crop_rank_from_original,
    display_crop_for_slot as _display_crop_for_slot,
    display_crop_from_original as _display_crop_from_original,
    extract_character_icon_bgr as _extract_character_icon_bgr,
)
from .recognition.image_preprocessing import (
    dump_debug_crops as _dump_debug_crops,
    prepare_factor_image,
)
from .recognition.star_rank import (
    apply_missing_green_star_fallbacks,
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
    prepared = prepare_factor_image(
        image_path,
        debug_crops_dir=debug_crops_dir,
        auto_debug=auto_debug,
    )
    img_orig = prepared.img_orig
    norm_img = prepared.norm_img
    scale = prepared.scale
    sections = prepared.sections
    boxes = prepared.boxes

    context = build_recognition_context(skip_ocr=skip_ocr)

    umas = [UmaFactors(), UmaFactors(), UmaFactors()]
    review = ReviewQueue()
    white_counters = {0: 0, 1: 0, 2: 0}

    recognize_characters(umas, sections, norm_img, context.char_pred)

    # Pass 0: 各 uma の緑 box 候補について OCR top1 conf と最大 gold_star_count を
    # 別々に事前計算する。従来の「gold_star_count>0 の先着 box を採用」だと、
    # ★全空の行（row=1 col=0、実はテキストが正解）が skip されて row=2 の
    # 別 box（OCR は空や雑音）が採用される事故が多発していた。
    # 因子名は「OCR top1 conf 最大の box」、★は「同 uma 内の緑 box の最大
    # gold_star_count」と別軸で採用することで、rank fallback が誤★を返す問題も回避。
    # 緑候補には UI 仕様上 row=1 col=0（青因子の下）の絶対位置 box も必ず含める。
    # detect_factor_color が緑→white 等に誤判定しても、位置ベースで緑候補に入れる
    # ことで「緑因子が白スキルに流入する」事故を防ぐ。
    green_prepass = compute_green_prepass(boxes, img_orig, scale, context.ocr)
    best_green_box = green_prepass.best_green_box
    best_green_score = green_prepass.best_green_score
    any_green_gold = green_prepass.any_green_gold

    for box in boxes:
        recognized = recognize_factor_box(
            umas,
            white_counters,
            context.factor_pred,
            context.rank_pred,
            context.ocr,
            img_orig,
            norm_img,
            box,
            boxes,
            scale,
            context.green_name_set,
            best_green_box,
            best_green_score,
        )
        review.add(recognized.review_item)

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
