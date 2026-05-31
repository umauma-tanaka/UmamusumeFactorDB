from __future__ import annotations

from umafactor.recognition.factor_name_matcher import FactorNameMatcher, normalize_factor_name
from umafactor.recognition.factor_name_ocr_pipeline import (
    FactorNameOcrInput,
    score_factor_name_ocr,
    select_best_factor_name_ocr,
)
from umafactor.recognition.name_roi import NameRoiOptions, compute_name_roi_from_body


def test_name_roi_uses_card_body_geometry() -> None:
    debug = compute_name_roi_from_body(
        (200, 400, 3),
        (100, 50, 300, 110),
        options=NameRoiOptions(),
    )

    assert debug.text_bbox == (126, 53, 294, 91)
    assert debug.icon_exclusion_bbox == (100, 50, 126, 110)


def test_factor_name_normalize_repairs_common_ocr_variants() -> None:
    assert normalize_factor_name("★★桜花赏") == "桜花賞"
    assert normalize_factor_name("巨步☆") == "巨歩"
    assert normalize_factor_name("右回りO") == "右回り○"
    assert normalize_factor_name("BC･テルマー") == "BC・テルマー"


def test_factor_name_normalize_repairs_known_rapidocr_variants() -> None:
    assert normalize_factor_name("右回リ0タ") == "右回り○"
    assert normalize_factor_name("札幌レース場Oー") == "札幌レース場○"
    assert normalize_factor_name("ヴイクトリアマイルー") == "ヴィクトリアマイルー"
    assert normalize_factor_name("エリサベス女王杯ー") == "エリザベス女王杯ー"
    assert normalize_factor_name("チャンピオンスCーラ") == "チャンピオンズC"
    assert normalize_factor_name("熱華の洗礼一") == "烈華の洗礼一"


def test_factor_name_candidate_selection_prefers_master_match() -> None:
    matcher = FactorNameMatcher()
    scores = [
        score_factor_name_ocr(
            FactorNameOcrInput(
                raw_text="マイルS",
                category="white",
                roi_profile="body_name",
                preprocess_mode="raw_upscaled",
            ),
            matcher=matcher,
            auto_threshold=0.92,
            review_threshold_value=0.82,
        ),
        score_factor_name_ocr(
            FactorNameOcrInput(
                raw_text="Cマル",
                category="white",
                roi_profile="body_name",
                preprocess_mode="gray_sharpen",
            ),
            matcher=matcher,
            auto_threshold=0.92,
            review_threshold_value=0.82,
        ),
    ]

    best, selected = select_best_factor_name_ocr(scores)

    assert best.canonical_name == "マイルCS"
    assert best.match_score is not None
    assert best.needs_review
    assert [score.selected for score in selected].count(True) == 1
