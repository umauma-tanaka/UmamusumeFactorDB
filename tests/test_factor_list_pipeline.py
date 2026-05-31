from __future__ import annotations

import numpy as np
import pytest

from umafactor.factor_list import pipeline as flp
from umafactor.detection.factor_list import FactorListDetection, FactorListTile
from umafactor.factor_list import FactorOcrOptions, FactorOcrResult, RecognizedFactor
from umafactor.recognition import factor_name_matcher as fnm
from umafactor.recognition.factor_list_ocr import factor_list_name_region_bbox
from umafactor.schema import COLUMNS


def _factor(
    *,
    role: str = "parent",
    order: int = 0,
    category: str = "white",
    name: str = "直線巧者",
    stars: int = 2,
) -> RecognizedFactor:
    return RecognizedFactor(
        role=role,  # type: ignore[arg-type]
        order=order,
        row=order,
        col=0,
        raw_name=name,
        normalized_name=name,
        category=category,
        stars=stars,
        bbox=(10, 10, 100, 50),
        ocr_bbox=(20, 12, 80, 30),
        ocr_confidence=None,
        match_confidence=None,
        detection_confidence=None,
        needs_review=False,
    )


def _tile(
    *,
    order: int,
    role: str = "parent",
    color: str = "white",
    star: int = 2,
) -> FactorListTile:
    return FactorListTile(
        order=order,
        section_index=0,
        role=role,  # type: ignore[arg-type]
        row_index=order,
        col_index=0,
        color=color,  # type: ignore[arg-type]
        star=star,
        bbox=(20, 20 + order * 70, 260, 80 + order * 70),
        bbox_norm=(20, 20 + order * 70, 260, 80 + order * 70),
    )


class _FakeOcr:
    def recognize(self, img_bgr):
        return "直線巧者"

    def recognize_blue(self, img_bgr):
        return "スピード"

    def recognize_red(self, img_bgr):
        return "短距離"

    def recognize_with_parts(self, img_bgr):
        return "弧線のプロフェッサー", ["弧線のプロフェッサー"]


def _clear_factor_name_caches() -> None:
    fnm.clear_factor_name_matcher_caches()
    flp._factor_name_candidates.cache_clear()
    flp._all_factor_names.cache_clear()
    flp._skill_factor_names.cache_clear()
    flp._skill_master_names.cache_clear()


def test_to_submission_maps_factor_list_roles_without_changing_sheet_shape() -> None:
    result = FactorOcrResult(
        factors=[
            _factor(role="parent", order=0, category="blue", name="スピード", stars=3),
            _factor(role="parent", order=1, category="red", name="短距離", stars=2),
            _factor(role="parent", order=2, category="green", name="弧線のプロフェッサー", stars=1),
            _factor(role="parent", order=3, category="white", name="直線巧者", stars=2),
            _factor(role="ancestor1", order=0, category="white", name="伏兵○", stars=1),
            _factor(role="ancestor2", order=0, category="white", name="末脚", stars=3),
        ],
        stitched_image_path=None,
        debug_dir=None,
    )

    submission = flp.to_submission(result, submitter_id="tester", image_path="captures/factor.png")

    assert submission.image_filename == "factor.png"
    assert submission.main.blue_type == "スピード"
    assert submission.main.blue_star == 3
    assert submission.main.red_type == "短距離"
    assert submission.main.green_name == "弧線のプロフェッサー"
    assert submission.main.skills[0].name == "直線巧者"
    assert submission.parent1.skills[0].name == "伏兵○"
    assert submission.parent2.skills[0].name == "末脚"
    rows = submission.to_rows()
    assert len(rows) == 3
    assert all(len(row) == len(COLUMNS) for row in rows)


def test_recognize_factor_list_image_uses_injected_ocr_without_debug_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_detect(image, *, section_index: int):
        if section_index > 0:
            raise IndexError("missing section")
        return FactorListDetection(
            image_width=image.shape[1],
            image_height=image.shape[0],
            scale=1.0,
            section_index=section_index,
            role="parent",
            tiles=[
                _tile(order=0, color="blue", star=3),
                _tile(order=1, color="red", star=2),
                _tile(order=2, color="green", star=1),
                _tile(order=3, color="white", star=2),
            ],
        )

    monkeypatch.setattr(flp, "detect_factor_list_cards", fake_detect)
    image = np.full((400, 400, 3), 255, dtype=np.uint8)

    result = flp.recognize_factor_list_image(
        image,
        options=FactorOcrOptions(
            use_paddle=False,
            preprocess_crop=False,
            ocr_execution_mode="sequential",
        ),
        ocr=_FakeOcr(),
    )

    assert [factor.normalized_name for factor in result.factors] == [
        "スピード",
        "短距離",
        "弧線のプロフェッサー",
        "直線巧者",
    ]
    assert [factor.needs_review for factor in result.factors] == [False, False, True, False]
    assert all(factor.match_confidence is not None for factor in result.factors)
    assert all(factor.detection_confidence == 1.0 for factor in result.factors)
    assert all(factor.ocr_bbox is not None for factor in result.factors)
    assert result.debug_dir is None
    assert result.stitched_image_path is None
    assert list(tmp_path.iterdir()) == []


def test_recognize_factor_list_image_writes_debug_overlay_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_detect(image, *, section_index: int):
        if section_index > 0:
            raise IndexError("missing section")
        return FactorListDetection(
            image_width=image.shape[1],
            image_height=image.shape[0],
            scale=1.0,
            section_index=section_index,
            role="parent",
            tiles=[_tile(order=0, color="white", star=2)],
        )

    monkeypatch.setattr(flp, "detect_factor_list_cards", fake_detect)
    image = np.full((180, 320, 3), 255, dtype=np.uint8)

    result = flp.recognize_factor_list_image(
        image,
        options=FactorOcrOptions(
            use_paddle=False,
            debug_dir=tmp_path,
            enable_overlay=True,
        ),
    )

    assert result.stitched_image_path == tmp_path / "stitched.png"
    assert (tmp_path / "stitched.png").exists()
    assert (tmp_path / "factor_ocr_result.json").exists()
    assert (tmp_path / "factor_ocr_overlay.png").exists()
    assert (tmp_path / "factor_ocr_debug.csv").exists()
    crop_names = {path.name for path in (tmp_path / "ocr_crops").iterdir()}
    assert "tile_000_parent_000_card.png" in crop_names
    assert "tile_000_parent_000_roi_raw_body_name.png" in crop_names
    assert "tile_000_parent_000_roi_upscaled_body_name.png" in crop_names
    assert "tile_000_parent_000_roi_preprocessed_body_name_raw_upscaled.png" in crop_names


def test_recognize_factor_list_image_builds_default_ocr_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_detect(image, *, section_index: int):
        if section_index > 0:
            raise IndexError("missing section")
        return FactorListDetection(
            image_width=image.shape[1],
            image_height=image.shape[0],
            scale=1.0,
            section_index=section_index,
            role="parent",
            tiles=[_tile(order=0), _tile(order=1), _tile(order=2)],
        )

    factory_calls = 0

    def fake_get_ocr(options):
        nonlocal factory_calls
        factory_calls += 1
        return _FakeOcr()

    monkeypatch.setattr(flp, "detect_factor_list_cards", fake_detect)
    monkeypatch.setattr(flp, "get_factor_ocr", fake_get_ocr)

    image = np.full((300, 320, 3), 255, dtype=np.uint8)
    flp.recognize_factor_list_image(
        image,
        options=FactorOcrOptions(
            preprocess_crop=False,
            ocr_execution_mode="sequential",
        ),
    )

    assert factory_calls == 1


def test_factor_ocr_options_defaults_use_name_roi_and_batched_recognition() -> None:
    options = FactorOcrOptions()

    assert options.ocr_crop_target == "name"
    assert options.crop_variant == "body_name"
    assert options.ocr_roi_profiles == ("body_name",)
    assert options.ocr_preprocess_modes == (
        "raw_upscaled",
        "gray_sharpen",
        "color_text_safe",
    )
    assert options.paddle_mode == "recognition"
    assert options.ocr_mode == "rapidocr"
    assert options.rapidocr_model_root_dir is None
    assert options.rapidocr_text_score is None
    assert options.rapidocr_ocr_version == "PP-OCRv4"
    assert options.rapidocr_lang_type == "japan"
    assert options.rapidocr_model_type == "mobile"
    assert options.rapidocr_rec_img_shape == (3, 48, 480)
    assert options.ocr_execution_mode == "batch"
    assert options.ocr_batch_size == 12
    assert options.ocr_sheet_max_side == 3600
    assert options.ocr_sheet_columns is None


def test_normalize_factor_name_removes_stars_and_repairs_common_ocr_variants() -> None:
    assert flp._normalize_factor_name("★★★樱花赏") == "桜花賞"
    assert flp._normalize_factor_name("巨步★★") == "巨歩"
    assert flp._normalize_factor_name("ウィクトリアマイル") == "ヴィクトリアマイル"
    assert flp._normalize_factor_name("BC･テルマー☆") == "BC・テルマー"
    assert flp._normalize_factor_name("伏兵0") == "伏兵○"
    assert flp._normalize_factor_name("大井レース場o") == "大井レース場○"
    assert flp._normalize_factor_name("0テンポアップ") == "0テンポアップ"


def test_fuzzy_match_repairs_known_factor_name_variants() -> None:
    candidate, score = flp._match_factor_name("スピート", "blue")

    assert candidate == "スピード"
    assert score is not None
    assert score >= FactorOcrOptions().fuzzy_match_threshold

    candidate, score = flp._match_factor_name("タード", "red")
    assert candidate == "ダート"
    assert score is not None
    assert score >= FactorOcrOptions().fuzzy_match_threshold

    candidate, score = flp._match_factor_name("マイルS", "white")
    assert candidate == "マイルCS"
    assert score is not None
    assert score >= FactorOcrOptions().fuzzy_review_threshold

    candidate, score = flp._match_factor_name("マイル", "red")
    assert candidate == "マイル"
    assert score is not None
    assert score >= FactorOcrOptions().fuzzy_match_threshold

    candidate, score = flp._match_factor_name("ウィクトウマイル", "white")
    assert candidate == "ヴィクトリアマイル"
    assert score is not None
    assert score >= FactorOcrOptions().fuzzy_match_threshold


def test_white_factor_names_include_generated_skill_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fnm, "load_labels", lambda: {"factor.name": []})
    monkeypatch.setattr(fnm, "green_factor_names", lambda: [])
    monkeypatch.setattr(fnm, "load_skill_master_names", lambda: ["SampleSkill"])
    _clear_factor_name_caches()

    try:
        candidate, score = flp._match_factor_name("SampleSkil", "white")

        assert candidate == "SampleSkill"
        assert score is not None
        assert score >= FactorOcrOptions().fuzzy_match_threshold
        assert "SampleSkill" in fnm._all_factor_names()
    finally:
        _clear_factor_name_caches()


def test_skill_master_is_auxiliary_after_existing_factor_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fnm,
        "load_labels",
        lambda: {"factor.name": ["RaceName", "OverlapName"]},
    )
    monkeypatch.setattr(fnm, "green_factor_names", lambda: [])
    monkeypatch.setattr(fnm, "load_skill_master_names", lambda: ["OverlapName", "ExtraSkill"])
    _clear_factor_name_caches()

    try:
        names = fnm._skill_factor_names()

        assert names[:3] == ("RaceName", "OverlapName", "ExtraSkill")
        assert names.count("OverlapName") == 1
    finally:
        _clear_factor_name_caches()


def test_default_upper_name_roi_excludes_left_icon_area() -> None:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    tile = _tile(order=0)

    x0, _y0, _x1, _y1 = factor_list_name_region_bbox(image, tile, variant="upper")
    tile_x0, _tile_y0, tile_x1, _tile_y1 = tile.bbox
    tile_width = tile_x1 - tile_x0

    assert x0 >= tile_x0 + int(round(tile_width * 0.13))
