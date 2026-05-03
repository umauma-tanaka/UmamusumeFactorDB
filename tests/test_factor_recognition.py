from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umafactor.recognition import factor_recognition
from umafactor.recognition.candidate_generation import FactorCandidateRecognition
from umafactor.recognition.context import RecognitionContext
from umafactor.recognition.factor_recognition import (
    RecognizedFactorBox,
    recognize_factor_box,
    run_factor_recognition,
)
from umafactor.recognition.green_prepass import GreenPrepassResult
from umafactor.recognition.image_preprocessing import PreparedFactorImage
from umafactor.recognition.slots import SlotFlags
from umafactor.review import ReviewItem
from umafactor.schema import UmaFactors


@dataclass
class DummyBox:
    uma_index: int = 1
    row_index: int = 2
    col_index: int = 1
    color: str = "white"
    bbox: tuple[int, int, int, int] = (20, 10, 40, 24)
    rank_bbox: tuple[int, int, int, int] | None = None
    gold_star_count: int | None = 0


@dataclass
class DummySection:
    uma_index: int = 0
    portrait_bbox: tuple[int, int, int, int] = (0, 0, 10, 10)


def _image(height: int = 80, width: int = 100) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def _review_item(uma_index: int, name: str) -> ReviewItem:
    return ReviewItem(
        uma_index=uma_index,
        uma_role="main",
        slot="white",
        white_index=0,
        image=_image(2, 3),
        candidates=[(name, 0.9)],
        current_name=name,
        current_star=1,
    )


def test_recognize_factor_box_updates_white_skill_and_review(monkeypatch) -> None:
    box = DummyBox()
    umas = [UmaFactors(), UmaFactors(), UmaFactors()]
    white_counters = {0: 0, 1: 4, 2: 0}
    img_orig = _image()
    norm_img = img_orig.copy()
    factor_pred = object()
    rank_pred = object()
    ocr = object()
    calls: list[str] = []

    def fake_recognize_factor_candidates(
        factor_pred_arg,
        ocr_arg,
        img_orig_arg,
        text_crop_norm,
        display_crop,
        bbox,
        scale,
        *,
        is_blue_slot,
        is_red_slot,
        green_adoptable,
        green_name_set,
        ext_bbox=None,
    ):
        calls.append("candidates")
        assert factor_pred_arg is factor_pred
        assert ocr_arg is ocr
        assert img_orig_arg is img_orig
        assert text_crop_norm.shape == (14, 20, 3)
        assert display_crop.shape == (18, 48, 3)
        assert bbox == box.bbox
        assert scale == 1.0
        assert is_blue_slot is False
        assert is_red_slot is False
        assert green_adoptable is False
        assert green_name_set == {"green-name"}
        assert ext_bbox is None
        return FactorCandidateRecognition(
            candidates=[("skill-name", 0.8)],
            sources={"skill-name": "onnx"},
            top_name="skill-name",
            ocr_raw="raw text",
            onnx_candidates=[("skill-name", 0.8)],
            ocr_candidates=[],
            template_candidates=[],
        )

    def fake_predict_factor_star(rank_pred_arg, img_orig_arg, box_arg, scale):
        calls.append("star")
        assert rank_pred_arg is rank_pred
        assert img_orig_arg is img_orig
        assert box_arg is box
        assert scale == 1.0
        return 2

    monkeypatch.setattr(
        factor_recognition,
        "recognize_factor_candidates",
        fake_recognize_factor_candidates,
    )
    monkeypatch.setattr(
        factor_recognition,
        "predict_factor_star",
        fake_predict_factor_star,
    )

    result = recognize_factor_box(
        umas,
        white_counters,
        factor_pred,
        rank_pred,
        ocr,
        img_orig,
        norm_img,
        box,
        [box],
        1.0,
        {"green-name"},
        {},
        {},
    )

    assert calls == ["candidates", "star"]
    assert result.green_adoptable is False
    assert result.slot_flags.is_blue is False
    assert result.slot_flags.is_red is False
    assert result.slot_flags.is_green is False
    assert white_counters[1] == 5
    assert [(s.color, s.name, s.star) for s in umas[1].skills] == [
        ("white", "skill-name", 2)
    ]
    assert result.review_item.uma_index == 1
    assert result.review_item.slot == "white"
    assert result.review_item.white_index == 4
    assert result.review_item.candidates == [("skill-name", 0.8)]
    assert result.review_item.candidate_sources == {"skill-name": "onnx"}
    assert result.review_item.ocr_raw == "raw text"
    assert result.review_item.current_name == "skill-name"
    assert result.review_item.current_star == 2


def test_run_factor_recognition_orders_image_level_steps(monkeypatch) -> None:
    img_orig = _image()
    norm_img = img_orig.copy()
    sections = [DummySection(0), DummySection(1), DummySection(2)]
    boxes = [DummyBox(uma_index=0), DummyBox(uma_index=2)]
    prepared = PreparedFactorImage(
        img_orig=img_orig,
        norm_img=norm_img,
        scale=2.0,
        sections=sections,
        boxes=boxes,
    )
    context = RecognitionContext(
        factor_pred=object(),
        rank_pred=object(),
        char_pred=object(),
        ocr=object(),
        green_name_set={"green-name"},
    )
    calls: list[str] = []

    def fake_recognize_characters(umas, section_arg, norm_arg, char_pred_arg):
        calls.append("characters")
        assert section_arg is sections
        assert norm_arg is norm_img
        assert char_pred_arg is context.char_pred
        umas[0].character = "main"

    def fake_compute_green_prepass(box_arg, img_arg, scale_arg, ocr_arg):
        calls.append("green_prepass")
        assert box_arg is boxes
        assert img_arg is img_orig
        assert scale_arg == 2.0
        assert ocr_arg is context.ocr
        return GreenPrepassResult(
            best_green_box={0: boxes[0]},
            best_green_score={0: 0.8},
            best_green_gold={},
            any_green_gold={2: 3},
        )

    def fake_recognize_factor_box(
        umas,
        white_counters,
        factor_pred,
        rank_pred,
        ocr,
        img_arg,
        norm_arg,
        box,
        boxes_arg,
        scale_arg,
        green_name_set,
        best_green_box,
        best_green_score,
    ):
        calls.append(f"box:{box.uma_index}")
        assert white_counters == {0: 0, 1: 0, 2: 0}
        assert factor_pred is context.factor_pred
        assert rank_pred is context.rank_pred
        assert ocr is context.ocr
        assert img_arg is img_orig
        assert norm_arg is norm_img
        assert boxes_arg is boxes
        assert scale_arg == 2.0
        assert green_name_set == {"green-name"}
        assert best_green_box == {0: boxes[0]}
        assert best_green_score == {0: 0.8}
        return RecognizedFactorBox(
            _review_item(box.uma_index, f"name-{box.uma_index}"),
            SlotFlags(is_blue=False, is_red=False, is_green=False),
            False,
        )

    def fake_apply_missing_green_star_fallbacks(umas, any_green_gold):
        calls.append("green_star_fallback")
        assert any_green_gold == {2: 3}
        umas[2].green_star = any_green_gold[2]

    def fake_apply_unique_skill_character_overrides(umas, unique_skill_to_character):
        calls.append("unique_override")
        assert unique_skill_to_character == {"unique": "character"}
        umas[1].character = "override"

    monkeypatch.setattr(
        factor_recognition,
        "recognize_characters",
        fake_recognize_characters,
    )
    monkeypatch.setattr(
        factor_recognition,
        "compute_green_prepass",
        fake_compute_green_prepass,
    )
    monkeypatch.setattr(
        factor_recognition,
        "recognize_factor_box",
        fake_recognize_factor_box,
    )
    monkeypatch.setattr(
        factor_recognition,
        "apply_missing_green_star_fallbacks",
        fake_apply_missing_green_star_fallbacks,
    )
    monkeypatch.setattr(
        factor_recognition,
        "apply_unique_skill_character_overrides",
        fake_apply_unique_skill_character_overrides,
    )

    result = run_factor_recognition(prepared, context, {"unique": "character"})

    assert calls == [
        "characters",
        "green_prepass",
        "box:0",
        "box:2",
        "green_star_fallback",
        "unique_override",
    ]
    assert [uma.character for uma in result.umas] == ["main", "override", ""]
    assert result.umas[2].green_star == 3
    assert [item.current_name for item in result.review.items] == ["name-0", "name-2"]
