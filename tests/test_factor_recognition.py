from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umafactor.recognition import factor_recognition
from umafactor.recognition.candidate_generation import FactorCandidateRecognition
from umafactor.recognition.factor_recognition import recognize_factor_box
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


def _image(height: int = 80, width: int = 100) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


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
