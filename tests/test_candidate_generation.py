from __future__ import annotations

import numpy as np

from umafactor.recognition import candidate_generation
from umafactor.recognition.candidate_generation import (
    filter_slot_candidates,
    match_template_candidates,
    predict_onnx_candidates,
    recognize_ocr_candidates,
)
from umafactor.recognition.constants import (
    BLUE_FACTOR_TYPES,
    PERTURBATIONS_BLUE,
    PERTURBATIONS_RED,
    RED_FACTOR_TYPES,
)


class DummyFactorPredictor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def topk_in_category(
        self,
        crops: list[np.ndarray],
        category_names: list[str],
        k: int,
        use_multi_interp: bool = False,
    ) -> list[tuple[str, float]]:
        self.calls.append(
            {
                "method": "topk_in_category",
                "crops": crops,
                "category_names": category_names,
                "k": k,
                "use_multi_interp": use_multi_interp,
            }
        )
        return [("category", 0.9)]

    def topk_ensemble(self, crops: list[np.ndarray], k: int) -> list[tuple[str, float]]:
        self.calls.append({"method": "topk_ensemble", "crops": crops, "k": k})
        return [("ensemble", 0.8)]


class DummyOCR:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def recognize_red(self, img: np.ndarray) -> str:
        self.calls.append(("recognize_red", img.shape))
        return "red raw"

    def recognize_blue(self, img: np.ndarray) -> str:
        self.calls.append(("recognize_blue", img.shape))
        return "blue raw"

    def recognize(self, img: np.ndarray) -> str:
        self.calls.append(("recognize", img.shape))
        return "plain raw"

    def recognize_with_parts(self, img: np.ndarray) -> tuple[str, list[str]]:
        self.calls.append(("recognize_with_parts", img.shape))
        return "green raw", ["green", "raw"]

    def match_to_green_factor_multi(
        self, text: str, fragments: list[str], top_k: int = 5
    ) -> list[tuple[str, float]]:
        self.calls.append(("match_to_green_factor_multi", text, fragments, top_k))
        return [("green", 0.7)]

    def match_to_factor(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        self.calls.append(("match_to_factor", text, top_k))
        return [("factor", 0.6)]


def _image(height: int = 40, width: int = 60) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_predict_onnx_candidates_uses_blue_category_perturbations() -> None:
    pred = DummyFactorPredictor()
    text_crop_norm = _image(4, 8)

    result = predict_onnx_candidates(
        pred,
        _image(),
        text_crop_norm,
        bbox=(4, 4, 20, 14),
        ext_bbox=(4, 4, 20, 14),
        scale=1.0,
        is_blue_slot=True,
        is_red_slot=False,
    )

    assert result == [("category", 0.9)]
    call = pred.calls[0]
    assert call["method"] == "topk_in_category"
    assert call["category_names"] == BLUE_FACTOR_TYPES
    assert call["k"] == 5
    assert call["use_multi_interp"] is False
    assert len(call["crops"]) == len(PERTURBATIONS_BLUE) + 1
    assert call["crops"][-1] is text_crop_norm


def test_predict_onnx_candidates_uses_red_multi_interp() -> None:
    pred = DummyFactorPredictor()
    text_crop_norm = _image(4, 8)

    predict_onnx_candidates(
        pred,
        _image(),
        text_crop_norm,
        bbox=(4, 4, 20, 14),
        ext_bbox=(4, 4, 20, 14),
        scale=1.0,
        is_blue_slot=False,
        is_red_slot=True,
    )

    call = pred.calls[0]
    assert call["method"] == "topk_in_category"
    assert call["category_names"] == RED_FACTOR_TYPES
    assert call["use_multi_interp"] is True
    assert len(call["crops"]) == len(PERTURBATIONS_RED) + 1
    assert call["crops"][-1] is text_crop_norm


def test_predict_onnx_candidates_uses_ensemble_for_non_color_slot() -> None:
    pred = DummyFactorPredictor()
    text_crop_norm = _image(4, 8)

    result = predict_onnx_candidates(
        pred,
        _image(),
        text_crop_norm,
        bbox=(4, 4, 20, 14),
        ext_bbox=(4, 4, 20, 14),
        scale=1.0,
        is_blue_slot=False,
        is_red_slot=False,
    )

    assert result == [("ensemble", 0.8)]
    call = pred.calls[0]
    assert call["method"] == "topk_ensemble"
    assert call["k"] == 5
    assert len(call["crops"]) == 2
    assert call["crops"][1] is text_crop_norm


def test_recognize_ocr_candidates_returns_empty_without_ocr() -> None:
    ocr_raw, candidates = recognize_ocr_candidates(
        None,
        _image(),
        is_blue_slot=False,
        is_red_slot=False,
        green_adoptable=False,
    )

    assert ocr_raw == ""
    assert candidates == []


def test_recognize_ocr_candidates_uses_blue_recognizer_then_factor_match() -> None:
    ocr = DummyOCR()

    ocr_raw, candidates = recognize_ocr_candidates(
        ocr,
        _image(5, 7),
        is_blue_slot=True,
        is_red_slot=False,
        green_adoptable=False,
    )

    assert ocr_raw == "blue raw"
    assert candidates == [("factor", 0.6)]
    assert ocr.calls == [
        ("recognize_blue", (5, 7, 3)),
        ("match_to_factor", "blue raw", 5),
    ]


def test_recognize_ocr_candidates_uses_green_fragments() -> None:
    ocr = DummyOCR()

    ocr_raw, candidates = recognize_ocr_candidates(
        ocr,
        _image(5, 7),
        is_blue_slot=False,
        is_red_slot=False,
        green_adoptable=True,
    )

    assert ocr_raw == "green raw"
    assert candidates == [("green", 0.7)]
    assert ocr.calls == [
        ("recognize_with_parts", (5, 7, 3)),
        ("match_to_green_factor_multi", "green raw", ["green", "raw"], 5),
    ]


def test_recognize_ocr_candidates_preserves_slot_priority_before_green_match() -> None:
    ocr = DummyOCR()

    ocr_raw, candidates = recognize_ocr_candidates(
        ocr,
        _image(5, 7),
        is_blue_slot=False,
        is_red_slot=True,
        green_adoptable=True,
    )

    assert ocr_raw == "red raw"
    assert candidates == [("green", 0.7)]
    assert ocr.calls == [
        ("recognize_red", (5, 7, 3)),
        ("match_to_green_factor_multi", "red raw", [], 5),
    ]


def test_filter_slot_candidates_limits_blue_ocr_candidates() -> None:
    onnx_candidates, ocr_candidates = filter_slot_candidates(
        [("any", 0.8)],
        [(BLUE_FACTOR_TYPES[0], 0.7), ("not-blue", 0.9)],
        is_blue_slot=True,
        is_red_slot=False,
        green_adoptable=False,
        green_name_set=set(),
    )

    assert onnx_candidates == [("any", 0.8)]
    assert ocr_candidates == [(BLUE_FACTOR_TYPES[0], 0.7)]


def test_filter_slot_candidates_removes_green_names_from_non_green_onnx() -> None:
    onnx_candidates, ocr_candidates = filter_slot_candidates(
        [("green-name", 0.8), ("plain", 0.7)],
        [("ocr", 0.6)],
        is_blue_slot=False,
        is_red_slot=False,
        green_adoptable=False,
        green_name_set={"green-name"},
    )

    assert onnx_candidates == [("plain", 0.7)]
    assert ocr_candidates == [("ocr", 0.6)]


def test_match_template_candidates_uses_red_category(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_match_templates(img: np.ndarray, category: str) -> list[tuple[str, float]]:
        calls.append((category, img.shape))
        return [(f"{category}-{i}", 1.0 - i * 0.01) for i in range(6)]

    monkeypatch.setattr(candidate_generation, "match_templates", fake_match_templates)

    candidates = match_template_candidates(
        _image(5, 7),
        _image(),
        bbox=(4, 4, 20, 14),
        scale=1.0,
        is_red_slot=True,
        is_blue_slot=False,
        green_adoptable=False,
    )

    assert candidates == [(f"red-{i}", 1.0 - i * 0.01) for i in range(5)]
    assert calls == [("red", (5, 7, 3))]


def test_match_template_candidates_crops_green_name_region(monkeypatch) -> None:
    crop = _image(3, 4)
    crop_calls: list[tuple[tuple[int, int, int, int], float, int]] = []

    def fake_display_crop_from_original(
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
        scale: float,
        pad_y_norm: int = 2,
    ) -> np.ndarray:
        crop_calls.append((bbox, scale, pad_y_norm))
        return crop

    def fake_match_green_name(img: np.ndarray) -> list[tuple[str, float]]:
        assert img is crop
        return [(f"green-{i}", 1.0 - i * 0.01) for i in range(6)]

    monkeypatch.setattr(
        candidate_generation,
        "display_crop_from_original",
        fake_display_crop_from_original,
    )
    monkeypatch.setattr(candidate_generation, "match_green_name", fake_match_green_name)

    candidates = match_template_candidates(
        _image(5, 7),
        _image(),
        bbox=(10, 20, 110, 40),
        scale=2.0,
        is_red_slot=False,
        is_blue_slot=False,
        green_adoptable=True,
    )

    assert candidates == [(f"green-{i}", 1.0 - i * 0.01) for i in range(5)]
    assert crop_calls == [((10, 20, 95, 40), 2.0, 2)]
