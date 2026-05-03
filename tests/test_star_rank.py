from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umafactor.recognition import star_rank
from umafactor.recognition.constants import PERTURBATIONS_RANK
from umafactor.recognition.star_rank import (
    apply_missing_green_star_fallbacks,
    nearest_green_gold_star,
    predict_factor_star,
    resolve_colored_star,
    resolve_green_star,
)


@dataclass
class DummyPrediction:
    label: str
    confidence: float


class DummyRankPredictor:
    def __init__(self, prediction: DummyPrediction) -> None:
        self.prediction = prediction
        self.calls: list[tuple[tuple[int, ...], list[tuple[int, int]]]] = []

    def predict_with_perturbation(
        self,
        img_hwc_bgr: np.ndarray,
        perturbations: list[tuple[int, int]],
    ) -> DummyPrediction:
        self.calls.append((img_hwc_bgr.shape, perturbations))
        return self.prediction


class FailingRankPredictor:
    def predict_with_perturbation(
        self,
        img_hwc_bgr: np.ndarray,
        perturbations: list[tuple[int, int]],
    ) -> DummyPrediction:
        raise AssertionError("rank predictor should not be called")


@dataclass
class DummyBox:
    uma_index: int = 0
    row_index: int = 0
    col_index: int = 0
    color: str = "white"
    bbox: tuple[int, int, int, int] = (10, 20, 110, 40)
    rank_bbox: tuple[int, int, int, int] | None = None
    gold_star_count: int | None = 0


@dataclass
class DummyUma:
    green_name: str = ""
    green_star: int = 0


def _image(height: int = 80, width: int = 140) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_predict_factor_star_uses_positive_gold_without_rank_model() -> None:
    star = predict_factor_star(
        FailingRankPredictor(),
        _image(),
        DummyBox(gold_star_count=3),
        scale=1.0,
    )

    assert star == 3


def test_predict_factor_star_falls_back_to_rank_model() -> None:
    rank = DummyRankPredictor(DummyPrediction(label="2", confidence=0.8))

    star = predict_factor_star(rank, _image(), DummyBox(gold_star_count=0), scale=1.0)

    assert star == 2
    assert rank.calls == [((16, 32, 3), PERTURBATIONS_RANK)]


def test_predict_factor_star_guarantees_one_for_low_confidence_row_zero() -> None:
    rank = DummyRankPredictor(DummyPrediction(label="0", confidence=0.5))

    star = predict_factor_star(
        rank,
        _image(),
        DummyBox(row_index=0, col_index=1, gold_star_count=0),
        scale=1.0,
    )

    assert star == 1


def test_predict_factor_star_invalid_rank_label_becomes_zero() -> None:
    rank = DummyRankPredictor(DummyPrediction(label="bad", confidence=0.9))

    star = predict_factor_star(
        rank,
        _image(),
        DummyBox(row_index=2, col_index=0, gold_star_count=None),
        scale=1.0,
    )

    assert star == 0


def test_resolve_colored_star_prefers_high_confidence_template(monkeypatch) -> None:
    crop = _image(3, 4)
    crop_calls: list[tuple[tuple[int, int, int, int], float, int]] = []
    match_calls: list[tuple[tuple[int, ...], str]] = []

    def fake_display_crop_from_original(
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
        scale: float,
        pad_y_norm: int = 2,
    ) -> np.ndarray:
        crop_calls.append((bbox, scale, pad_y_norm))
        return crop

    def fake_match_star(img: np.ndarray, color: str) -> list[tuple[int, float]]:
        match_calls.append((img.shape, color))
        return [(3, 0.92)]

    monkeypatch.setattr(
        star_rank,
        "display_crop_from_original",
        fake_display_crop_from_original,
    )
    monkeypatch.setattr(star_rank, "match_star", fake_match_star)

    star = resolve_colored_star(
        _image(),
        bbox=(10, 20, 110, 40),
        scale=2.0,
        color="red",
        fallback_star=1,
    )

    assert star == 3
    assert crop_calls == [((60, 20, 110, 40), 2.0, 2)]
    assert match_calls == [((3, 4, 3), "red")]


def test_resolve_colored_star_uses_fallback_below_template_threshold(monkeypatch) -> None:
    monkeypatch.setattr(star_rank, "match_star", lambda _img, _color: [(3, 0.91)])

    star = resolve_colored_star(
        _image(),
        bbox=(10, 20, 110, 40),
        scale=1.0,
        color="blue",
        fallback_star=2,
    )

    assert star == 2


def test_nearest_green_gold_star_uses_same_uma_nearest_row() -> None:
    box = DummyBox(uma_index=0, row_index=3, color="green", gold_star_count=0)
    candidates = [
        DummyBox(uma_index=0, row_index=1, color="green", gold_star_count=2),
        DummyBox(uma_index=0, row_index=4, color="green", gold_star_count=1),
        DummyBox(uma_index=1, row_index=3, color="green", gold_star_count=3),
        DummyBox(uma_index=0, row_index=2, color="white", gold_star_count=3),
    ]

    assert nearest_green_gold_star(box, candidates) == 1


def test_resolve_green_star_prefers_high_confidence_template(monkeypatch) -> None:
    monkeypatch.setattr(star_rank, "match_green_star", lambda _img: [(2, 0.92)])

    star = resolve_green_star(
        _image(),
        DummyBox(color="green", gold_star_count=3),
        [],
        scale=1.0,
        fallback_star=1,
    )

    assert star == 2


def test_resolve_green_star_uses_own_gold_before_nearest_and_rank(monkeypatch) -> None:
    monkeypatch.setattr(star_rank, "match_green_star", lambda _img: [])

    star = resolve_green_star(
        _image(),
        DummyBox(color="green", gold_star_count=3),
        [DummyBox(color="green", row_index=1, gold_star_count=2)],
        scale=1.0,
        fallback_star=1,
    )

    assert star == 3


def test_resolve_green_star_uses_nearest_gold_before_rank(monkeypatch) -> None:
    monkeypatch.setattr(star_rank, "match_green_star", lambda _img: [])
    box = DummyBox(uma_index=0, row_index=3, color="green", gold_star_count=0)

    star = resolve_green_star(
        _image(),
        box,
        [DummyBox(uma_index=0, row_index=4, color="green", gold_star_count=2)],
        scale=1.0,
        fallback_star=1,
    )

    assert star == 2


def test_resolve_green_star_uses_rank_then_minimum_one(monkeypatch) -> None:
    monkeypatch.setattr(star_rank, "match_green_star", lambda _img: [])
    box = DummyBox(color="green", gold_star_count=0)

    assert resolve_green_star(_image(), box, [], scale=1.0, fallback_star=2) == 2
    assert resolve_green_star(_image(), box, [], scale=1.0, fallback_star=0) == 1


def test_apply_missing_green_star_fallbacks_only_fills_empty_green_name() -> None:
    umas = [
        DummyUma(green_name="", green_star=0),
        DummyUma(green_name="known", green_star=0),
        DummyUma(green_name="", green_star=1),
    ]

    apply_missing_green_star_fallbacks(umas, {0: 3, 1: 2, 2: 2})

    assert [uma.green_star for uma in umas] == [3, 0, 1]
