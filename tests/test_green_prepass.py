from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umafactor.recognition import green_prepass
from umafactor.recognition.green_prepass import compute_green_prepass


@dataclass
class DummyBox:
    uma_index: int = 0
    row_index: int = 0
    col_index: int = 0
    color: str = "white"
    bbox: tuple[int, int, int, int] = (10, 20, 110, 40)
    gold_star_count: int | None = 0


class DummyOCR:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple] = []

    def recognize_with_parts(self, img: np.ndarray) -> tuple[str, list[str]]:
        self.calls.append(("recognize_with_parts", img.shape))
        return "raw", ["ra", "w"]

    def match_to_green_factor_multi(
        self, text: str, fragments: list[str], top_k: int = 1
    ) -> list[tuple[str, float]]:
        self.calls.append(("match_to_green_factor_multi", text, fragments, top_k))
        if not self.scores:
            return []
        return [("green-name", self.scores.pop(0))]


def _image(height: int = 80, width: int = 140) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_compute_green_prepass_tracks_any_gold_for_candidate_boxes() -> None:
    boxes = [
        DummyBox(uma_index=0, row_index=2, col_index=1, color="green", gold_star_count=3),
        DummyBox(uma_index=0, row_index=2, col_index=1, color="white", gold_star_count=5),
    ]

    result = compute_green_prepass(boxes, _image(), scale=1.0, ocr=None)

    assert result.any_green_gold == {0: 3}
    assert result.best_green_box == {}
    assert result.best_green_score == {}
    assert result.best_green_gold == {}


def test_compute_green_prepass_selects_best_col0_candidate(monkeypatch) -> None:
    boxes = [
        DummyBox(uma_index=0, row_index=1, col_index=0, color="white", gold_star_count=1),
        DummyBox(uma_index=0, row_index=2, col_index=0, color="green", gold_star_count=2),
    ]
    template_scores = [0.3, 0.8]

    def fake_match_green_name(img: np.ndarray) -> list[tuple[str, float]]:
        return [("template", template_scores.pop(0))]

    monkeypatch.setattr(green_prepass, "match_green_name", fake_match_green_name)

    result = compute_green_prepass(
        boxes,
        _image(),
        scale=1.0,
        ocr=DummyOCR([0.4, 0.1]),
    )

    assert result.best_green_box == {0: boxes[1]}
    assert result.best_green_score == {0: 0.8}
    assert result.best_green_gold == {0: 2}
    assert result.any_green_gold == {0: 2}


def test_compute_green_prepass_uses_left_85_percent_name_crop(monkeypatch) -> None:
    box = DummyBox(row_index=1, col_index=0, color="white", bbox=(10, 20, 110, 40))
    crop_calls: list[tuple[tuple[int, int, int, int], float, int]] = []

    def fake_display_crop_from_original(
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
        scale: float,
        pad_y_norm: int = 2,
    ) -> np.ndarray:
        crop_calls.append((bbox, scale, pad_y_norm))
        return _image(5, 7)

    monkeypatch.setattr(
        green_prepass,
        "display_crop_from_original",
        fake_display_crop_from_original,
    )
    monkeypatch.setattr(green_prepass, "match_green_name", lambda _img: [])

    result = compute_green_prepass([box], _image(), scale=2.0, ocr=None)

    assert crop_calls == [
        ((10, 20, 110, 40), 2.0, 2),
        ((10, 20, 95, 40), 2.0, 2),
    ]
    assert result.best_green_box == {}
    assert result.best_green_score == {}


def test_compute_green_prepass_calls_green_ocr_only_for_col0_candidates(monkeypatch) -> None:
    col0 = DummyBox(uma_index=0, row_index=1, col_index=0, color="white")
    col1 = DummyBox(uma_index=0, row_index=2, col_index=1, color="green")
    ocr = DummyOCR([0.7])

    monkeypatch.setattr(green_prepass, "match_green_name", lambda _img: [])

    compute_green_prepass([col0, col1], _image(), scale=1.0, ocr=ocr)

    assert ocr.calls == [
        ("recognize_with_parts", (24, 118, 3)),
        ("match_to_green_factor_multi", "raw", ["ra", "w"], 1),
    ]
