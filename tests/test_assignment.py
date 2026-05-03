from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umafactor.recognition import assignment
from umafactor.recognition.assignment import (
    AssignmentResult,
    apply_factor_result,
    build_review_item,
)
from umafactor.recognition.constants import BLUE_FACTOR_TYPES, RED_FACTOR_TYPES
from umafactor.schema import UmaFactors


@dataclass
class DummyBox:
    uma_index: int = 0
    row_index: int = 0
    col_index: int = 0
    color: str = "white"
    bbox: tuple[int, int, int, int] = (10, 20, 110, 40)
    rank_bbox: tuple[int, int, int, int] | None = None
    gold_star_count: int | None = 0


def _image(height: int = 40, width: int = 60) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_apply_factor_result_assigns_blue_with_template_star(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, int, int, int], float, str, int]] = []

    def fake_resolve_colored_star(
        img_orig: np.ndarray,
        bbox: tuple[int, int, int, int],
        scale: float,
        color: str,
        fallback_star: int,
    ) -> int:
        calls.append(("colored", bbox, scale, color, fallback_star))
        return 3

    monkeypatch.setattr(assignment, "resolve_colored_star", fake_resolve_colored_star)
    uma = UmaFactors()
    box = DummyBox()

    result = apply_factor_result(
        uma,
        {0: 0},
        _image(),
        box,
        [box],
        2.0,
        BLUE_FACTOR_TYPES[0],
        1,
        is_blue_slot=True,
        is_red_slot=False,
        green_adoptable=False,
    )

    assert result == AssignmentResult(slot_kind="blue", white_index=0)
    assert uma.blue_type == BLUE_FACTOR_TYPES[0]
    assert uma.blue_star == 3
    assert uma.skills == []
    assert calls == [("colored", box.bbox, 2.0, "blue", 1)]


def test_apply_factor_result_assigns_red_with_template_star(monkeypatch) -> None:
    monkeypatch.setattr(
        assignment,
        "resolve_colored_star",
        lambda _img, _bbox, _scale, _color, _fallback: 2,
    )
    uma = UmaFactors()
    box = DummyBox()

    result = apply_factor_result(
        uma,
        {0: 0},
        _image(),
        box,
        [box],
        1.0,
        RED_FACTOR_TYPES[0],
        1,
        is_blue_slot=False,
        is_red_slot=True,
        green_adoptable=False,
    )

    assert result == AssignmentResult(slot_kind="red", white_index=0)
    assert uma.red_type == RED_FACTOR_TYPES[0]
    assert uma.red_star == 2
    assert uma.skills == []


def test_apply_factor_result_assigns_green(monkeypatch) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_resolve_green_star(
        img_orig: np.ndarray,
        box: DummyBox,
        boxes: list[DummyBox],
        scale: float,
        fallback_star: int,
    ) -> int:
        calls.append(("green", box.color, fallback_star))
        return 1

    monkeypatch.setattr(assignment, "resolve_green_star", fake_resolve_green_star)
    uma = UmaFactors()
    box = DummyBox(color="green")

    result = apply_factor_result(
        uma,
        {0: 0},
        _image(),
        box,
        [box],
        1.0,
        "green-name",
        0,
        is_blue_slot=False,
        is_red_slot=False,
        green_adoptable=True,
    )

    assert result == AssignmentResult(slot_kind="green", white_index=0)
    assert uma.green_name == "green-name"
    assert uma.green_star == 1
    assert uma.skills == []
    assert calls == [("green", "green", 0)]


def test_apply_factor_result_falls_back_to_white_and_increments_counter() -> None:
    uma = UmaFactors()
    box = DummyBox(uma_index=1, color="white")
    white_counters = {0: 0, 1: 4, 2: 0}

    result = apply_factor_result(
        uma,
        white_counters,
        _image(),
        box,
        [box],
        1.0,
        "skill-name",
        2,
        is_blue_slot=False,
        is_red_slot=False,
        green_adoptable=False,
    )

    assert result == AssignmentResult(slot_kind="white", white_index=4)
    assert white_counters[1] == 5
    assert [(s.color, s.name, s.star) for s in uma.skills] == [
        ("white", "skill-name", 2)
    ]


def test_apply_factor_result_falls_back_to_white_when_blue_already_filled() -> None:
    uma = UmaFactors(blue_type=BLUE_FACTOR_TYPES[0], blue_star=1)
    box = DummyBox(color="blue")
    white_counters = {0: 0, 1: 0, 2: 0}

    result = apply_factor_result(
        uma,
        white_counters,
        _image(),
        box,
        [box],
        1.0,
        BLUE_FACTOR_TYPES[1],
        2,
        is_blue_slot=True,
        is_red_slot=False,
        green_adoptable=False,
    )

    assert result == AssignmentResult(slot_kind="white", white_index=0)
    assert uma.blue_type == BLUE_FACTOR_TYPES[0]
    assert [(s.color, s.name, s.star) for s in uma.skills] == [
        ("blue", BLUE_FACTOR_TYPES[1], 2)
    ]


def test_build_review_item_copies_image_and_sets_role() -> None:
    box = DummyBox(uma_index=2)
    image = _image(5, 7)
    image[0, 0, 0] = 255
    candidates = [("A", 0.9)]
    sources = {"A": "onnx"}

    item = build_review_item(
        box,
        AssignmentResult(slot_kind="white", white_index=3),
        image,
        candidates,
        sources,
        "raw",
        "A",
        2,
    )
    image[0, 0, 0] = 0

    assert item.uma_index == 2
    assert item.uma_role == "parent2"
    assert item.slot == "white"
    assert item.white_index == 3
    assert item.candidates is candidates
    assert item.candidate_sources is sources
    assert item.ocr_raw == "raw"
    assert item.current_name == "A"
    assert item.current_star == 2
    assert item.image[0, 0, 0] == 255
