from __future__ import annotations

from dataclasses import dataclass

from umafactor.recognition.slots import (
    SlotFlags,
    classify_factor_slot,
    is_green_candidate_box,
    should_adopt_green_box,
)


@dataclass
class DummyBox:
    uma_index: int = 0
    row_index: int = 0
    col_index: int = 0
    color: str = "white"
    gold_star_count: int | None = 0


def test_is_green_candidate_accepts_green_color_or_absolute_position() -> None:
    assert is_green_candidate_box(DummyBox(row_index=2, col_index=0, color="green"))
    assert is_green_candidate_box(DummyBox(row_index=1, col_index=0, color="white"))
    assert not is_green_candidate_box(DummyBox(row_index=1, col_index=1, color="white"))


def test_classify_factor_slot_uses_absolute_positions_first() -> None:
    assert classify_factor_slot(DummyBox(row_index=0, col_index=0, color="red")) == SlotFlags(
        is_blue=True,
        is_red=False,
        is_green=False,
    )
    assert classify_factor_slot(DummyBox(row_index=0, col_index=1, color="blue")) == SlotFlags(
        is_blue=False,
        is_red=True,
        is_green=False,
    )
    assert classify_factor_slot(DummyBox(row_index=1, col_index=0, color="red")) == SlotFlags(
        is_blue=False,
        is_red=False,
        is_green=True,
    )


def test_classify_factor_slot_falls_back_to_color() -> None:
    assert classify_factor_slot(DummyBox(row_index=2, col_index=0, color="blue")) == SlotFlags(
        is_blue=True,
        is_red=False,
        is_green=False,
    )
    assert classify_factor_slot(DummyBox(row_index=2, col_index=0, color="red")) == SlotFlags(
        is_blue=False,
        is_red=True,
        is_green=False,
    )
    assert classify_factor_slot(DummyBox(row_index=2, col_index=0, color="green")) == SlotFlags(
        is_blue=False,
        is_red=False,
        is_green=True,
    )
    assert classify_factor_slot(DummyBox(row_index=2, col_index=1, color="green")) == SlotFlags(
        is_blue=False,
        is_red=False,
        is_green=False,
    )


def test_should_adopt_green_box_rejects_non_green_or_already_filled() -> None:
    box = DummyBox(row_index=1, col_index=0, color="green", gold_star_count=3)

    assert not should_adopt_green_box(
        box,
        [box],
        is_green_slot=False,
        current_green_name="",
        best_green_box=None,
        best_green_score=0.0,
    )
    assert not should_adopt_green_box(
        box,
        [box],
        is_green_slot=True,
        current_green_name="existing",
        best_green_box=None,
        best_green_score=0.0,
    )


def test_should_adopt_green_box_uses_best_box_identity_when_confident() -> None:
    best = DummyBox(row_index=2, col_index=0, color="green", gold_star_count=1)
    other = DummyBox(row_index=1, col_index=0, color="green", gold_star_count=3)

    assert should_adopt_green_box(
        best,
        [best, other],
        is_green_slot=True,
        current_green_name="",
        best_green_box=best,
        best_green_score=0.5,
    )
    assert not should_adopt_green_box(
        other,
        [best, other],
        is_green_slot=True,
        current_green_name="",
        best_green_box=best,
        best_green_score=0.5,
    )


def test_should_adopt_green_box_accepts_absolute_position_when_no_other_green() -> None:
    box = DummyBox(row_index=1, col_index=0, color="white", gold_star_count=0)

    assert should_adopt_green_box(
        box,
        [box],
        is_green_slot=True,
        current_green_name="",
        best_green_box=None,
        best_green_score=0.0,
    )


def test_should_adopt_green_box_falls_back_to_star_condition_when_other_green_exists() -> None:
    box = DummyBox(row_index=1, col_index=0, color="white", gold_star_count=0)
    other_green = DummyBox(row_index=2, col_index=0, color="green", gold_star_count=1)
    empty_star = DummyBox(row_index=2, col_index=0, color="green", gold_star_count=0)
    unknown_star = DummyBox(row_index=2, col_index=0, color="green", gold_star_count=None)

    assert not should_adopt_green_box(
        box,
        [box, other_green],
        is_green_slot=True,
        current_green_name="",
        best_green_box=None,
        best_green_score=0.0,
    )
    assert not should_adopt_green_box(
        empty_star,
        [box, other_green],
        is_green_slot=True,
        current_green_name="",
        best_green_box=None,
        best_green_score=0.0,
    )
    assert should_adopt_green_box(
        unknown_star,
        [box, other_green],
        is_green_slot=True,
        current_green_name="",
        best_green_box=None,
        best_green_score=0.0,
    )
    assert should_adopt_green_box(
        other_green,
        [box, other_green],
        is_green_slot=True,
        current_green_name="",
        best_green_box=None,
        best_green_score=0.0,
    )

