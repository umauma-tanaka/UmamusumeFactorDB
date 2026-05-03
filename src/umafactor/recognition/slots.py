"""Slot classification helpers for factor boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class FactorBoxLike(Protocol):
    uma_index: int
    row_index: int
    col_index: int
    color: str
    gold_star_count: int | None


@dataclass(frozen=True)
class SlotFlags:
    is_blue: bool
    is_red: bool
    is_green: bool


def is_green_candidate_box(box: FactorBoxLike) -> bool:
    """Return whether the box should be considered in the green pre-pass."""
    return box.color == "green" or (box.row_index == 1 and box.col_index == 0)


def classify_factor_slot(box: FactorBoxLike) -> SlotFlags:
    """Classify a factor box into positional blue/red/green slots."""
    if box.row_index == 0 and box.col_index == 0:
        return SlotFlags(is_blue=True, is_red=False, is_green=False)
    if box.row_index == 0 and box.col_index == 1:
        return SlotFlags(is_blue=False, is_red=True, is_green=False)
    if box.row_index == 1 and box.col_index == 0:
        return SlotFlags(is_blue=False, is_red=False, is_green=True)
    return SlotFlags(
        is_blue=box.color == "blue",
        is_red=box.color == "red",
        is_green=box.color == "green" and box.col_index == 0,
    )


def should_adopt_green_box(
    box: FactorBoxLike,
    boxes: Sequence[FactorBoxLike],
    *,
    is_green_slot: bool,
    current_green_name: str,
    best_green_box: FactorBoxLike | None,
    best_green_score: float,
) -> bool:
    """Return whether this box should be treated as the adopted green slot."""
    if not is_green_slot or current_green_name:
        return False

    if best_green_box is not None and best_green_score >= 0.5:
        return box is best_green_box

    same_uma_green_others = any(
        b for b in boxes
        if b.uma_index == box.uma_index
        and b.color == "green"
        and b.col_index == 0
        and not (b.row_index == 1 and b.col_index == 0)
    )
    pos_absolute = (
        box.row_index == 1
        and box.col_index == 0
        and not same_uma_green_others
    )
    if pos_absolute:
        return True
    return box.gold_star_count is None or box.gold_star_count > 0

