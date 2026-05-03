"""Shared constants for factor recognition."""

from __future__ import annotations

BLUE_FACTOR_TYPES = ["スピード", "スタミナ", "パワー", "根性", "賢さ"]
RED_FACTOR_TYPES = [
    "芝", "ダート",
    "短距離", "マイル", "中距離", "長距離",
    "逃げ", "先行", "差し", "追込",
]

PERTURBATIONS_BLUE: list[tuple[int, int]] = [
    (dy, dx) for dy in range(-2, 3) for dx in range(-1, 2)
]
PERTURBATIONS_RED: list[tuple[int, int]] = [
    (dy, dx) for dy in range(-5, 6) for dx in range(-3, 4)
]
PERTURBATIONS_RANK: list[tuple[int, int]] = [
    (dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)
]

UMA_ROLES = ["main", "parent1", "parent2"]

