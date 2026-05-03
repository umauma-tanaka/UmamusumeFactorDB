from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umafactor.recognition import characters
from umafactor.recognition.characters import (
    apply_unique_skill_character_overrides,
    recognize_characters,
)
from umafactor.schema import UmaFactors


@dataclass
class DummySection:
    uma_index: int
    portrait_bbox: tuple[int, int, int, int] = (0, 0, 10, 10)


@dataclass
class DummyPrediction:
    label: str


class DummyCharacterPredictor:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.calls: list[tuple[int, ...]] = []

    def predict(self, img_hwc_bgr: np.ndarray) -> DummyPrediction:
        self.calls.append(img_hwc_bgr.shape)
        return DummyPrediction(self.labels.pop(0))


def _image(height: int = 80, width: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_recognize_characters_sets_uma_character_by_section_index(monkeypatch) -> None:
    crop = _image(32, 32)
    crop_calls: list[DummySection] = []

    def fake_extract_character_icon_bgr(
        img: np.ndarray, section: DummySection
    ) -> np.ndarray:
        crop_calls.append(section)
        return crop

    monkeypatch.setattr(
        characters,
        "extract_character_icon_bgr",
        fake_extract_character_icon_bgr,
    )
    umas = [UmaFactors(), UmaFactors(), UmaFactors()]
    sections = [DummySection(2), DummySection(0), DummySection(1)]
    predictor = DummyCharacterPredictor(["parent2", "main", "parent1"])

    recognize_characters(umas, sections, _image(), predictor)

    assert [uma.character for uma in umas] == ["main", "parent1", "parent2"]
    assert crop_calls == sections
    assert predictor.calls == [(32, 32, 3), (32, 32, 3), (32, 32, 3)]


def test_apply_unique_skill_character_overrides_uses_exact_name() -> None:
    umas = [
        UmaFactors(character="onnx", green_name="unique-a"),
        UmaFactors(character="keep", green_name=""),
    ]

    apply_unique_skill_character_overrides(umas, {"unique-a": "mapped"})

    assert [uma.character for uma in umas] == ["mapped", "keep"]


def test_apply_unique_skill_character_overrides_uses_stripped_name_without_mutating() -> None:
    uma = UmaFactors(character="onnx", green_name=" unique-b ")

    apply_unique_skill_character_overrides([uma], {"unique-b": "mapped"})

    assert uma.character == "mapped"
    assert uma.green_name == " unique-b "


def test_apply_unique_skill_character_overrides_leaves_unmatched_values() -> None:
    umas = [UmaFactors(character="onnx", green_name="unknown")]

    apply_unique_skill_character_overrides(umas, {"other": "mapped"})

    assert umas[0].character == "onnx"
