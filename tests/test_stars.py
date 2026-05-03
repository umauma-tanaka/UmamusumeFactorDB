from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from umafactor.recognition import stars


def test_prep_star_slot_resizes_to_expected_size() -> None:
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    resized = stars._prep_star_slot(image)

    assert resized.shape == (stars.STAR_SLOT_SIZE, stars.STAR_SLOT_SIZE, 3)
    assert resized.dtype == np.uint8


def test_predict_stars_batch_handles_empty_input() -> None:
    assert stars.predict_stars_batch([]) == []


def test_predict_stars_batch_uses_hsv_fallback_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []
    images = [
        np.zeros((stars.STAR_SLOT_SIZE, stars.STAR_SLOT_SIZE, 3), dtype=np.uint8),
        np.ones((stars.STAR_SLOT_SIZE, stars.STAR_SLOT_SIZE, 3), dtype=np.uint8),
    ]

    def fake_predict_star_hsv(image: np.ndarray) -> tuple[str, float]:
        calls.append(image)
        return "empty", 0.75

    monkeypatch.setattr(stars, "_star_model_path", lambda: Path("missing.onnx"))
    monkeypatch.setattr(stars, "_allow_missing_star_classifier", lambda: True)
    monkeypatch.setattr(stars, "_predict_star_hsv", fake_predict_star_hsv)

    assert stars.predict_stars_batch(images) == [("empty", 0.75), ("empty", 0.75)]
    assert calls == images


def test_predict_star_uses_session_when_model_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExistingPath:
        def exists(self) -> bool:
            return True

    class DummySession:
        def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert output_names == ["index", "confidence"]
            assert inputs["images"].shape == (1, stars.STAR_SLOT_SIZE, stars.STAR_SLOT_SIZE, 3)
            return [np.array([1], dtype=np.int64), np.array([0.9], dtype=np.float32)]

    monkeypatch.setattr(stars, "_star_model_path", lambda: ExistingPath())
    monkeypatch.setattr(stars, "_get_star_session", lambda: DummySession())

    assert stars.predict_star(np.zeros((12, 10, 3), dtype=np.uint8)) == (
        "gold",
        pytest.approx(0.9),
    )
