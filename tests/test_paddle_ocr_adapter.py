from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from umafactor.recognition import paddle_ocr_adapter
from umafactor.recognition.paddle_ocr_adapter import PaddleFactorOCR


def test_require_paddlepaddle_dependency_reports_install_hint(monkeypatch) -> None:
    monkeypatch.setattr(paddle_ocr_adapter, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match="pip install paddlepaddle"):
        paddle_ocr_adapter._require_paddlepaddle_dependency()


def test_require_paddlepaddle_dependency_accepts_paddle_module(monkeypatch) -> None:
    monkeypatch.setattr(paddle_ocr_adapter, "find_spec", lambda name: object())

    paddle_ocr_adapter._require_paddlepaddle_dependency()


def test_paddle_factor_ocr_uses_preserve_scale_detection_defaults(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = _FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    monkeypatch.setattr(
        paddle_ocr_adapter,
        "find_spec",
        lambda name: object() if name == "paddle" else None,
    )

    PaddleFactorOCR(mode="ocr")

    assert captured_kwargs["text_det_limit_side_len"] == (
        paddle_ocr_adapter.DEFAULT_TEXT_DET_LIMIT_SIDE_LEN
    )
    assert captured_kwargs["text_det_limit_type"] == paddle_ocr_adapter.DEFAULT_TEXT_DET_LIMIT_TYPE


def test_paddle_factor_ocr_recognize_many_uses_engine_batch_predict() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, images):
            self.calls += 1
            assert len(images) == 2
            return [{"res": {"rec_texts": ["one"]}}, {"res": {"rec_texts": ["two"]}}]

    engine = _Engine()
    ocr = PaddleFactorOCR.__new__(PaddleFactorOCR)
    ocr._engine = engine
    ocr.mode = "ocr"

    names = ocr.recognize_many(
        [
            np.zeros((10, 20, 3), dtype=np.uint8),
            np.zeros((10, 20, 3), dtype=np.uint8),
        ]
    )

    assert names == ["one", "two"]
    assert engine.calls == 1


def test_paddle_factor_ocr_recognize_canvas_assigns_texts_by_bbox() -> None:
    class _Engine:
        def predict(self, image):
            assert image.shape[:2] == (80, 40)
            return [
                {
                    "res": {
                        "rec_texts": ["upper", "lower"],
                        "rec_boxes": [[5, 5, 30, 20], [5, 45, 30, 65]],
                    }
                }
            ]

    ocr = PaddleFactorOCR.__new__(PaddleFactorOCR)
    ocr._engine = _Engine()
    ocr.mode = "ocr"

    names = ocr.recognize_canvas(
        np.zeros((80, 40, 3), dtype=np.uint8),
        [(0, 0, 40, 35), (0, 40, 40, 80)],
    )

    assert names == ["upper", "lower"]
