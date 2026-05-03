from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from umafactor import pipeline
from umafactor.recognition import image_preprocessing
from umafactor.recognition.image_preprocessing import prepare_factor_image


@dataclass
class DummySection:
    portrait_bbox: tuple[int, int, int, int] = (1, 2, 3, 4)


@dataclass
class DummyBox:
    uma_index: int = 0
    row_index: int = 0
    col_index: int = 0
    color: str = "white"
    text_img: np.ndarray | None = None
    rank_img: np.ndarray | None = None
    bbox: tuple[int, int, int, int] = (10, 20, 30, 40)


def _image(height: int = 8, width: int = 10) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _boxes(count: int) -> list[DummyBox]:
    img = _image(2, 3)
    return [
        DummyBox(
            uma_index=i % 3,
            row_index=i,
            col_index=0,
            text_img=img,
            rank_img=img,
        )
        for i in range(count)
    ]


def test_pipeline_keeps_debug_crop_helper_alias() -> None:
    assert pipeline._dump_debug_crops is image_preprocessing.dump_debug_crops


def test_prepare_factor_image_loads_and_extracts_boxes(monkeypatch) -> None:
    img_orig = _image()
    norm_img = _image(16, 20)
    sections = [DummySection(), DummySection(), DummySection()]
    boxes = _boxes(9)
    calls: list[str] = []

    monkeypatch.setattr(image_preprocessing.cv2, "imread", lambda path: img_orig)

    def fake_normalize_width(img: np.ndarray, target_width: int):
        calls.append("normalize")
        assert img is img_orig
        assert target_width == image_preprocessing.BASE_WIDTH
        return norm_img, 2.0

    def fake_detect_chara_sections(img: np.ndarray):
        calls.append("sections")
        assert img is norm_img
        return sections

    def fake_extract_factor_boxes(img: np.ndarray, section_arg):
        calls.append("boxes")
        assert img is norm_img
        assert section_arg is sections
        return boxes

    monkeypatch.setattr(image_preprocessing, "normalize_width", fake_normalize_width)
    monkeypatch.setattr(
        image_preprocessing, "detect_chara_sections", fake_detect_chara_sections
    )
    monkeypatch.setattr(image_preprocessing, "extract_factor_boxes", fake_extract_factor_boxes)

    prepared = prepare_factor_image("input.png", auto_debug=False)

    assert calls == ["normalize", "sections", "boxes"]
    assert prepared.img_orig is img_orig
    assert prepared.norm_img is norm_img
    assert prepared.scale == 2.0
    assert prepared.sections is sections
    assert prepared.boxes is boxes


def test_prepare_factor_image_raises_when_image_missing(monkeypatch) -> None:
    monkeypatch.setattr(image_preprocessing.cv2, "imread", lambda path: None)

    with pytest.raises(FileNotFoundError):
        prepare_factor_image("missing.png")


def test_prepare_factor_image_raises_when_sections_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(image_preprocessing.cv2, "imread", lambda path: _image())
    monkeypatch.setattr(
        image_preprocessing,
        "normalize_width",
        lambda img, target_width: (_image(16, 20), 2.0),
    )
    monkeypatch.setattr(image_preprocessing, "detect_chara_sections", lambda img: [])

    with pytest.raises(RuntimeError):
        prepare_factor_image("input.png")


def test_prepare_factor_image_auto_dumps_when_box_count_is_low(monkeypatch) -> None:
    img_orig = _image()
    norm_img = _image(16, 20)
    sections = [DummySection(), DummySection(), DummySection()]
    boxes = _boxes(8)
    dump_calls: list[tuple[np.ndarray, list[DummySection], list[DummyBox], str]] = []

    monkeypatch.setattr(image_preprocessing.cv2, "imread", lambda path: img_orig)
    monkeypatch.setattr(
        image_preprocessing,
        "normalize_width",
        lambda img, target_width: (norm_img, 2.0),
    )
    monkeypatch.setattr(image_preprocessing, "detect_chara_sections", lambda img: sections)
    monkeypatch.setattr(image_preprocessing, "extract_factor_boxes", lambda img, sec: boxes)
    monkeypatch.setattr(
        image_preprocessing,
        "dump_debug_crops",
        lambda img, sec, box_list, out_dir: dump_calls.append(
            (img, sec, box_list, out_dir)
        ),
    )

    prepare_factor_image(r"fixtures\sample.png", auto_debug=True)

    assert len(dump_calls) == 1
    dump_img, dump_sections, dump_boxes, dump_dir = dump_calls[0]
    assert dump_img is norm_img
    assert dump_sections is sections
    assert dump_boxes is boxes
    assert dump_dir == image_preprocessing.os.path.join(
        "tests", "fixtures", "debug_crops", "sample"
    )


def test_prepare_factor_image_uses_explicit_debug_dir(monkeypatch) -> None:
    img_orig = _image()
    norm_img = _image(16, 20)
    sections = [DummySection(), DummySection(), DummySection()]
    boxes = _boxes(9)
    dump_dirs: list[str] = []

    monkeypatch.setattr(image_preprocessing.cv2, "imread", lambda path: img_orig)
    monkeypatch.setattr(
        image_preprocessing,
        "normalize_width",
        lambda img, target_width: (norm_img, 2.0),
    )
    monkeypatch.setattr(image_preprocessing, "detect_chara_sections", lambda img: sections)
    monkeypatch.setattr(image_preprocessing, "extract_factor_boxes", lambda img, sec: boxes)
    monkeypatch.setattr(
        image_preprocessing,
        "dump_debug_crops",
        lambda img, sec, box_list, out_dir: dump_dirs.append(out_dir),
    )

    prepare_factor_image("sample.png", debug_crops_dir="debug-out", auto_debug=False)

    assert dump_dirs == ["debug-out"]
