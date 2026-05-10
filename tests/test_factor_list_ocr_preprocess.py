from __future__ import annotations

import numpy as np

from umafactor.detection.factor_list import FactorListTile
from umafactor.recognition.factor_list_ocr import (
    crop_factor_list_card_region,
    pack_ocr_canvas,
    prepare_factor_list_ocr_crop,
    recognize_factor_list_tile_names,
)


class RecordingOCR:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def recognize(self, img_bgr: np.ndarray) -> str:
        self.calls.append(img_bgr.shape)
        return "raw"

    def recognize_blue(self, img_bgr: np.ndarray) -> str:
        self.calls.append(img_bgr.shape)
        return "blue"

    def recognize_red(self, img_bgr: np.ndarray) -> str:
        self.calls.append(img_bgr.shape)
        return "red"

    def recognize_with_parts(self, img_bgr: np.ndarray) -> tuple[str, list[str]]:
        self.calls.append(img_bgr.shape)
        return "green", ["green"]


class RecordingBatchOCR(RecordingOCR):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[list[tuple[int, int, int]]] = []

    def recognize_many(self, images_bgr: list[np.ndarray]) -> list[str]:
        self.batch_calls.append([image.shape for image in images_bgr])
        return [f"batch-{index}" for index in range(len(images_bgr))]


class RecordingCanvasOCR:
    def __init__(self) -> None:
        self.canvas_calls: list[tuple[tuple[int, int, int], list[tuple[int, int, int, int]]]] = []

    def recognize_canvas(
        self,
        canvas_bgr: np.ndarray,
        regions: list[tuple[int, int, int, int]],
    ) -> list[str]:
        self.canvas_calls.append((canvas_bgr.shape, list(regions)))
        return [f"canvas-{index}" for index in range(len(regions))]


def test_prepare_factor_list_ocr_crop_upscales_to_minimum_size() -> None:
    crop = np.full((32, 80, 3), 160, dtype=np.uint8)

    prepared = prepare_factor_list_ocr_crop(
        crop,
        min_width=240,
        min_height=120,
        max_upscale=4.0,
        sharpen_strength=0,
        contrast_clip_limit=0,
    )

    assert prepared.dtype == np.uint8
    assert prepared.shape[1] >= 240
    assert prepared.shape[0] >= 120
    assert crop.shape == (32, 80, 3)


def test_prepare_factor_list_ocr_crop_caps_upscale() -> None:
    crop = np.full((10, 20, 3), 160, dtype=np.uint8)

    prepared = prepare_factor_list_ocr_crop(
        crop,
        min_width=400,
        min_height=400,
        max_upscale=2.0,
        sharpen_strength=0,
        contrast_clip_limit=0,
    )

    assert prepared.shape[:2] == (20, 40)


def test_recognize_factor_list_tile_names_preprocesses_crop_before_ocr() -> None:
    image = np.full((80, 120, 3), 210, dtype=np.uint8)
    tile = _tile()
    ocr = RecordingOCR()

    tiles = recognize_factor_list_tile_names(
        image,
        [tile],
        ocr,
        crop_target="card",
        min_crop_width=220,
        min_crop_height=100,
        max_upscale=4.0,
        sharpen_strength=0,
        contrast_clip_limit=0,
    )

    assert tiles[0].raw_name == "raw"
    assert ocr.calls[0][1] >= 220
    assert ocr.calls[0][0] >= 100


def test_recognize_factor_list_tile_names_can_disable_preprocess() -> None:
    image = np.full((80, 120, 3), 210, dtype=np.uint8)
    tile = _tile()
    raw_crop = crop_factor_list_card_region(image, tile)
    ocr = RecordingOCR()

    recognize_factor_list_tile_names(
        image,
        [tile],
        ocr,
        crop_target="card",
        preprocess_crop=False,
    )

    assert ocr.calls == [raw_crop.shape]


def test_recognize_factor_list_tile_names_can_use_batch_ocr() -> None:
    image = np.full((120, 180, 3), 210, dtype=np.uint8)
    tiles = [_tile(order=index, y=30 + index * 20) for index in range(3)]
    ocr = RecordingBatchOCR()

    recognized = recognize_factor_list_tile_names(
        image,
        tiles,
        ocr,
        crop_target="card",
        ocr_execution_mode="batch",
        canvas_batch_size=2,
        preprocess_crop=False,
    )

    assert [tile.raw_name for tile in recognized] == ["batch-0", "batch-1", "batch-0"]
    assert [len(call) for call in ocr.batch_calls] == [2, 1]
    assert ocr.calls == []


def test_recognize_factor_list_tile_names_can_use_canvas_ocr() -> None:
    image = np.full((120, 180, 3), 210, dtype=np.uint8)
    tiles = [_tile(order=index, y=30 + index * 20) for index in range(3)]
    ocr = RecordingCanvasOCR()

    recognized = recognize_factor_list_tile_names(
        image,
        tiles,
        ocr,  # type: ignore[arg-type]
        crop_target="card",
        ocr_execution_mode="canvas",
        canvas_batch_size=2,
        canvas_padding=8,
        preprocess_crop=False,
    )

    assert [tile.raw_name for tile in recognized] == ["canvas-0", "canvas-1", "canvas-0"]
    assert [len(regions) for _shape, regions in ocr.canvas_calls] == [2, 1]
    assert all(shape[0] > 0 and shape[1] > 0 for shape, _regions in ocr.canvas_calls)


def test_recognize_factor_list_tile_names_canvas_batch_size_zero_uses_one_canvas() -> None:
    image = np.full((120, 180, 3), 210, dtype=np.uint8)
    tiles = [_tile(order=index, y=30 + index * 20) for index in range(3)]
    ocr = RecordingCanvasOCR()

    recognized = recognize_factor_list_tile_names(
        image,
        tiles,
        ocr,  # type: ignore[arg-type]
        crop_target="card",
        ocr_execution_mode="canvas",
        canvas_batch_size=0,
        canvas_padding=8,
        preprocess_crop=False,
    )

    assert [tile.raw_name for tile in recognized] == ["canvas-0", "canvas-1", "canvas-2"]
    assert [len(regions) for _shape, regions in ocr.canvas_calls] == [3]


def test_recognize_factor_list_tile_names_batch_size_zero_uses_one_batch() -> None:
    image = np.full((120, 180, 3), 210, dtype=np.uint8)
    tiles = [_tile(order=index, y=30 + index * 20) for index in range(3)]
    ocr = RecordingBatchOCR()

    recognized = recognize_factor_list_tile_names(
        image,
        tiles,
        ocr,
        crop_target="card",
        ocr_execution_mode="batch",
        canvas_batch_size=0,
        preprocess_crop=False,
    )

    assert [tile.raw_name for tile in recognized] == ["batch-0", "batch-1", "batch-2"]
    assert [len(call) for call in ocr.batch_calls] == [3]


def test_pack_ocr_canvas_returns_regions_for_each_crop() -> None:
    crops = [
        np.full((10, 20, 3), 100, dtype=np.uint8),
        np.full((12, 18, 3), 140, dtype=np.uint8),
    ]

    packed = pack_ocr_canvas(crops, padding=5)

    assert packed.canvas.shape[:2] == (37, 30)
    assert packed.regions == [(5, 5, 25, 15), (5, 20, 23, 32)]


def _tile(order: int = 0, y: int = 30) -> FactorListTile:
    return FactorListTile(
        order=order,
        section_index=0,
        role="parent",
        row_index=order,
        col_index=0,
        color="white",
        star=2,
        bbox=(10, y, 70, y + 20),
        bbox_norm=(10, y, 70, y + 20),
    )
