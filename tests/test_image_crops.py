from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umafactor import pipeline
from umafactor.recognition import image_crops


@dataclass
class DummySection:
    portrait_bbox: tuple[int, int, int, int]


def _image(height: int = 80, width: int = 100) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def test_pipeline_keeps_crop_helper_aliases() -> None:
    assert pipeline._crop_from_original is image_crops.crop_from_original
    assert pipeline._display_crop_from_original is image_crops.display_crop_from_original
    assert pipeline._display_crop_for_slot is image_crops.display_crop_for_slot
    assert pipeline._crop_rank_from_original is image_crops.crop_rank_from_original
    assert pipeline._extract_character_icon_bgr is image_crops.extract_character_icon_bgr


def test_crop_from_original_projects_bbox_with_shift() -> None:
    img = _image()

    crop = image_crops.crop_from_original(
        img,
        bbox=(20, 10, 50, 30),
        scale=2.0,
        dy=1,
        dx=-2,
    )

    assert crop.shape == (10, 15, 3)
    np.testing.assert_array_equal(crop, img[6:16, 8:23])


def test_crop_from_original_preserves_empty_crop_when_bbox_is_outside() -> None:
    img = _image(4, 4)

    crop = image_crops.crop_from_original(img, bbox=(20, 20, 30, 30), scale=1.0)

    assert crop.shape == (0, 0, 3)


def test_display_crop_from_original_adds_fixed_horizontal_padding() -> None:
    img = _image()

    crop = image_crops.display_crop_from_original(
        img,
        bbox=(40, 10, 50, 20),
        scale=1.0,
        pad_y_norm=2,
    )

    assert crop.shape == (14, 50, 3)
    np.testing.assert_array_equal(crop, img[8:22, 8:58])


def test_display_crop_for_slot_uses_blue_vertical_padding() -> None:
    img = _image()

    crop = image_crops.display_crop_for_slot(
        img,
        norm_img_shape=img.shape,
        bbox=(40, 20, 50, 30),
        scale=1.0,
        is_blue_slot=True,
        is_red_slot=False,
    )

    np.testing.assert_array_equal(crop, img[12:38, 8:58])


def test_display_crop_for_slot_extends_red_bottom_only() -> None:
    img = _image()

    crop = image_crops.display_crop_for_slot(
        img,
        norm_img_shape=(35, 100, 3),
        bbox=(40, 20, 50, 30),
        scale=1.0,
        is_blue_slot=False,
        is_red_slot=True,
    )

    np.testing.assert_array_equal(crop, img[18:37, 8:58])


def test_display_crop_for_slot_uses_default_crop_for_other_slots() -> None:
    img = _image()

    crop = image_crops.display_crop_for_slot(
        img,
        norm_img_shape=img.shape,
        bbox=(40, 20, 50, 30),
        scale=1.0,
        is_blue_slot=False,
        is_red_slot=False,
    )

    np.testing.assert_array_equal(crop, img[18:32, 8:58])


def test_crop_rank_from_original_uses_legacy_rank_region() -> None:
    img = _image()

    crop = image_crops.crop_rank_from_original(
        img,
        bbox=(10, 20, 110, 60),
        scale=2.0,
    )

    np.testing.assert_array_equal(crop, img[16:24, 39:55])


def test_crop_rank_from_original_uses_detected_rank_bbox_with_padding() -> None:
    img = _image()

    crop = image_crops.crop_rank_from_original(
        img,
        bbox=(0, 0, 0, 0),
        scale=1.0,
        rank_bbox=(30, 20, 60, 30),
    )

    np.testing.assert_array_equal(crop, img[18:32, 30:60])


def test_extract_character_icon_bgr_returns_32_square() -> None:
    img = _image()
    section = DummySection(portrait_bbox=(10, 5, 50, 55))

    crop = image_crops.extract_character_icon_bgr(img, section)

    assert crop.shape == (32, 32, 3)
    assert crop.dtype == np.uint8


def test_extract_character_icon_bgr_returns_zero_image_for_empty_crop() -> None:
    img = _image(4, 4)
    section = DummySection(portrait_bbox=(20, 20, 30, 30))

    crop = image_crops.extract_character_icon_bgr(img, section)

    assert crop.shape == (32, 32, 3)
    assert crop.dtype == np.uint8
    assert not crop.any()
