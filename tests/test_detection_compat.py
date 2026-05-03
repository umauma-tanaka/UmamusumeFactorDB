from __future__ import annotations

from umafactor import cropper
from umafactor import detection
from umafactor.detection import boxes, rows, sections, stars, types


def test_cropper_reexports_public_detection_api() -> None:
    assert cropper.BASE_WIDTH == detection.BASE_WIDTH
    assert cropper.BASE_HEIGHT == detection.BASE_HEIGHT
    assert cropper.FactorBox is detection.FactorBox
    assert cropper.CharaSection is detection.CharaSection
    assert cropper.normalize_width is detection.normalize_width
    assert cropper.detect_chara_sections is detection.detect_chara_sections
    assert cropper.detect_factor_color is detection.detect_factor_color
    assert cropper.extract_factor_boxes is detection.extract_factor_boxes


def test_detection_modules_own_extracted_helpers() -> None:
    assert detection.FactorBox is types.FactorBox
    assert detection.CharaSection is types.CharaSection
    assert detection.detect_chara_sections is sections.detect_chara_sections
    assert detection.extract_factor_boxes is boxes.extract_factor_boxes
    assert cropper._detect_factor_rows is rows._detect_factor_rows
    assert cropper._cluster_stars_into_rows is stars._cluster_stars_into_rows
