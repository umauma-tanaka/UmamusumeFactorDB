from __future__ import annotations

import subprocess
import sys
import shutil
import uuid
from pathlib import Path

import cv2

from umafactor.detection.factor_list import detect_stitched_factor_list
from umafactor.evaluation.ocr_dataset import (
    evaluate_ocr_factors,
    load_expected_ocr_factors,
)
from umafactor.recognition.factor_list_ocr import (
    crop_factor_list_card_region,
    crop_factor_list_name_region,
    factor_list_card_region_bbox,
    recognize_factor_list_tile_names,
)
from umafactor.recognition.paddle_ocr_adapter import _extract_texts, _normalize_text


CASE_DIR = Path("datasets") / "test_factor_01"


class _FakeOCR:
    def recognize(self, img_bgr) -> str:
        return "桜花賞"

    def recognize_blue(self, img_bgr) -> str:
        return "スピード"

    def recognize_red(self, img_bgr) -> str:
        return "ダート"

    def recognize_with_parts(self, img_bgr) -> tuple[str, list[str]]:
        return "ゆきあかり、おいかけて", ["ゆきあかり", "おいかけて"]


def test_load_expected_ocr_factors_reads_utf8_csv() -> None:
    expected = load_expected_ocr_factors(CASE_DIR / "expected_ocr.csv")

    assert len(expected) == 44
    assert expected[0].name == "スピード"
    assert expected[0].star == 3
    assert expected[-1].name == "叩き良化型"
    assert expected[-1].star == 3


def test_detect_stitched_factor_list_parent_tiles_matches_expected_star_sequence() -> None:
    image = cv2.imread(str(CASE_DIR / "expected_stitched.png"))
    assert image is not None
    expected = load_expected_ocr_factors(CASE_DIR / "expected_ocr.csv")

    detection = detect_stitched_factor_list(image, section_index=0)
    metrics = evaluate_ocr_factors(expected, detection.tiles)

    assert detection.role == "parent"
    assert len(detection.tiles) == 44
    assert metrics["detected_count"] == 44
    assert metrics["count_delta"] == 0
    assert metrics["star_accuracy"] == 1.0
    assert metrics["extra_count"] == 0
    assert metrics["missing_count"] == 0


def test_detect_stitched_factor_list_sections_do_not_overlap() -> None:
    image = cv2.imread(str(CASE_DIR / "expected_stitched.png"))
    assert image is not None

    detections = [detect_stitched_factor_list(image, section_index=index) for index in range(3)]
    bbox_sets = [{tile.bbox for tile in detection.tiles} for detection in detections]

    assert bbox_sets[0].isdisjoint(bbox_sets[1])
    assert bbox_sets[1].isdisjoint(bbox_sets[2])


def test_recognize_factor_list_tile_names_fills_raw_names_for_evaluation() -> None:
    image = cv2.imread(str(CASE_DIR / "expected_stitched.png"))
    assert image is not None
    expected = load_expected_ocr_factors(CASE_DIR / "expected_ocr.csv")[:4]
    detection = detect_stitched_factor_list(image, section_index=0)

    tiles = recognize_factor_list_tile_names(image, detection.tiles[:4], _FakeOCR())
    metrics = evaluate_ocr_factors(expected, tiles, evaluate_names=True)

    assert [tile.raw_name for tile in tiles] == [
        "スピード",
        "ダート",
        "ゆきあかり、おいかけて",
        "桜花賞",
    ]
    assert metrics["name_evaluated_count"] == 4
    assert metrics["blank_name_count"] == 0
    assert metrics["name_accuracy"] == 1.0
    assert metrics["name_similarity_mean"] == 1.0


def test_recognize_factor_list_tile_names_accepts_crop_variant() -> None:
    image = cv2.imread(str(CASE_DIR / "expected_stitched.png"))
    assert image is not None
    detection = detect_stitched_factor_list(image, section_index=0)

    tiles = recognize_factor_list_tile_names(
        image,
        detection.tiles[:1],
        _FakeOCR(),
        crop_variant="full",
    )

    assert tiles[0].raw_name == "スピード"


def test_crop_factor_list_card_region_is_larger_than_name_region() -> None:
    image = cv2.imread(str(CASE_DIR / "expected_stitched.png"))
    assert image is not None
    detection = detect_stitched_factor_list(image, section_index=0)
    tile = detection.tiles[0]

    name_crop = crop_factor_list_name_region(image, tile)
    card_crop = crop_factor_list_card_region(image, tile)

    assert card_crop.shape[0] > name_crop.shape[0]
    assert card_crop.shape[1] > name_crop.shape[1]


def test_factor_list_card_region_keeps_long_left_column_names() -> None:
    image = cv2.imread(str(CASE_DIR / "expected_stitched.png"))
    assert image is not None
    detection = detect_stitched_factor_list(image, section_index=0)
    tile = detection.tiles[2]
    tile_width = tile.bbox[2] - tile.bbox[0]

    x0, y0, x1, y1 = factor_list_card_region_bbox(image, tile)

    assert x0 > tile.bbox[0]
    assert x1 > tile.bbox[2] + int(round(tile_width * 0.30))
    assert y0 < tile.bbox[1]
    assert y1 > tile.bbox[3]


def test_evaluate_factor_ocr_writes_all_section_overlays() -> None:
    output_dir = Path("outputs") / "test_runs" / f"factor_ocr_overlay_{uuid.uuid4().hex}"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_factor_ocr.py",
                "--case",
                str(CASE_DIR),
                "--output",
                str(output_dir),
                "--skip-ocr",
                "--overlay-sections",
                "all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        original = cv2.imread(str(CASE_DIR / "expected_stitched.png"))
        assert original is not None
        for filename in ("parent_overlay.png", "ancestor1_overlay.png", "ancestor2_overlay.png"):
            overlay = cv2.imread(str(output_dir / filename))
            assert overlay is not None, filename
            assert overlay.shape[0] == original.shape[0]
            assert overlay.shape[1] > original.shape[1]
        combined = cv2.imread(str(output_dir / "all_roles_overlay.png"))
        assert combined is not None
        assert combined.shape[0] == original.shape[0]
        assert combined.shape[1] > original.shape[1]
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_evaluate_ocr_factors_reports_similarity_buckets() -> None:
    expected = load_expected_ocr_factors(CASE_DIR / "expected_ocr.csv")[:3]

    class _Detected:
        def __init__(self, order: int, raw_name: str, star: int) -> None:
            self.order = order
            self.raw_name = raw_name
            self.star = star

    detected = [
        _Detected(0, expected[0].name, expected[0].star),
        _Detected(1, expected[1].name[:-1], expected[1].star),
        _Detected(2, "", expected[2].star),
    ]

    metrics = evaluate_ocr_factors(expected, detected, evaluate_names=True)

    assert metrics["name_correct"] == 1
    assert metrics["name_similarity_buckets"]["100%"] == 1
    assert metrics["name_similarity_buckets"]["blank"] == 1
    assert sum(metrics["name_similarity_buckets"].values()) == 3


def test_paddle_ocr_text_extraction_accepts_v3_ocr_payload() -> None:
    result = [{"res": {"rec_texts": ["skill", "name"]}}]

    assert _extract_texts(result) == ["skill", "name"]


def test_paddle_ocr_text_normalization_unifies_circle_like_chars() -> None:
    assert _normalize_text("skill 0") == "skill\u25cb"


def test_paddle_ocr_text_normalization_removes_card_ui_suffix_noise() -> None:
    assert _normalize_text("RANKスピード★★★") == "スピード"
