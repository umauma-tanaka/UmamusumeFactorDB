from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import cv2

from umafactor.detection.factor_list import FactorListTile
from umafactor.detection.card_body_detector import (
    detect_card_bodies,
    evaluate_card_bodies,
)
from umafactor.recognition.factor_list_ocr import build_factor_list_card_debugs


FIXTURE_DIR = Path("tests/fixtures/card_crop_reference")
EXPECTED_PATH = FIXTURE_DIR / "card_bbox_expected.json"


def test_card_body_detector_passes_manual_bbox_fixture() -> None:
    cases = [
        ("sources/stitched_1.png", 44),
        ("sources/stitched_2.png", 40),
        ("sources/stitched_3.png", 34),
    ]

    for source, expected_count in cases:
        image_path = FIXTURE_DIR / source
        run = detect_card_bodies(image_path)
        evaluation = evaluate_card_bodies(run.result, EXPECTED_PATH, image_path=image_path)

        assert len(run.result.cards) == expected_count
        assert evaluation["manual_count"] == expected_count
        assert evaluation["matched_count"] == expected_count
        assert evaluation["hard_failure_count"] == 0
        assert evaluation["mean_iou"] >= 0.87
        assert evaluation["min_iou"] >= 0.80


def test_card_body_detector_debug_cli_writes_outputs(tmp_path: Path) -> None:
    out = tmp_path / "card_body"
    image_path = FIXTURE_DIR / "sources" / "stitched_1.png"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/debug_card_body_detector.py",
            str(image_path),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["card_count"] == 44
    assert summary["evaluation"]["hard_failure_count"] == 0
    for filename in [
        "card_body_mask.png",
        "card_body_mask_clean.png",
        "x_projection.png",
        "row_projection_left.png",
        "row_projection_right.png",
        "card_body_detection_overlay.png",
        "card_body_detection_result.json",
        "card_body_detection_debug.csv",
        "contact_sheet.png",
    ]:
        assert (out / filename).exists()
    assert len(list((out / "crops").glob("*.png"))) == 44


def test_card_body_detector_has_no_ocr_or_sheet_imports() -> None:
    module_path = Path("src/umafactor/detection/card_body_detector.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    forbidden = {
        "umafactor.recognition.factor_list_ocr",
        "umafactor.detection.star_slots",
        "umafactor.recognition.paddle_ocr_adapter",
        "umafactor.sheet_writer",
        "umafactor.schema",
    }
    assert imported_modules.isdisjoint(forbidden)


def test_factor_list_card_debugs_use_card_body_detector_on_manual_reference() -> None:
    image_path = FIXTURE_DIR / "sources" / "stitched_1.png"
    image = cv2.imread(str(image_path))
    assert image is not None
    tiles, manual_bboxes = _manual_tiles("stitched_1")

    debug_by_tile = build_factor_list_card_debugs(image, tiles, card_detector="card-body")

    assert len(debug_by_tile) == len(tiles)
    assert {debug.icon_selected_by for debug in debug_by_tile.values()} == {"card_body"}
    assert not any(debug.fallback for debug in debug_by_tile.values())
    min_iou = min(
        _bbox_iou(debug_by_tile[(0, tile.order, tile.row_index, tile.col_index)].card_bbox, manual)
        for tile, manual in zip(tiles, manual_bboxes)
    )
    assert min_iou >= 0.80


def _manual_tiles(image_id: str) -> tuple[list[FactorListTile], list[tuple[int, int, int, int]]]:
    data = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    items = [item for item in data["items"] if item["image_id"] == image_id]
    tiles: list[FactorListTile] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for order, item in enumerate(items):
        bbox = tuple(int(value) for value in item["bbox"])
        color = item.get("color_estimate") or "white"
        tile = FactorListTile(
            order=order,
            section_index=0,
            role=item["role"],
            row_index=int(item["row"]),
            col_index=int(item["col"]),
            color=color,
            star=0,
            bbox=bbox,
            bbox_norm=bbox,
        )
        tiles.append(tile)
        bboxes.append(bbox)
    return tiles, bboxes


def _bbox_iou(
    lhs: tuple[int, int, int, int],
    rhs: tuple[int, int, int, int],
) -> float:
    left = max(lhs[0], rhs[0])
    top = max(lhs[1], rhs[1])
    right = min(lhs[2], rhs[2])
    bottom = min(lhs[3], rhs[3])
    inter = max(0, right - left) * max(0, bottom - top)
    lhs_area = max(0, lhs[2] - lhs[0]) * max(0, lhs[3] - lhs[1])
    rhs_area = max(0, rhs[2] - rhs[0]) * max(0, rhs[3] - rhs[1])
    union = lhs_area + rhs_area - inter
    return float(inter / union) if union else 0.0
