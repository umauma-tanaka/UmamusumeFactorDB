"""Debug artifact writer for formal card-body detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from ..detection.card_body_detector import detect_card_bodies


@dataclass(frozen=True)
class CardDetectionDebugResult:
    summary: dict[str, Any]


def write_card_detection_debug(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    role_hint: str | None = None,
    expected_rows: int | None = None,
) -> CardDetectionDebugResult:
    """Write card detection debug artifacts for one stitched image."""

    del expected_rows
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "cards"
    crops_dir.mkdir(exist_ok=True)

    run = detect_card_bodies(image_path, role=role_hint)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        for index, card in enumerate(run.result.cards):
            x0, y0, x1, y1 = card.item_bbox
            crop = image[y0:y1, x0:x1]
            if crop.size:
                cv2.imwrite(str(crops_dir / f"card_{index:03d}_r{card.row:02d}_c{card.col}.png"), crop)

    payload = run.result.to_dict()
    (output_dir / "card_detection_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "card_count": len(run.result.cards),
        "valid_count": sum(1 for card in run.result.cards if card.valid),
        "invalid_count": sum(1 for card in run.result.cards if not card.valid),
        "median_body_w": run.result.median_body_w,
        "median_body_h": run.result.median_body_h,
        "median_row_pitch": run.result.median_row_pitch,
        "result_path": str(output_dir / "card_detection_result.json"),
        "crops_dir": str(crops_dir),
    }
    return CardDetectionDebugResult(summary=summary)
