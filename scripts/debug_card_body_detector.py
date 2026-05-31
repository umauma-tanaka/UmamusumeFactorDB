from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from umafactor.detection.card_body_detector import (  # noqa: E402
    BBox,
    CardBodyDetectorOptions,
    DetectedCardBody,
    detect_card_bodies,
    evaluate_card_bodies,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug independent card body detector.")
    parser.add_argument("image", type=Path, help="Input stitched factor-list image.")
    parser.add_argument("--out", type=Path, required=True, help="Output debug directory.")
    parser.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "card_crop_reference" / "card_bbox_expected.json",
        help="Optional manual bbox JSON for evaluation.",
    )
    parser.add_argument("--role", default=None, help="Optional role label stored in result cards.")
    parser.add_argument("--saturation-threshold", type=int, default=45)
    parser.add_argument("--gray-delta-l-threshold", type=int, default=4)
    parser.add_argument("--min-component-area", type=int, default=200)
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    options = CardBodyDetectorOptions(
        saturation_threshold=args.saturation_threshold,
        gray_delta_l_threshold=args.gray_delta_l_threshold,
        min_component_area=args.min_component_area,
        debug=True,
    )
    run = detect_card_bodies(args.image, options=options, role=args.role)
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)

    evaluation = None
    expected_path = args.expected if args.expected.is_absolute() else ROOT / args.expected
    if expected_path.exists():
        evaluation = evaluate_card_bodies(run.result, expected_path, image_path=args.image)

    cv2.imwrite(str(out / "card_body_mask.png"), run.debug.raw_mask)
    cv2.imwrite(str(out / "card_body_mask_clean.png"), run.debug.clean_mask)
    cv2.imwrite(str(out / "x_projection.png"), _projection_image(run.debug.x_projection, axis="x"))
    cv2.imwrite(str(out / "row_projection_left.png"), _projection_image(run.debug.row_projection_left, axis="y"))
    cv2.imwrite(str(out / "row_projection_right.png"), _projection_image(run.debug.row_projection_right, axis="y"))
    cv2.imwrite(str(out / "card_body_detection_overlay.png"), _overlay(image_bgr, run.result.cards, run.result.column_ranges))
    _write_crops(image_bgr, run.result.cards, crops_dir)
    _write_csv(out / "card_body_detection_debug.csv", run.result.cards)
    contact_sheet = _contact_sheet(image_bgr, run.result.cards, evaluation)
    contact_sheet.save(out / "contact_sheet.png")

    summary = _summary(run.result.cards)
    payload: dict[str, Any] = {
        "result": run.result.to_dict(),
        "summary": summary,
        "evaluation": evaluation,
        "debug_outputs": {
            "card_body_mask": str(out / "card_body_mask.png"),
            "card_body_mask_clean": str(out / "card_body_mask_clean.png"),
            "x_projection": str(out / "x_projection.png"),
            "row_projection_left": str(out / "row_projection_left.png"),
            "row_projection_right": str(out / "row_projection_right.png"),
            "overlay": str(out / "card_body_detection_overlay.png"),
            "csv": str(out / "card_body_detection_debug.csv"),
            "crops_dir": str(crops_dir),
            "contact_sheet": str(out / "contact_sheet.png"),
        },
    }
    (out / "card_body_detection_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_console_summary(run.result.cards, evaluation, out), ensure_ascii=False, indent=2))
    return 0


def _projection_image(values: np.ndarray, *, axis: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_value = float(values.max()) or 1.0
    normalized = np.clip(values / max_value, 0.0, 1.0)
    if axis == "x":
        height = 160
        image = np.full((height, values.size, 3), 255, dtype=np.uint8)
        for x, value in enumerate(normalized):
            y = int(round(height - 1 - value * (height - 1)))
            image[y:, x] = (80, 80, 220)
        return image
    width = 360
    image = np.full((values.size, width, 3), 255, dtype=np.uint8)
    for y, value in enumerate(normalized):
        x = int(round(value * (width - 1)))
        image[y, :x] = (80, 80, 220)
    return image


def _overlay(
    image_bgr: np.ndarray,
    cards: Sequence[DetectedCardBody],
    column_ranges: dict[str, BBox],
) -> np.ndarray:
    overlay = image_bgr.copy()
    height = overlay.shape[0]
    for _name, bbox in column_ranges.items():
        x1, _y1, x2, _y2 = bbox
        cv2.line(overlay, (x1, 0), (x1, height - 1), (0, 180, 255), 2)
        cv2.line(overlay, (x2, 0), (x2, height - 1), (0, 180, 255), 2)
    for card in cards:
        cv2.rectangle(overlay, (card.body_bbox[0], card.body_bbox[1]), (card.body_bbox[2], card.body_bbox[3]), (255, 0, 0), 2)
        cv2.rectangle(overlay, (card.item_bbox[0], card.item_bbox[1]), (card.item_bbox[2], card.item_bbox[3]), (0, 220, 0), 2)
        cv2.line(overlay, (card.body_bbox[0], card.body_bbox[1]), (card.body_bbox[2], card.body_bbox[1]), (0, 0, 255), 1)
        cv2.putText(
            overlay,
            f"r{card.row} c{card.col}",
            (card.item_bbox[0], max(12, card.item_bbox[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _write_crops(image_bgr: np.ndarray, cards: Sequence[DetectedCardBody], crops_dir: Path) -> None:
    for index, card in enumerate(cards):
        x1, y1, x2, y2 = card.item_bbox
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cv2.imwrite(str(crops_dir / f"tile_{index:03d}_r{card.row:02d}_c{card.col}.png"), crop)


def _write_csv(path: Path, cards: Sequence[DetectedCardBody]) -> None:
    fields = [
        "row",
        "col",
        "body_bbox",
        "item_bbox",
        "item_w",
        "item_h",
        "confidence",
        "source",
        "valid",
        "invalid_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for card in cards:
            writer.writerow(
                {
                    "row": card.row,
                    "col": card.col,
                    "body_bbox": _format_bbox(card.body_bbox),
                    "item_bbox": _format_bbox(card.item_bbox),
                    "item_w": card.item_bbox[2] - card.item_bbox[0],
                    "item_h": card.item_bbox[3] - card.item_bbox[1],
                    "confidence": f"{card.confidence:.4f}",
                    "source": card.source,
                    "valid": card.valid,
                    "invalid_reason": card.invalid_reason or "",
                }
            )


def _contact_sheet(
    image_bgr: np.ndarray,
    cards: Sequence[DetectedCardBody],
    evaluation: dict[str, Any] | None,
) -> Image.Image:
    cell_w, cell_h = 260, 120
    cols = 3
    rows = max(1, int(np.ceil(len(cards) / cols)))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    eval_by_key = {}
    if evaluation:
        for row in evaluation.get("rows", []):
            eval_by_key[(row.get("row"), row.get("col"))] = row
    for index, card in enumerate(cards):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        crop = _crop_pil(image_bgr, card.item_bbox)
        crop.thumbnail((cell_w - 8, cell_h - 28))
        sheet.paste(crop, (x + 4, y + 22))
        eval_row = eval_by_key.get((card.row, card.col), {})
        iou = eval_row.get("iou")
        suffix = "" if iou is None else f" IoU={iou:.3f}"
        failures = eval_row.get("hard_failures") or []
        status = "NG" if failures else "OK"
        draw.text((x + 4, y + 4), f"r{card.row} c{card.col} {status}{suffix}", fill=(0, 0, 0), font=font)
    return sheet


def _crop_pil(image_bgr: np.ndarray, bbox: BBox) -> Image.Image:
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return Image.new("RGB", (1, 1), "white")
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _summary(cards: Sequence[DetectedCardBody]) -> dict[str, Any]:
    widths = np.array([card.item_bbox[2] - card.item_bbox[0] for card in cards], dtype=float)
    heights = np.array([card.item_bbox[3] - card.item_bbox[1] for card in cards], dtype=float)
    return {
        "card_count": len(cards),
        "valid_count": sum(1 for card in cards if card.valid),
        "invalid_count": sum(1 for card in cards if not card.valid),
        "item_w": _stats(widths),
        "item_h": _stats(heights),
    }


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"median": None, "min": None, "max": None, "std": None}
    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _console_summary(
    cards: Sequence[DetectedCardBody],
    evaluation: dict[str, Any] | None,
    out: Path,
) -> dict[str, Any]:
    summary = _summary(cards)
    payload = {
        "output_dir": str(out),
        "card_count": summary["card_count"],
        "valid_count": summary["valid_count"],
        "invalid_count": summary["invalid_count"],
        "item_w": summary["item_w"],
        "item_h": summary["item_h"],
    }
    if evaluation:
        payload["evaluation"] = {
            "manual_count": evaluation.get("manual_count"),
            "matched_count": evaluation.get("matched_count"),
            "mean_iou": evaluation.get("mean_iou"),
            "min_iou": evaluation.get("min_iou"),
            "iou_below_0_75_count": len(evaluation.get("iou_below_0_75", [])),
            "hard_failure_count": evaluation.get("hard_failure_count"),
        }
    return payload


def _format_bbox(bbox: BBox) -> str:
    return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"


if __name__ == "__main__":
    raise SystemExit(main())
