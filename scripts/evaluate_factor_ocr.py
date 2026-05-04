"""Evaluate parent-factor detection on a stitched factor image.

Example:
    python scripts/evaluate_factor_ocr.py --case datasets/test_factor_01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - fallback for minimal environments
    Image = None
    ImageDraw = None
    ImageFont = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from umafactor.detection.factor_list import (  # noqa: E402
    FactorListTile,
    detect_stitched_factor_list,
)
from umafactor.evaluation.ocr_dataset import (  # noqa: E402
    ExpectedOcrFactor,
    evaluate_ocr_factors,
    load_expected_ocr_factors,
)
from umafactor.recognition.factor_list_ocr import (  # noqa: E402
    factor_list_ocr_region_bbox,
    recognize_factor_list_tile_names,
)


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Evaluate OCR-ready factor list detection.")
    parser.add_argument(
        "--case",
        type=Path,
        default=ROOT / "datasets" / "test_factor_01",
        help="Case directory containing expected_stitched.png and expected_ocr.csv.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Stitched image path. Defaults to expected_stitched.png in the case directory.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="Expected OCR CSV path. Defaults to expected_ocr.csv in the case directory.",
    )
    parser.add_argument(
        "--section-index",
        type=int,
        default=0,
        help="Section index to evaluate. 0 is parent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "ocr_eval" / "test_factor_01",
        help="Directory for metrics and debug artifacts.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR and evaluate detection/star extraction only.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["easyocr", "paddleocr"],
        default="paddleocr",
        help="OCR engine used for raw factor-name extraction.",
    )
    parser.add_argument(
        "--ocr-crop-target",
        choices=["name", "card"],
        default="card",
        help="Image region passed to OCR for each detected factor tile.",
    )
    parser.add_argument(
        "--crop-variant",
        choices=["current", "wide", "upper", "full"],
        default="current",
        help="Name crop variant used before OCR.",
    )
    parser.add_argument(
        "--overlay-sections",
        choices=["selected", "all"],
        default="all",
        help="Sections to render as overlay images. all writes parent and ancestor overlays.",
    )
    parser.add_argument(
        "--paddle-mode",
        choices=["recognition", "ocr"],
        default="ocr",
        help="PaddleOCR mode. recognition runs text recognition directly on name crops.",
    )
    parser.add_argument(
        "--paddle-lang",
        default="japan",
        help="PaddleOCR language used when --paddle-mode=ocr.",
    )
    parser.add_argument(
        "--paddle-cache-dir",
        type=Path,
        default=None,
        help="Workspace-local PaddleOCR model/cache directory.",
    )
    parser.add_argument(
        "--paddle-det-limit-side-len",
        type=int,
        default=None,
        help="PaddleOCR text_det_limit_side_len. Leave unset to use PaddleOCR defaults.",
    )
    parser.add_argument(
        "--paddle-det-limit-type",
        choices=["min", "max"],
        default=None,
        help="PaddleOCR text_det_limit_type. Use min to upscale small card crops.",
    )
    parser.add_argument(
        "--paddle-det-thresh",
        type=float,
        default=None,
        help="PaddleOCR text_det_thresh.",
    )
    parser.add_argument(
        "--paddle-det-box-thresh",
        type=float,
        default=None,
        help="PaddleOCR text_det_box_thresh.",
    )
    parser.add_argument(
        "--paddle-det-unclip-ratio",
        type=float,
        default=None,
        help="PaddleOCR text_det_unclip_ratio.",
    )
    parser.add_argument(
        "--paddle-rec-score-thresh",
        type=float,
        default=None,
        help="PaddleOCR text_rec_score_thresh.",
    )
    args = parser.parse_args()

    case_dir = _resolve_path(args.case)
    image_path = _resolve_path(args.image) if args.image else case_dir / "expected_stitched.png"
    expected_path = _resolve_path(args.expected) if args.expected else case_dir / "expected_ocr.csv"
    output_dir = _resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"failed to load image: {image_path}")

    expected = load_expected_ocr_factors(expected_path)
    ocr = None if args.skip_ocr else _build_ocr(args)
    detection = detect_stitched_factor_list(image, section_index=args.section_index)
    tiles = _recognize_tiles_if_needed(image, detection.tiles, ocr, args)
    metrics = evaluate_ocr_factors(expected, tiles, evaluate_names=not args.skip_ocr)

    report = {
        "case_dir": str(case_dir),
        "image_path": str(image_path),
        "expected_path": str(expected_path),
        "section_index": args.section_index,
        "role": detection.role,
        "ocr_enabled": not args.skip_ocr,
        "ocr_engine": args.ocr_engine if not args.skip_ocr else "none",
        "ocr_crop_target": args.ocr_crop_target,
        "crop_variant": args.crop_variant,
        "overlay_sections": args.overlay_sections,
        "paddle_mode": args.paddle_mode if args.ocr_engine == "paddleocr" else None,
        "paddle_lang": args.paddle_lang if args.ocr_engine == "paddleocr" else None,
        "paddle_params": _paddle_params(args) if args.ocr_engine == "paddleocr" else None,
        "metrics": metrics,
        "detected": [
            {
                "order": tile.order,
                "row_index": tile.row_index,
                "col_index": tile.col_index,
                "color": tile.color,
                "star": tile.star,
                "raw_name": tile.raw_name,
                "bbox": list(tile.bbox),
                "bbox_norm": list(tile.bbox_norm),
            }
            for tile in tiles
        ],
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_detected_csv(output_dir / "detected_parent_factors.csv", tiles)
    _write_comparison_csv(output_dir / "comparison_parent_factors.csv", expected, tiles)
    _write_summary(output_dir / "summary.md", report)
    _write_section_overlays(output_dir, image, args, ocr, detection.section_index, tiles)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote: {output_dir / 'metrics.json'}")
    return 0


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def _build_ocr(args: argparse.Namespace) -> Any:
    if args.ocr_engine == "easyocr":
        from umafactor.ocr import get_ocr

        return get_ocr()
    if args.ocr_engine == "paddleocr":
        from umafactor.recognition.paddle_ocr_adapter import PaddleFactorOCR

        cache_dir = _resolve_path(args.paddle_cache_dir) if args.paddle_cache_dir else None
        return PaddleFactorOCR(
            lang=args.paddle_lang,
            mode=args.paddle_mode,
            cache_dir=cache_dir,
            text_det_limit_side_len=args.paddle_det_limit_side_len,
            text_det_limit_type=args.paddle_det_limit_type,
            text_det_thresh=args.paddle_det_thresh,
            text_det_box_thresh=args.paddle_det_box_thresh,
            text_det_unclip_ratio=args.paddle_det_unclip_ratio,
            text_rec_score_thresh=args.paddle_rec_score_thresh,
        )
    raise ValueError(f"unknown OCR engine: {args.ocr_engine}")


def _paddle_params(args: argparse.Namespace) -> dict[str, object | None]:
    return {
        "text_det_limit_side_len": args.paddle_det_limit_side_len,
        "text_det_limit_type": args.paddle_det_limit_type,
        "text_det_thresh": args.paddle_det_thresh,
        "text_det_box_thresh": args.paddle_det_box_thresh,
        "text_det_unclip_ratio": args.paddle_det_unclip_ratio,
        "text_rec_score_thresh": args.paddle_rec_score_thresh,
    }


def _recognize_tiles_if_needed(
    image,
    tiles: Sequence[FactorListTile],
    ocr: Any,
    args: argparse.Namespace,
) -> list[FactorListTile]:
    if ocr is None:
        return list(tiles)
    return recognize_factor_list_tile_names(
        image,
        tiles,
        ocr,
        crop_variant=args.crop_variant,
        crop_target=args.ocr_crop_target,
    )


def _write_section_overlays(
    output_dir: Path,
    image,
    args: argparse.Namespace,
    ocr: Any,
    selected_section_index: int,
    selected_tiles: Sequence[FactorListTile],
) -> None:
    section_tiles: dict[int, tuple[str, Sequence[FactorListTile]]] = {}
    selected_role = selected_tiles[0].role if selected_tiles else _role_name(selected_section_index)
    section_tiles[selected_section_index] = (selected_role, selected_tiles)

    if args.overlay_sections == "all":
        for section_index in range(3):
            if section_index == selected_section_index:
                continue
            try:
                detection = detect_stitched_factor_list(image, section_index=section_index)
            except IndexError:
                continue
            tiles = _recognize_tiles_if_needed(image, detection.tiles, ocr, args)
            section_tiles[section_index] = (detection.role, tiles)

    for section_index, (role, tiles) in sorted(section_tiles.items()):
        _write_overlay(
            output_dir / f"{role}_overlay.png",
            image,
            tiles,
            crop_target=args.ocr_crop_target,
            crop_variant=args.crop_variant,
        )


def _role_name(section_index: int) -> str:
    if section_index == 0:
        return "parent"
    if section_index == 1:
        return "ancestor1"
    return "ancestor2"


def _write_detected_csv(path: Path, tiles: Sequence[FactorListTile]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "row_index", "col_index", "color", "star", "raw_name", "bbox"])
        for tile in tiles:
            writer.writerow(
                [
                    tile.order,
                    tile.row_index,
                    tile.col_index,
                    tile.color,
                    tile.star,
                    tile.raw_name,
                    " ".join(str(value) for value in tile.bbox),
                ]
            )


def _write_comparison_csv(
    path: Path,
    expected: Sequence[ExpectedOcrFactor],
    tiles: Sequence[FactorListTile],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order",
                "expected_name",
                "raw_name",
                "name_similarity",
                "name_match",
                "expected_star",
                "detected_star",
                "star_match",
                "color",
                "bbox",
            ]
        )
        for index in range(max(len(expected), len(tiles))):
            exp = expected[index] if index < len(expected) else None
            tile = tiles[index] if index < len(tiles) else None
            expected_name = exp.name if exp is not None else ""
            expected_star = exp.star if exp is not None else ""
            raw_name = tile.raw_name if tile is not None else ""
            detected_star = tile.star if tile is not None else ""
            name_similarity = (
                _name_similarity(raw_name, expected_name)
                if raw_name or expected_name
                else ""
            )
            writer.writerow(
                [
                    index,
                    expected_name,
                    raw_name,
                    f"{name_similarity:.6f}" if isinstance(name_similarity, float) else "",
                    raw_name == expected_name if raw_name or expected_name else "",
                    expected_star,
                    detected_star,
                    detected_star == expected_star if tile is not None and exp is not None else "",
                    tile.color if tile is not None else "",
                    " ".join(str(value) for value in tile.bbox) if tile is not None else "",
                ]
            )


def _write_summary(path: Path, report: dict[str, object]) -> None:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    lines = [
        "# Factor OCR Evaluation",
        "",
        f"- image: `{report['image_path']}`",
        f"- expected: `{report['expected_path']}`",
        f"- role: `{report['role']}`",
        f"- ocr_enabled: `{report['ocr_enabled']}`",
        f"- ocr_engine: `{report['ocr_engine']}`",
        f"- ocr_crop_target: `{report['ocr_crop_target']}`",
        f"- crop_variant: `{report['crop_variant']}`",
        f"- overlay_sections: `{report['overlay_sections']}`",
    ]
    if report.get("paddle_mode"):
        lines.extend(
            [
                f"- paddle_mode: `{report['paddle_mode']}`",
                f"- paddle_lang: `{report['paddle_lang']}`",
                f"- paddle_params: `{report['paddle_params']}`",
            ]
        )
    lines.extend(
        [
            "",
            "| metric | value |",
            "|---|---:|",
            f"| expected_count | {metrics['expected_count']} |",
            f"| detected_count | {metrics['detected_count']} |",
            f"| count_delta | {metrics['count_delta']} |",
            f"| star_accuracy | {_fmt(metrics['star_accuracy'])} |",
            f"| name_evaluated_count | {metrics['name_evaluated_count']} |",
            f"| blank_name_count | {metrics['blank_name_count']} |",
            f"| name_accuracy | {_fmt(metrics['name_accuracy'])} |",
            f"| name_similarity_mean | {_fmt(metrics['name_similarity_mean'])} |",
            f"| name_similarity_min | {_fmt(metrics['name_similarity_min'])} |",
            f"| name_correct | {metrics['name_correct']} |",
            "",
            "| similarity bucket | count | percent |",
            "|---|---:|---:|",
        ]
    )
    buckets = metrics["name_similarity_buckets"]
    percentages = metrics["name_similarity_bucket_percentages"]
    assert isinstance(buckets, dict)
    assert isinstance(percentages, dict)
    for label, count in buckets.items():
        lines.append(f"| {label} | {count} | {_fmt_percent(percentages[label])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overlay(
    path: Path,
    image,
    tiles: Sequence[FactorListTile],
    *,
    crop_target: str,
    crop_variant: str,
) -> None:
    panel_width = _overlay_panel_width(image.shape[1])
    overlay = cv2.copyMakeBorder(
        image,
        0,
        0,
        0,
        panel_width,
        cv2.BORDER_CONSTANT,
        value=(248, 248, 248),
    )
    colors = {
        "blue": (255, 0, 0),
        "red": (0, 0, 255),
        "green": (0, 180, 0),
        "white": (180, 180, 180),
    }
    cv2.line(
        overlay,
        (image.shape[1], 0),
        (image.shape[1], image.shape[0]),
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    for tile in tiles:
        x0, y0, x1, y1 = factor_list_ocr_region_bbox(
            image,
            tile,
            target=crop_target,
            variant=crop_variant,
        )
        color = colors.get(tile.color, (255, 255, 255))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay,
            f"#{tile.order} s{tile.star}",
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    overlay = _draw_overlay_text_panel(
        overlay,
        image.shape[1],
        tiles,
        colors,
        crop_target=crop_target,
        crop_variant=crop_variant,
    )
    cv2.imwrite(str(path), overlay)


def _overlay_panel_width(image_width: int) -> int:
    return max(360, int(round(image_width * 0.35)))


def _draw_overlay_text_panel(
    overlay,
    image_width: int,
    tiles: Sequence[FactorListTile],
    colors: dict[str, tuple[int, int, int]],
    *,
    crop_target: str,
    crop_variant: str,
):
    if Image is None or ImageDraw is None or ImageFont is None:
        return _draw_overlay_text_panel_cv2(
            overlay,
            image_width,
            tiles,
            colors,
            crop_target=crop_target,
            crop_variant=crop_variant,
        )

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = _load_overlay_font(max(13, int(round(overlay.shape[1] * 0.010))))
    line_height = _text_height(draw, font) + 4
    text_x = image_width + 12
    max_width = overlay.shape[1] - text_x - 8
    label_positions = _overlay_label_positions(
        overlay[:, :image_width],
        tiles,
        line_height,
        crop_target=crop_target,
        crop_variant=crop_variant,
    )
    for tile in tiles:
        _x0, y0, x1, _y1 = factor_list_ocr_region_bbox(
            overlay[:, :image_width],
            tile,
            target=crop_target,
            variant=crop_variant,
        )
        label_y = label_positions.get(tile.order, max(2, min(overlay.shape[0] - line_height - 2, y0)))
        color = colors.get(tile.color, (255, 255, 255))
        rgb_color = (color[2], color[1], color[0], 255)
        text = _trim_text_to_width(draw, _overlay_label(tile), font, max_width - 16)
        draw.rectangle(
            (text_x - 6, label_y - 1, overlay.shape[1] - 6, label_y + line_height - 1),
            fill=(255, 255, 255, 190),
        )
        draw.rectangle(
            (text_x - 2, label_y + 4, text_x + 8, label_y + line_height - 6),
            fill=rgb_color,
        )
        draw.text((text_x + 14, label_y), text, font=font, fill=(35, 35, 35, 255))
        draw.line(
            (x1, label_y + line_height // 2, image_width, label_y + line_height // 2),
            fill=rgb_color,
            width=1,
        )
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _draw_overlay_text_panel_cv2(
    overlay,
    image_width: int,
    tiles: Sequence[FactorListTile],
    colors: dict[str, tuple[int, int, int]],
    *,
    crop_target: str,
    crop_variant: str,
):
    label_positions = _overlay_label_positions(
        overlay[:, :image_width],
        tiles,
        18,
        crop_target=crop_target,
        crop_variant=crop_variant,
    )
    for tile in tiles:
        _x0, y0, _x1, _y1 = factor_list_ocr_region_bbox(
            overlay[:, :image_width],
            tile,
            target=crop_target,
            variant=crop_variant,
        )
        label_y = label_positions.get(tile.order, y0)
        color = colors.get(tile.color, (80, 80, 80))
        cv2.putText(
            overlay,
            _overlay_label(tile).encode("ascii", errors="replace").decode("ascii"),
            (image_width + 12, max(16, label_y + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _overlay_label_positions(
    image,
    tiles: Sequence[FactorListTile],
    line_height: int,
    *,
    crop_target: str,
    crop_variant: str,
) -> dict[int, int]:
    max_y = max(2, image.shape[0] - line_height - 2)
    entries: list[tuple[int, int, FactorListTile]] = []
    for tile in tiles:
        _x0, y0, _x1, _y1 = factor_list_ocr_region_bbox(
            image,
            tile,
            target=crop_target,
            variant=crop_variant,
        )
        entries.append((y0, tile.order, tile))

    positions: dict[int, int] = {}
    next_y = 2
    for y0, _order, tile in sorted(entries):
        label_y = max(2, min(max_y, y0))
        if label_y < next_y:
            label_y = next_y
        if label_y > max_y:
            label_y = max_y
        positions[tile.order] = label_y
        next_y = label_y + line_height + 2
    return positions


def _load_overlay_font(size: int):
    font_paths = [
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def _overlay_label(tile: FactorListTile) -> str:
    raw_name = tile.raw_name.strip() if tile.raw_name else "(blank)"
    return f"#{tile.order} star={tile.star} {raw_name}"


def _trim_text_to_width(draw, text: str, font, max_width: int) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    available = max(0, max_width - _text_width(draw, suffix, font))
    trimmed = text
    while trimmed and _text_width(draw, trimmed, font) > available:
        trimmed = trimmed[:-1]
    return f"{trimmed}{suffix}" if trimmed else suffix


def _text_width(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _text_height(draw, font) -> int:
    bbox = draw.textbbox((0, 0), "#0 s3", font=font)
    return bbox[3] - bbox[1]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_percent(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value * 100:.1f}%"
    return str(value)


def _name_similarity(actual: str, expected: str) -> float:
    if not actual and not expected:
        return 1.0
    return SequenceMatcher(None, actual, expected).ratio()


if __name__ == "__main__":
    raise SystemExit(main())
