"""Evaluate static screenshot scroll stitching algorithms.

Example:
    python scripts/evaluate_static_stitch.py --input datasets/test_factor_01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from umafactor.capture.scraper_types import ScrollFrame  # noqa: E402
from umafactor.capture.static_stitch import (  # noqa: E402
    detect_dynamic_roi,
    load_scroll_frames_from_dir,
    stitch_static_scroll_frames,
)
from umafactor.capture.stitcher import ScrollAreaStitcher  # noqa: E402
from umafactor.evaluation.static_stitch_metrics import (  # noqa: E402
    evaluate_static_stitch_failure,
    evaluate_static_stitch_success,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate static scroll stitching.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "datasets" / "test_factor_01",
        help="Directory containing ordered screenshots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "stitch_eval" / "test_factor_01",
        help="Directory for stitched images and metrics.",
    )
    parser.add_argument(
        "--roi-limit",
        type=int,
        default=3,
        help="Number of cropped ROI debug images to write.",
    )
    parser.add_argument(
        "--use-scrollbar-hint",
        action="store_true",
        help="Use detected scrollbar thumb movement as a weak optional match hint.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help=(
            "Accepted stitched image for missing/duplicate evaluation. "
            "Defaults to expected_stitched.png in the input directory when present."
        ),
    )
    args = parser.parse_args()

    input_dir = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_path = _resolve_expected_path(input_dir, args.expected)
    expected_image = cv2.imread(str(expected_path)) if expected_path is not None else None
    if expected_path is not None and expected_image is None:
        raise FileNotFoundError(f"failed to load expected image: {expected_path}")

    frames = load_scroll_frames_from_dir(input_dir)
    roi = detect_dynamic_roi(frames)
    cv2.imwrite(str(output_dir / "dynamic_mask.png"), roi.mask)
    for frame in frames[: max(0, args.roi_limit)]:
        cv2.imwrite(
            str(output_dir / f"roi_{frame.frame_index:02d}.png"),
            roi.crop(frame.image),
        )

    report: dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "expected_path": str(expected_path) if expected_path is not None else None,
        "frame_count": len(frames),
        "source_paths": [frame.source_path for frame in frames],
        "algorithms": {},
    }

    cropped_frames = tuple(
        ScrollFrame(
            image=roi.crop(frame.image),
            frame_index=frame.frame_index,
            source_path=frame.source_path,
            offset_y=None,
        )
        for frame in frames
    )
    try:
        current = ScrollAreaStitcher().stitch(cropped_frames)
        cv2.imwrite(str(output_dir / "current_metadata_stitched.png"), current.image)
        report["algorithms"]["current_metadata"] = {
            "status": "success",
            "metadata": current.to_metadata(),
        }
    except Exception as exc:  # noqa: BLE001 - evaluation should capture failures.
        report["algorithms"]["current_metadata"] = evaluate_static_stitch_failure(
            algorithm="current_metadata",
            frame_count=len(frames),
            error=exc,
            roi=roi.rect,
        ).to_dict()

    try:
        if args.use_scrollbar_hint:
            prototype = stitch_static_scroll_frames(frames, use_scrollbar_hint=True)
        else:
            prototype = stitch_static_scroll_frames(frames)
        cv2.imwrite(str(output_dir / "prototype_stitched.png"), prototype.image)
        report["algorithms"]["dynamic_roi_template"] = {
            **evaluate_static_stitch_success(
                prototype,
                frame_count=len(frames),
                expected_image=expected_image,
            ).to_dict(),
            "metadata": prototype.to_metadata(),
        }
    except Exception as exc:  # noqa: BLE001 - evaluation should capture failures.
        report["algorithms"]["dynamic_roi_template"] = evaluate_static_stitch_failure(
            algorithm="dynamic_roi_template",
            frame_count=len(frames),
            error=exc,
            roi=roi.rect,
        ).to_dict()

    report_path = output_dir / "metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(output_dir / "summary.md", report)
    print(json.dumps(report["algorithms"], ensure_ascii=False, indent=2))
    print(f"wrote: {report_path}")
    return 0


def _resolve_expected_path(input_dir: Path, expected_arg: Path | None) -> Path | None:
    if expected_arg is not None:
        return expected_arg if expected_arg.is_absolute() else ROOT / expected_arg
    default_path = input_dir / "expected_stitched.png"
    return default_path if default_path.exists() else None


def _write_summary(path: Path, report: dict[str, object]) -> None:
    algorithms = report["algorithms"]
    assert isinstance(algorithms, dict)
    lines = [
        "# Static Stitch Evaluation",
        "",
        f"- input: `{report['input_dir']}`",
        f"- expected: `{report['expected_path']}`",
        f"- frames: {report['frame_count']}",
        "",
        "| algorithm | status | height | delta mean | confidence mean | seam score | missing px | duplicate px | ref score | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, raw_metrics in algorithms.items():
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        lines.append(
            (
                "| {name} | {status} | {height} | {delta} | {conf} | {seam} | "
                "{missing} | {duplicate} | {ref_score} | {error} |"
            ).format(
                name=name,
                status=metrics.get("status", ""),
                height=_fmt(metrics.get("stitched_height")),
                delta=_fmt(metrics.get("delta_y_mean")),
                conf=_fmt(metrics.get("match_confidence_mean")),
                seam=_fmt(metrics.get("seam_discontinuity_score")),
                missing=_fmt(metrics.get("reference_missing_px")),
                duplicate=_fmt(metrics.get("reference_duplicate_px")),
                ref_score=_fmt(metrics.get("reference_match_score_mean")),
                error=str(metrics.get("error", "")).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
