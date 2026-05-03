"""Quantitative metrics for scroll stitching outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..capture.scraper_types import FrameOffset, StitchPlacement, StitchResult


@dataclass(frozen=True)
class StitchBandMetrics:
    duplicate_band_px: int
    missing_band_px: int

    def to_dict(self) -> dict[str, int]:
        return {
            "duplicate_band_px": self.duplicate_band_px,
            "missing_band_px": self.missing_band_px,
        }


def compute_band_metrics(placements: Sequence[StitchPlacement]) -> StitchBandMetrics:
    if len(placements) < 2:
        return StitchBandMetrics(duplicate_band_px=0, missing_band_px=0)

    sorted_placements = sorted(placements, key=lambda placement: placement.y0)
    duplicate = 0
    missing = 0
    previous_end = sorted_placements[0].y1
    for placement in sorted_placements[1:]:
        if placement.y0 < previous_end:
            duplicate += previous_end - placement.y0
        elif placement.y0 > previous_end:
            missing += placement.y0 - previous_end
        previous_end = max(previous_end, placement.y1)
    return StitchBandMetrics(duplicate_band_px=duplicate, missing_band_px=missing)


def compute_seam_discontinuity_score(
    image: np.ndarray,
    placements: Sequence[StitchPlacement],
) -> float:
    seam_rows = sorted(
        {
            placement.y0
            for placement in placements
            if 0 < placement.y0 < image.shape[0]
        }
    )
    if not seam_rows:
        return 0.0

    scores: list[float] = []
    image_f = image.astype(np.float32)
    for y in seam_rows:
        scores.append(float(np.mean(np.abs(image_f[y] - image_f[y - 1]))))
    return float(np.mean(scores)) if scores else 0.0


def summarize_offset_errors(
    actual: Sequence[FrameOffset],
    expected_offsets: Mapping[int, int],
) -> dict[str, Any]:
    errors: list[int] = []
    actual_by_frame = {offset.frame_index: offset.offset_y for offset in actual}
    missing = sorted(set(expected_offsets) - set(actual_by_frame))
    extra = sorted(set(actual_by_frame) - set(expected_offsets))
    for frame_index, expected_y in expected_offsets.items():
        if frame_index in actual_by_frame:
            errors.append(abs(actual_by_frame[frame_index] - expected_y))

    return {
        "offset_error_px": max(errors) if errors else 0,
        "offset_error_mean_px": float(sum(errors) / len(errors)) if errors else 0.0,
        "offset_missing_frames": missing,
        "offset_extra_frames": extra,
    }


def evaluate_stitch_result(
    result: StitchResult,
    *,
    expected_offsets: Mapping[int, int] | None = None,
    expected_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    band_metrics = compute_band_metrics(result.placements)
    offset_metrics = (
        summarize_offset_errors(result.offsets, expected_offsets)
        if expected_offsets is not None
        else {
            "offset_error_px": 0,
            "offset_error_mean_px": 0.0,
            "offset_missing_frames": [],
            "offset_extra_frames": [],
        }
    )
    width = int(result.image.shape[1])
    height = int(result.image.shape[0])
    expected_width, expected_height = expected_size if expected_size is not None else (width, height)
    confidences = [offset.confidence for offset in result.offsets]

    return {
        **offset_metrics,
        **band_metrics.to_dict(),
        "seam_discontinuity_score": compute_seam_discontinuity_score(
            result.image,
            result.placements,
        ),
        "stitch_confidence": min(confidences) if confidences else 0.0,
        "stitched_width": width,
        "stitched_height": height,
        "expected_width": int(expected_width),
        "expected_height": int(expected_height),
        "size_matches": (width, height) == (int(expected_width), int(expected_height)),
        "stitch_skipped": result.skipped,
        "stitch_mode": result.mode,
    }
