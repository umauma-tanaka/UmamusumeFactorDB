from __future__ import annotations

import numpy as np
import pytest

from umafactor.capture import FrameOffset, ScrollAreaStitcher, ScrollFrame, stitch_single_image
from umafactor.capture.scraper_types import StitchPlacement
from umafactor.evaluation.stitch_metrics import (
    compute_band_metrics,
    compute_seam_discontinuity_score,
    evaluate_stitch_result,
    summarize_offset_errors,
)


def _image(value: int, *, height: int = 3, width: int = 2) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_compute_band_metrics_reports_overlap_and_gap() -> None:
    placements = [
        StitchPlacement(frame_index=0, x0=0, y0=0, x1=2, y1=4),
        StitchPlacement(frame_index=1, x0=0, y0=3, x1=2, y1=6),
        StitchPlacement(frame_index=2, x0=0, y0=8, x1=2, y1=10),
    ]

    result = compute_band_metrics(placements)

    assert result.duplicate_band_px == 1
    assert result.missing_band_px == 2
    assert result.to_dict() == {
        "duplicate_band_px": 1,
        "missing_band_px": 2,
    }


def test_compute_seam_discontinuity_score_uses_placement_starts() -> None:
    image = np.zeros((4, 2, 3), dtype=np.uint8)
    image[2:] = 100
    placements = [
        StitchPlacement(frame_index=0, x0=0, y0=0, x1=2, y1=2),
        StitchPlacement(frame_index=1, x0=0, y0=2, x1=2, y1=4),
    ]

    assert compute_seam_discontinuity_score(image, placements) == pytest.approx(100.0)


def test_summarize_offset_errors_reports_missing_and_extra_frames() -> None:
    actual = [
        FrameOffset(frame_index=0, offset_y=0),
        FrameOffset(frame_index=2, offset_y=7),
    ]

    metrics = summarize_offset_errors(actual, {0: 0, 1: 3, 2: 5})

    assert metrics == {
        "offset_error_px": 2,
        "offset_error_mean_px": 1.0,
        "offset_missing_frames": [1],
        "offset_extra_frames": [],
    }


def test_evaluate_stitch_result_combines_stitch_metrics() -> None:
    frames = [
        ScrollFrame(_image(10, height=2), frame_index=0, offset_y=0),
        ScrollFrame(_image(20, height=2), frame_index=1, offset_y=2),
    ]
    result = ScrollAreaStitcher().stitch(frames)

    metrics = evaluate_stitch_result(
        result,
        expected_offsets={0: 0, 1: 2},
        expected_size=(2, 4),
    )

    assert metrics["offset_error_px"] == 0
    assert metrics["duplicate_band_px"] == 0
    assert metrics["missing_band_px"] == 0
    assert metrics["seam_discontinuity_score"] == pytest.approx(10.0)
    assert metrics["stitch_confidence"] == 1.0
    assert metrics["size_matches"] is True
    assert metrics["stitch_skipped"] is False
    assert metrics["stitch_mode"] == "provided_offsets"


def test_evaluate_stitch_result_reports_single_image_skip() -> None:
    result = stitch_single_image(_image(5, height=2), source_path="full.png")

    metrics = evaluate_stitch_result(result, expected_size=(2, 2))

    assert metrics["stitch_skipped"] is True
    assert metrics["stitch_mode"] == "single_frame"
    assert metrics["size_matches"] is True
