from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from umafactor.capture.scraper_types import ScrollFrame
from umafactor.capture.static_stitch import (
    detect_dynamic_roi,
    estimate_pair_offset,
    load_scroll_frames_from_dir,
    stitch_static_scroll_frames,
)
from umafactor.evaluation.static_stitch_metrics import (
    evaluate_reference_coverage,
    evaluate_static_stitch_success,
)


def _document(*, height: int = 260, width: int = 100) -> np.ndarray:
    image = np.full((height, width, 3), 238, dtype=np.uint8)
    for row, y in enumerate(range(8, height - 12, 22)):
        color = (40 + row * 7 % 160, 40 + row * 13 % 160, 40 + row * 17 % 160)
        cv2.rectangle(image, (8, y), (width - 8, y + 14), color, 1)
        cv2.putText(
            image,
            f"R{row:02d}",
            (14, y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return image


def _screen_frames(offsets: list[int]) -> tuple[ScrollFrame, ...]:
    doc = _document()
    viewport_h = 120
    frames: list[ScrollFrame] = []
    for index, offset in enumerate(offsets):
        screen = doc[offset : offset + viewport_h].copy()
        frames.append(ScrollFrame(screen, frame_index=index))
    return tuple(frames)


def test_estimate_pair_offset_uses_full_overlap_score() -> None:
    doc = _document()
    previous = doc[0:120]
    current = doc[37:157]

    pair = estimate_pair_offset(previous, current, previous_frame_index=0, frame_index=1)

    assert abs(pair.delta_y - 37) <= 1
    assert pair.confidence > 0.5
    assert pair.inlier_count >= 1


def test_stitch_static_scroll_frames_estimates_offsets() -> None:
    frames = _screen_frames([0, 37, 74])

    result = stitch_static_scroll_frames(frames)

    offsets = [offset.offset_y for offset in result.offsets]
    assert len(offsets) == 3
    assert abs(offsets[1] - 37) <= 1
    assert abs(offsets[2] - 74) <= 2
    assert result.image.shape[0] >= result.roi.rect.height + 72


def test_evaluate_static_stitch_success_reports_core_metrics() -> None:
    result = stitch_static_scroll_frames(_screen_frames([0, 45]))

    metrics = evaluate_static_stitch_success(result, frame_count=2).to_dict()

    assert metrics["status"] == "success"
    assert metrics["offset_monotonic"] is True
    assert metrics["pair_count"] == 1
    assert metrics["content_gain_px"] > 0


def test_reference_coverage_accepts_identical_image() -> None:
    expected = _document(height=340, width=140)

    metrics = evaluate_reference_coverage(expected.copy(), expected)

    assert metrics["reference_missing_px"] == 0
    assert metrics["reference_duplicate_px"] == 0
    assert metrics["reference_content_error_px"] == 0
    assert metrics["reference_match_score_mean"] > 0.95


def test_reference_coverage_detects_missing_content() -> None:
    expected = _document(height=420, width=140)
    candidate = np.vstack([expected[:170], expected[230:]])

    metrics = evaluate_reference_coverage(candidate, expected)

    assert metrics["reference_missing_px"] > 0
    assert metrics["reference_duplicate_px"] == 0
    assert metrics["reference_content_error_px"] == metrics["reference_missing_px"]


def test_reference_coverage_detects_duplicate_content() -> None:
    expected = _document(height=420, width=140)
    candidate = np.vstack([expected[:210], expected[170:210], expected[210:]])

    metrics = evaluate_reference_coverage(candidate, expected)

    assert metrics["reference_duplicate_px"] > 0
    assert metrics["reference_missing_px"] == 0
    assert metrics["reference_content_error_px"] == metrics["reference_duplicate_px"]


def test_stitch_static_scroll_frames_scales_with_image_size() -> None:
    frames = _screen_frames([0, 37, 74])
    scaled_frames = tuple(
        ScrollFrame(
            image=cv2.resize(frame.image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST),
            frame_index=frame.frame_index,
            source_path=frame.source_path,
            offset_y=frame.offset_y,
        )
        for frame in frames
    )

    result = stitch_static_scroll_frames(scaled_frames)

    offsets = [offset.offset_y for offset in result.offsets]
    assert len(offsets) == 3
    assert abs(offsets[1] - 74) <= 2
    assert abs(offsets[2] - 148) <= 4


def test_detect_dynamic_roi_uses_contour_when_projection_falls_back_to_full_screen() -> None:
    base = np.full((500, 1000, 3), 240, dtype=np.uint8)
    previous = base.copy()
    current = base.copy()
    for x0 in (260, 390, 520):
        cv2.rectangle(previous, (x0, 180), (x0 + 90, 330), (80, 80, 80), -1)
        cv2.rectangle(current, (x0, 180), (x0 + 90, 330), (150, 150, 150), -1)
    frames = (
        ScrollFrame(previous, frame_index=0),
        ScrollFrame(current, frame_index=1),
    )

    roi = detect_dynamic_roi(frames)

    assert roi.rect.width < previous.shape[1]
    assert roi.rect.height < previous.shape[0]
    assert 220 <= roi.rect.x0 <= 280
    assert 600 <= roi.rect.x1 <= 660


def test_load_scroll_frames_ignores_expected_images() -> None:
    case_dir = Path("outputs") / "test_static_stitch" / "expected_filter"
    case_dir.mkdir(parents=True, exist_ok=True)
    frame = np.full((20, 20, 3), 128, dtype=np.uint8)
    expected = np.full((40, 20, 3), 192, dtype=np.uint8)
    assert cv2.imwrite(str(case_dir / "000.png"), frame)
    assert cv2.imwrite(str(case_dir / "001.png"), frame)
    assert cv2.imwrite(str(case_dir / "expected_stitched.png"), expected)

    frames = load_scroll_frames_from_dir(case_dir)

    assert len(frames) == 2
    assert all(frame.image.shape == (20, 20, 3) for frame in frames)
