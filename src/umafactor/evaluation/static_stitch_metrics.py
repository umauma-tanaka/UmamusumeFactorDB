"""Metrics for offline static screenshot stitching experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..capture.static_stitch import StaticStitchResult
from ..core.geometry import Rect


@dataclass(frozen=True)
class StaticStitchMetrics:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def evaluate_static_stitch_success(
    result: StaticStitchResult,
    *,
    frame_count: int,
    expected_image: np.ndarray | None = None,
) -> StaticStitchMetrics:
    roi = result.roi.rect
    image = result.image
    pair_offsets = result.pair_offsets
    deltas = [pair.delta_y for pair in pair_offsets]
    confidences = [pair.confidence for pair in pair_offsets]
    scores = [pair.score for pair in pair_offsets]
    inliers = [pair.inlier_count for pair in pair_offsets]
    overlap_px = [max(0, roi.height - delta) for delta in deltas]

    values: dict[str, Any] = {
        "status": "success",
        "frame_count": frame_count,
        "roi_bbox": list(roi.as_tuple()),
        "roi_width": roi.width,
        "roi_height": roi.height,
        "stitched_width": int(image.shape[1]),
        "stitched_height": int(image.shape[0]),
        "content_gain_px": max(0, int(image.shape[0]) - roi.height),
        "coverage_ratio": _coverage_ratio(int(image.shape[0]), roi.height, frame_count),
        "black_pixel_ratio": _black_pixel_ratio(image),
        "seam_discontinuity_score": compute_seam_discontinuity_score(
            image,
            result.seam_rows,
        ),
        "seam_count": len(result.seam_rows),
        "offset_monotonic": _is_monotonic([offset.offset_y for offset in result.offsets]),
        "pair_count": len(pair_offsets),
        "pair_reject_count": sum(1 for pair in pair_offsets if pair.reject_reason),
        "scrollbar_hint_enabled": result.scrollbar_hint_enabled,
        "scrollbar_thumb_detected_count": sum(
            1 for center in result.scrollbar_thumb_centers if center is not None
        ),
        "offsets": [offset.to_dict() for offset in result.offsets],
        "pair_offsets": [pair.to_dict() for pair in pair_offsets],
    }
    values.update(_series_metrics("delta_y", deltas))
    values.update(_series_metrics("overlap_px", overlap_px))
    values.update(_series_metrics("match_confidence", confidences))
    values.update(_series_metrics("match_score", scores))
    values.update(_series_metrics("match_inlier_count", inliers))
    if expected_image is not None:
        values.update(evaluate_reference_coverage(image, expected_image))
    return StaticStitchMetrics(values)


def evaluate_static_stitch_failure(
    *,
    algorithm: str,
    frame_count: int,
    error: BaseException | str,
    roi: Rect | None = None,
) -> StaticStitchMetrics:
    values: dict[str, Any] = {
        "status": "failed",
        "algorithm": algorithm,
        "frame_count": frame_count,
        "error": str(error),
    }
    if roi is not None:
        values.update(
            {
                "roi_bbox": list(roi.as_tuple()),
                "roi_width": roi.width,
                "roi_height": roi.height,
            }
        )
    return StaticStitchMetrics(values)


def compute_seam_discontinuity_score(image: np.ndarray, seam_rows: tuple[int, ...]) -> float:
    if not seam_rows:
        return 0.0
    image_f = image.astype(np.float32)
    scores: list[float] = []
    for y in seam_rows:
        if 0 < y < image.shape[0]:
            scores.append(float(np.mean(np.abs(image_f[y] - image_f[y - 1]))))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_reference_coverage(
    candidate: np.ndarray,
    expected: np.ndarray,
) -> dict[str, Any]:
    """Estimate missing/duplicate content against an accepted stitched image.

    The metric samples horizontal bands from the candidate image, matches them on
    the expected content axis, and compares candidate progress with expected
    progress.  Larger expected progress indicates missing content.  Larger
    candidate progress indicates duplicated content.
    """

    if candidate.size == 0 or expected.size == 0:
        return _empty_reference_metrics(candidate, expected)

    candidate_feat, expected_feat = _reference_feature_images(candidate, expected)
    candidate_h = int(candidate_feat.shape[0])
    expected_h = int(expected_feat.shape[0])
    if candidate_h < 16 or expected_h < 16:
        return _empty_reference_metrics(candidate, expected)

    band_h = max(16, min(96, min(candidate_h, expected_h) // 45))
    stride = max(12, band_h * 2)
    matches: list[tuple[int, int, float]] = []
    for cand_y in range(0, max(1, candidate_h - band_h + 1), stride):
        band = candidate_feat[cand_y : cand_y + band_h]
        if float(band.std()) < 2.0:
            continue
        result = cv2.matchTemplate(expected_feat, band, cv2.TM_CCOEFF_NORMED)
        _, score, _, best_loc = cv2.minMaxLoc(result)
        if score >= 0.35:
            matches.append((cand_y, int(best_loc[1]), float(score)))

    if len(matches) < 2:
        return {
            **_empty_reference_metrics(candidate, expected),
            "reference_matched_band_count": len(matches),
            "reference_match_score_mean": float(np.mean([m[2] for m in matches]))
            if matches
            else 0.0,
        }

    missing_px = 0
    duplicate_px = 0
    order_violations = 0
    tolerance = max(6, band_h // 2)
    sorted_matches = sorted(matches, key=lambda item: item[0])
    previous_cand_y, previous_ref_y, _ = sorted_matches[0]
    for cand_y, ref_y, _ in sorted_matches[1:]:
        cand_delta = cand_y - previous_cand_y
        ref_delta = ref_y - previous_ref_y
        if ref_delta < -tolerance:
            order_violations += 1
        elif ref_delta - cand_delta > tolerance:
            missing_px += ref_delta - cand_delta
        elif cand_delta - ref_delta > tolerance:
            duplicate_px += cand_delta - ref_delta
        previous_cand_y, previous_ref_y = cand_y, ref_y

    first_cand_y, first_ref_y, _ = sorted_matches[0]
    last_cand_y, last_ref_y, _ = sorted_matches[-1]
    head_delta = first_ref_y - first_cand_y
    if head_delta > tolerance:
        missing_px += head_delta
    elif head_delta < -tolerance:
        duplicate_px += abs(head_delta)

    candidate_tail = candidate_h - (last_cand_y + band_h)
    expected_tail = expected_h - (last_ref_y + band_h)
    tail_delta = expected_tail - candidate_tail
    if tail_delta > tolerance:
        missing_px += tail_delta
    elif tail_delta < -tolerance:
        duplicate_px += abs(tail_delta)

    match_scores = [score for _, _, score in sorted_matches]
    expected_scale_y = expected.shape[0] / max(1, expected_h)
    missing_px = int(round(max(0, missing_px) * expected_scale_y))
    duplicate_px = int(round(max(0, duplicate_px) * expected_scale_y))
    return {
        "reference_width": int(expected.shape[1]),
        "reference_height": int(expected.shape[0]),
        "reference_size_delta_width_px": int(candidate.shape[1] - expected.shape[1]),
        "reference_size_delta_height_px": int(candidate.shape[0] - expected.shape[0]),
        "reference_matched_band_count": len(sorted_matches),
        "reference_match_score_mean": float(np.mean(match_scores)),
        "reference_match_score_min": float(np.min(match_scores)),
        "reference_missing_px": missing_px,
        "reference_duplicate_px": duplicate_px,
        "reference_content_error_px": missing_px + duplicate_px,
        "reference_order_violation_count": order_violations,
        "reference_coverage_ratio": max(0.0, 1.0 - missing_px / max(1, expected.shape[0])),
    }


def _series_metrics(prefix: str, values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    arr = np.array(values, dtype=np.float32)
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_mean": float(arr.mean()),
    }


def _coverage_ratio(stitched_height: int, roi_height: int, frame_count: int) -> float:
    denominator = max(1, roi_height * frame_count)
    return float(stitched_height / denominator)


def _black_pixel_ratio(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    if image.ndim == 3:
        black = np.all(image == 0, axis=2)
    else:
        black = image == 0
    return float(np.mean(black))


def _is_monotonic(values: list[int]) -> bool:
    return all(b >= a for a, b in zip(values, values[1:]))


def _reference_feature_images(
    candidate: np.ndarray,
    expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_width = min(420, int(expected.shape[1]))
    candidate_feat = _reference_feature_image(candidate, target_width)
    expected_feat = _reference_feature_image(expected, target_width)
    return candidate_feat, expected_feat


def _reference_feature_image(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] != target_width:
        new_h = max(1, int(round(gray.shape[0] * target_width / gray.shape[1])))
        gray = cv2.resize(gray, (target_width, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 135)
    dark = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
    return cv2.max(edges, dark)


def _empty_reference_metrics(candidate: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    return {
        "reference_width": int(expected.shape[1]) if expected.size else 0,
        "reference_height": int(expected.shape[0]) if expected.size else 0,
        "reference_size_delta_width_px": int(candidate.shape[1] - expected.shape[1])
        if candidate.size and expected.size
        else 0,
        "reference_size_delta_height_px": int(candidate.shape[0] - expected.shape[0])
        if candidate.size and expected.size
        else 0,
        "reference_matched_band_count": 0,
        "reference_match_score_mean": 0.0,
        "reference_match_score_min": 0.0,
        "reference_missing_px": 0,
        "reference_duplicate_px": 0,
        "reference_content_error_px": 0,
        "reference_order_violation_count": 0,
        "reference_coverage_ratio": 0.0,
    }
