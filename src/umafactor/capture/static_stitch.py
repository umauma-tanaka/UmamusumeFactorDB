"""Static screenshot scroll stitching experiments.

This module is intentionally experimental.  It provides an offline path for
evaluating scroll stitching ideas against fixed screenshot sequences before the
capture loop is connected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.geometry import Rect, Size
from .scraper_types import FrameOffset, ScrollFrame


@dataclass(frozen=True)
class DynamicRoiResult:
    rect: Rect
    mask: np.ndarray
    source: str

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.rect.y0 : self.rect.y1, self.rect.x0 : self.rect.x1]


@dataclass(frozen=True)
class BandMatch:
    band_y0: int
    match_y0: int
    delta_y: int
    score: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "band_y0": self.band_y0,
            "match_y0": self.match_y0,
            "delta_y": self.delta_y,
            "score": self.score,
        }


@dataclass(frozen=True)
class PairOffset:
    previous_frame_index: int
    frame_index: int
    delta_y: int
    confidence: float
    score: float
    inlier_count: int
    matches: tuple[BandMatch, ...]
    reject_reason: str = ""

    def to_frame_offset(self, absolute_y: int) -> FrameOffset:
        return FrameOffset(
            frame_index=self.frame_index,
            offset_y=absolute_y,
            confidence=self.confidence,
            source="dynamic_roi_template",
            inlier_count=self.inlier_count,
            reject_reason=self.reject_reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_frame_index": self.previous_frame_index,
            "frame_index": self.frame_index,
            "delta_y": self.delta_y,
            "confidence": self.confidence,
            "score": self.score,
            "inlier_count": self.inlier_count,
            "reject_reason": self.reject_reason,
            "matches": [match.to_dict() for match in self.matches],
        }


@dataclass(frozen=True)
class StaticStitchResult:
    image: np.ndarray
    roi: DynamicRoiResult
    offsets: tuple[FrameOffset, ...]
    pair_offsets: tuple[PairOffset, ...]
    seam_rows: tuple[int, ...]
    scrollbar_hint_enabled: bool = False
    scrollbar_thumb_centers: tuple[float | None, ...] = ()

    def to_metadata(self) -> dict[str, object]:
        return {
            "roi": self.roi.rect.as_tuple(),
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "offsets": [offset.to_dict() for offset in self.offsets],
            "pair_offsets": [pair.to_dict() for pair in self.pair_offsets],
            "seam_rows": list(self.seam_rows),
            "scrollbar_hint_enabled": self.scrollbar_hint_enabled,
            "scrollbar_thumb_centers": list(self.scrollbar_thumb_centers),
        }


def load_scroll_frames_from_dir(input_dir: Path) -> tuple[ScrollFrame, ...]:
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and not path.stem.startswith("expected_")
    )
    frames: list[ScrollFrame] = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"image is missing or unreadable: {path}")
        frames.append(
            ScrollFrame(
                image=image,
                frame_index=index,
                source_path=str(path),
                offset_y=None,
            )
        )
    if not frames:
        raise ValueError(f"no screenshot images found in {input_dir}")
    return tuple(frames)


def detect_dynamic_roi(frames: Sequence[ScrollFrame]) -> DynamicRoiResult:
    if len(frames) < 2:
        size = frames[0].size
        mask = np.full((size.height, size.width), 255, dtype=np.uint8)
        return DynamicRoiResult(Rect(0, 0, size.width, size.height), mask, "single_frame")

    masks = [_frame_diff_mask(a.image, b.image) for a, b in zip(frames, frames[1:])]
    combined = np.maximum.reduce(masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    connected = cv2.dilate(connected, kernel, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds = Size.from_image_shape(frames[0].image.shape)
    candidates: list[tuple[float, Rect]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect = Rect(x, y, x + w, y + h).clamp(bounds)
        if rect.width < bounds.width * 0.25 or rect.height < bounds.height * 0.10:
            continue
        area_score = float(rect.area)
        width_bonus = min(1.0, rect.width / max(1.0, bounds.width * 0.75))
        candidates.append((area_score * width_bonus, rect))

    projection_rect = _roi_from_projection(combined)
    if (
        projection_rect.width >= bounds.width * 0.45
        and projection_rect.height >= bounds.height * 0.15
    ):
        rect = projection_rect
    elif candidates:
        rect = max(candidates, key=lambda item: item[0])[1]
    else:
        rect = projection_rect

    margin_y = 2
    expanded = Rect(
        rect.x0,
        rect.y0 - margin_y,
        rect.x1,
        rect.y1 + margin_y,
    ).clamp(bounds)
    return DynamicRoiResult(expanded, combined, "dynamic_diff")


def stitch_static_scroll_frames(
    frames: Sequence[ScrollFrame],
    *,
    use_scrollbar_hint: bool = False,
) -> StaticStitchResult:
    if not frames:
        raise ValueError("frames is empty")

    roi = detect_dynamic_roi(frames)
    cropped = [roi.crop(frame.image) for frame in frames]
    if len(cropped) == 1:
        offset = FrameOffset(frame_index=frames[0].frame_index, offset_y=0, source="single_frame")
        return StaticStitchResult(
            image=cropped[0].copy(),
            roi=roi,
            offsets=(offset,),
            pair_offsets=(),
            seam_rows=(),
            scrollbar_hint_enabled=use_scrollbar_hint,
            scrollbar_thumb_centers=(_detect_scrollbar_thumb_center(cropped[0]),)
            if use_scrollbar_hint
            else (),
        )

    scrollbar_centers = (
        tuple(_detect_scrollbar_thumb_center(image) for image in cropped)
        if use_scrollbar_hint
        else ()
    )
    pair_offsets = tuple(
        estimate_pair_offset(
            cropped[index],
            cropped[index + 1],
            previous_frame_index=frames[index].frame_index,
            frame_index=frames[index + 1].frame_index,
            scrollbar_motion=(
                None
                if not use_scrollbar_hint
                or scrollbar_centers[index] is None
                or scrollbar_centers[index + 1] is None
                else scrollbar_centers[index + 1] - scrollbar_centers[index]  # type: ignore[operator]
            ),
        )
        for index in range(len(cropped) - 1)
    )

    absolute_offsets = [0]
    for pair in pair_offsets:
        if pair.delta_y <= 0:
            raise RuntimeError(
                f"non-positive scroll delta for frame {pair.frame_index}: {pair.reject_reason}"
            )
        absolute_offsets.append(absolute_offsets[-1] + pair.delta_y)

    canvas = cropped[0].copy()
    seam_rows: list[int] = []
    offsets: list[FrameOffset] = [
        FrameOffset(frame_index=frames[0].frame_index, offset_y=0, source="dynamic_roi_template")
    ]

    for index, image in enumerate(cropped[1:], start=1):
        absolute_y = absolute_offsets[index]
        seam_y = _choose_canvas_seam(canvas, image, absolute_y)
        if seam_y is None:
            copy_y0 = max(0, int(canvas.shape[0]) - absolute_y)
            if copy_y0 >= image.shape[0]:
                continue
            seam_rows.append(int(canvas.shape[0]))
            canvas = np.vstack([canvas, image[copy_y0:]])
            continue
        global_seam, image_seam = seam_y
        if image_seam >= image.shape[0]:
            continue
        seam_rows.append(global_seam)
        canvas = np.vstack([canvas[:global_seam], image[image_seam:]])
        offsets.append(pair_offsets[index - 1].to_frame_offset(absolute_y))

    return StaticStitchResult(
        image=canvas,
        roi=roi,
        offsets=tuple(offsets),
        pair_offsets=pair_offsets,
        seam_rows=tuple(seam_rows),
        scrollbar_hint_enabled=use_scrollbar_hint,
        scrollbar_thumb_centers=scrollbar_centers,
    )


def estimate_pair_offset(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    previous_frame_index: int,
    frame_index: int,
    scrollbar_motion: float | None = None,
) -> PairOffset:
    previous_match = _prepare_match_image(previous)
    current_match = _prepare_match_image(current)
    height = previous_match.shape[0]
    if height < 80:
        return PairOffset(
            previous_frame_index=previous_frame_index,
            frame_index=frame_index,
            delta_y=0,
            confidence=0.0,
            score=0.0,
            inlier_count=0,
            matches=(),
            reject_reason="roi too short for template matching",
        )

    overlap_matches = _scan_overlap_matches(
        previous_match,
        current_match,
        scrollbar_motion=scrollbar_motion,
    )
    if overlap_matches:
        best = overlap_matches[0]
        near_best = [
            match
            for match in overlap_matches
            if abs(match.delta_y - best.delta_y) <= max(8, int(round(height * 0.015)))
        ]
        spread = float(np.std([match.delta_y for match in near_best])) if len(near_best) > 1 else 0.0
        consistency = 1.0 / (1.0 + spread / 16.0)
        confidence = max(0.0, min(1.0, best.score * 1.5 * consistency))
        reject_reason = "" if confidence > 0.20 else "low confidence overlap match"
        return PairOffset(
            previous_frame_index=previous_frame_index,
            frame_index=frame_index,
            delta_y=best.delta_y,
            confidence=confidence,
            score=best.score,
            inlier_count=len(near_best),
            matches=tuple(overlap_matches[:5]),
            reject_reason=reject_reason,
        )

    band_height = max(64, min(int(round(height * 0.18)), height // 3))
    starts = [
        int(round(height * ratio))
        for ratio in (0.35, 0.45, 0.55, 0.65, 0.75)
        if int(round(height * ratio)) + band_height < height
    ]
    matches: list[BandMatch] = []
    for y0 in starts:
        template = previous_match[y0 : y0 + band_height]
        if template.std() < 5.0:
            continue
        result = cv2.matchTemplate(current_match, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, best_loc = cv2.minMaxLoc(result)
        match_y0 = int(best_loc[1])
        delta_y = int(y0 - match_y0)
        if 0 < delta_y < height - 32:
            matches.append(
                BandMatch(
                    band_y0=int(y0),
                    match_y0=match_y0,
                    delta_y=delta_y,
                    score=float(score),
                )
            )

    if not matches:
        return PairOffset(
            previous_frame_index=previous_frame_index,
            frame_index=frame_index,
            delta_y=0,
            confidence=0.0,
            score=0.0,
            inlier_count=0,
            matches=(),
            reject_reason="no positive template matches",
        )

    deltas = np.array([match.delta_y for match in matches], dtype=np.float32)
    median = float(np.median(deltas))
    tolerance = max(8.0, height * 0.025)
    inliers = [match for match in matches if abs(match.delta_y - median) <= tolerance]
    if not inliers:
        inliers = [max(matches, key=lambda match: match.score)]

    weights = np.array([max(0.001, match.score) for match in inliers], dtype=np.float32)
    values = np.array([match.delta_y for match in inliers], dtype=np.float32)
    delta_y = int(round(float(np.average(values, weights=weights))))
    score = float(np.mean([match.score for match in inliers]))
    spread = float(np.std(values)) if len(values) > 1 else 0.0
    consistency = 1.0 / (1.0 + spread / 16.0)
    confidence = max(0.0, min(1.0, score * consistency * min(1.0, len(inliers) / 2.0)))
    reject_reason = "" if confidence > 0.15 else "low confidence template matches"

    return PairOffset(
        previous_frame_index=previous_frame_index,
        frame_index=frame_index,
        delta_y=delta_y,
        confidence=confidence,
        score=score,
        inlier_count=len(inliers),
        matches=tuple(matches),
        reject_reason=reject_reason,
    )


def _frame_diff_mask(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    if previous.shape != current.shape:
        raise ValueError("all frames must have the same shape")
    diff = cv2.absdiff(previous, current)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    active = gray[gray > 0]
    threshold = 12.0 if active.size == 0 else max(12.0, float(np.percentile(active, 82)))
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return mask


def _roi_from_projection(mask: np.ndarray) -> Rect:
    height, width = mask.shape[:2]
    rows = mask.mean(axis=1) / 255.0
    y0, y1 = _longest_active_run(rows, min_ratio=0.02, min_len=max(32, height // 12))
    if y1 <= y0:
        return Rect(0, 0, width, height)
    cols = mask[y0:y1].mean(axis=0) / 255.0
    x0, x1 = _longest_active_run(cols, min_ratio=0.02, min_len=max(32, width // 4))
    if y1 <= y0 or x1 <= x0:
        return Rect(0, 0, width, height)
    expand_left = int(round(width * 0.05))
    expand_right = int(round(width * 0.03))
    return Rect(
        max(0, x0 - expand_left),
        y0,
        min(width, x1 + expand_right),
        y1,
    )


def _longest_active_run(values: np.ndarray, *, min_ratio: float, min_len: int) -> tuple[int, int]:
    active = values >= min_ratio
    best = (0, 0)
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_len and index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and len(values) - start >= min_len:
        if len(values) - start > best[1] - best[0]:
            best = (start, len(values))
    return best


def _prepare_match_image(image: np.ndarray) -> np.ndarray:
    _, width = image.shape[:2]
    x0 = int(round(width * 0.08))
    x1 = int(round(width * 0.92))
    crop = image[:, x0:x1] if x1 > x0 else image
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 135)
    dark = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
    return cv2.max(edges, dark)


def _scan_overlap_matches(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    scrollbar_motion: float | None = None,
) -> list[BandMatch]:
    height = previous.shape[0]
    min_overlap = max(80, int(round(height * 0.18)))
    max_delta = height - min_overlap
    if max_delta <= 0:
        return []

    coarse: list[BandMatch] = []
    for delta_y in range(8, max_delta + 1, 4):
        score = _edge_overlap_score(previous, current, delta_y)
        if score > 0.0:
            coarse.append(
                BandMatch(
                    band_y0=delta_y,
                    match_y0=0,
                    delta_y=delta_y,
                    score=score,
                )
            )
    if not coarse:
        return []

    refine_seeds = sorted(coarse, key=lambda match: match.score, reverse=True)[:8]
    refined_by_delta: dict[int, BandMatch] = {}
    for seed in refine_seeds:
        for delta_y in range(
            max(8, seed.delta_y - 8),
            min(max_delta, seed.delta_y + 8) + 1,
        ):
            score = _edge_overlap_score(previous, current, delta_y)
            current_best = refined_by_delta.get(delta_y)
            if current_best is None or score > current_best.score:
                refined_by_delta[delta_y] = BandMatch(
                    band_y0=delta_y,
                    match_y0=0,
                    delta_y=delta_y,
                    score=score,
                )
    matches = list(refined_by_delta.values())
    return sorted(
        matches,
        key=lambda match: _overlap_rank_score(match, height, scrollbar_motion),
        reverse=True,
    )


def _edge_overlap_score(previous: np.ndarray, current: np.ndarray, delta_y: int) -> float:
    height = previous.shape[0]
    if delta_y <= 0 or delta_y >= height:
        return 0.0
    prev_overlap = previous[delta_y:]
    curr_overlap = current[: height - delta_y]
    active = (prev_overlap > 0) | (curr_overlap > 0)
    active_count = int(active.sum())
    if active_count < 500:
        return 0.0
    diff = np.abs(prev_overlap.astype(np.float32) - curr_overlap.astype(np.float32))
    return float(1.0 - np.mean(diff[active]) / 255.0)


def _overlap_rank_score(
    match: BandMatch,
    roi_height: int,
    scrollbar_motion: float | None,
) -> float:
    if scrollbar_motion is None or abs(scrollbar_motion) < 1.0:
        return match.score
    # The scrollbar hint is deliberately weak and optional: it only helps break
    # ties away from tiny repeated-row matches when the thumb visibly moved.
    if scrollbar_motion > 0:
        size_bonus = min(0.08, 0.08 * match.delta_y / max(1, roi_height))
    else:
        size_bonus = min(0.08, 0.08 * (roi_height - match.delta_y) / max(1, roi_height))
    return match.score + size_bonus


def _choose_canvas_seam(
    canvas: np.ndarray,
    image: np.ndarray,
    absolute_y: int,
) -> tuple[int, int] | None:
    canvas_height = int(canvas.shape[0])
    image_height = int(image.shape[0])
    overlap_start = max(0, absolute_y)
    overlap_end = min(canvas_height, absolute_y + image_height)
    if overlap_end <= overlap_start:
        return None

    image_y0 = overlap_start - absolute_y
    image_y1 = overlap_end - absolute_y
    overlap_h = image_y1 - image_y0
    guard = min(max(6, image_height // 40), max(0, overlap_h // 3))
    candidate_start = image_y0 + guard
    candidate_end = image_y1 - guard
    if candidate_end <= candidate_start:
        candidate_start = image_y0
        candidate_end = image_y1

    canvas_overlap = canvas[overlap_start:overlap_end]
    image_overlap = image[image_y0:image_y1]
    canvas_f = canvas_overlap.astype(np.float32)
    image_f = image_overlap.astype(np.float32)
    diff = np.mean(np.abs(canvas_f - image_f), axis=(1, 2))
    gray = cv2.cvtColor(image_overlap, cv2.COLOR_BGR2GRAY)
    row_variance = gray.std(axis=1)

    best_score: float | None = None
    best_image_y = candidate_start
    for image_y in range(candidate_start, candidate_end):
        local_y = image_y - image_y0
        window0 = max(0, local_y - 2)
        window1 = min(overlap_h, local_y + 3)
        seam_diff = float(np.mean(diff[window0:window1]))
        # Avoid preferring completely flat white spacer rows unless they are the
        # only good match. Those rows are visually perceived as gaps.
        flat_penalty = 8.0 if float(row_variance[local_y]) < 4.0 else 0.0
        edge_penalty = 2.0 * min(
            abs(image_y - candidate_start),
            abs(candidate_end - image_y),
        ) / max(1, candidate_end - candidate_start)
        score = seam_diff + flat_penalty - edge_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_image_y = image_y

    return absolute_y + best_image_y, best_image_y


def _detect_scrollbar_thumb_center(image: np.ndarray) -> float | None:
    height, width = image.shape[:2]
    x0 = max(0, int(round(width * 0.94)))
    strip = image[:, x0:width]
    if strip.size == 0:
        return None
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask = ((gray < 150) & (hsv[:, :, 1] > 15)).astype(np.uint8)
    row_score = mask.mean(axis=1)
    runs: list[tuple[int, int, float]] = []
    start: int | None = None
    for index, value in enumerate(row_score):
        active = value > 0.08
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= max(8, height // 30):
                runs.append((start, index, float(row_score[start:index].mean())))
            start = None
    if start is not None and height - start >= max(8, height // 30):
        runs.append((start, height, float(row_score[start:].mean())))
    if not runs:
        return None
    y0, y1, _ = max(runs, key=lambda run: (run[1] - run[0]) * run[2])
    return float((y0 + y1) / 2.0)
