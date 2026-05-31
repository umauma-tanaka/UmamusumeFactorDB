"""Independent card body detector.

This module intentionally does not call the legacy factor-list detector,
Hough-circle based icon anchors, star detection, OCR, or Submission code.
It detects the horizontal card body mask first, fits the two-column UI grid,
and expands the body boxes to item crop boxes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class CardBodyDetectorOptions:
    saturation_threshold: int = 45
    gray_delta_l_threshold: int = 4
    min_component_area: int = 200
    min_body_aspect: float = 3.0
    max_body_aspect: float = 8.0
    pad_left_ratio: float = 0.02
    pad_right_ratio: float = 0.02
    pad_top_ratio: float = 0.05
    pad_top_pitch_gap_ratio: float = 0.45
    star_margin_body_ratio: float = 0.95
    vertical_guard_ratio: float = 0.08
    debug: bool = False
    analysis_x1_ratio: float = 0.03
    analysis_x2_ratio: float = 0.97
    analysis_y1_ratio: float = 0.0
    analysis_y2_ratio: float = 1.0
    color_value_threshold: int = 120
    gray_value_threshold: int = 145
    gray_saturation_max: int = 95
    min_grid_fill_body_ratio: float = 0.20


@dataclass(frozen=True)
class DetectedCardBody:
    source_image: str
    role: str | None
    row: int
    col: int
    body_bbox: BBox
    item_bbox: BBox
    confidence: float
    source: str
    valid: bool
    invalid_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image": self.source_image,
            "role": self.role,
            "row": self.row,
            "col": self.col,
            "body_bbox": list(self.body_bbox),
            "item_bbox": list(self.item_bbox),
            "confidence": self.confidence,
            "source": self.source,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True)
class CardBodyDetectionResult:
    image_size: tuple[int, int]
    column_ranges: dict[str, BBox]
    median_body_w: float
    median_body_h: float
    median_row_pitch: float | None
    cards: list[DetectedCardBody]
    analysis_roi: BBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "analysis_roi": list(self.analysis_roi),
            "column_ranges": {
                key: list(value)
                for key, value in self.column_ranges.items()
            },
            "median_body_w": self.median_body_w,
            "median_body_h": self.median_body_h,
            "median_row_pitch": self.median_row_pitch,
            "cards": [card.to_dict() for card in self.cards],
        }


@dataclass(frozen=True)
class CardBodyDetectionDebug:
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    x_projection: np.ndarray
    row_projection_left: np.ndarray
    row_projection_right: np.ndarray
    body_components: tuple[BBox, ...]
    left_bands: tuple[tuple[int, int, float], ...]
    right_bands: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class CardBodyDetectionRun:
    result: CardBodyDetectionResult
    debug: CardBodyDetectionDebug


@dataclass(frozen=True)
class _Band:
    y1: int
    y2: int
    score: float

    @property
    def center(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class _ManualItem:
    id: str
    image_id: str | None
    source_image: str | None
    role: str | None
    row: int
    col: int
    bbox: BBox
    tolerance_px: int


def detect_card_bodies(
    image: str | Path | np.ndarray,
    *,
    options: CardBodyDetectorOptions | None = None,
    source_image: str | None = None,
    role: str | None = None,
) -> CardBodyDetectionRun:
    """Detect card bodies and item bboxes with the mask/grid algorithm."""

    opts = options or CardBodyDetectorOptions()
    image_bgr, resolved_source = _load_image(image)
    source_name = source_image or (str(resolved_source) if resolved_source is not None else "")
    image_h, image_w = image_bgr.shape[:2]
    analysis_roi = _analysis_roi(image_bgr.shape, opts)
    raw_mask = _build_card_body_mask(image_bgr, analysis_roi, opts)
    clean_mask = _clean_card_body_mask(raw_mask, image_bgr.shape, opts)
    components = _extract_body_components(clean_mask, image_bgr.shape, opts)
    columns = _estimate_columns(clean_mask, components, image_bgr.shape, opts)
    left_range = columns["left"]
    right_range = columns["right"]
    left_projection = _row_projection(clean_mask, left_range)
    right_projection = _row_projection(clean_mask, right_range)
    left_bands = _detect_body_bands(left_projection, image_w, opts)
    right_bands = _detect_body_bands(right_projection, image_w, opts)
    row_centers = _merge_row_centers(left_bands, right_bands, image_w)
    body_h = _median_body_height(left_bands, right_bands)
    row_pitch = _median_row_pitch(row_centers)
    body_w = _median_column_width(columns)

    cards = _build_cards(
        source_name=source_name,
        role=role,
        image_shape=image_bgr.shape,
        columns=columns,
        left_bands=left_bands,
        right_bands=right_bands,
        row_centers=row_centers,
        median_body_h=body_h,
        median_row_pitch=row_pitch,
        clean_mask=clean_mask,
        options=opts,
    )
    result = CardBodyDetectionResult(
        image_size=(image_w, image_h),
        analysis_roi=analysis_roi,
        column_ranges={
            "left": (left_range[0], 0, left_range[1], image_h),
            "right": (right_range[0], 0, right_range[1], image_h),
        },
        median_body_w=body_w,
        median_body_h=body_h,
        median_row_pitch=row_pitch,
        cards=cards,
    )
    debug = CardBodyDetectionDebug(
        raw_mask=raw_mask,
        clean_mask=clean_mask,
        x_projection=clean_mask.mean(axis=0).astype(float),
        row_projection_left=left_projection,
        row_projection_right=right_projection,
        body_components=tuple(components),
        left_bands=tuple((band.y1, band.y2, band.score) for band in left_bands),
        right_bands=tuple((band.y1, band.y2, band.score) for band in right_bands),
    )
    return CardBodyDetectionRun(result=result, debug=debug)


def evaluate_card_bodies(
    result: CardBodyDetectionResult,
    expected_path: str | Path,
    *,
    image_path: str | Path | None = None,
    image_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate detected item bboxes against a manual bbox fixture."""

    manual_items = _load_manual_items(expected_path, image_path=image_path, image_id=image_id)
    if not manual_items:
        return {
            "manual_count": 0,
            "matched_count": 0,
            "mean_iou": None,
            "min_iou": None,
            "iou_below_0_75": [],
            "hard_failure_count": 0,
            "hard_failures": [],
        }

    detected_by_grid = {(card.row, card.col): card for card in result.cards}
    rows: list[dict[str, Any]] = []
    ious: list[float] = []
    hard_failures: list[dict[str, Any]] = []
    for item in manual_items:
        card = detected_by_grid.get((item.row, item.col))
        if card is None:
            row = {
                "id": item.id,
                "row": item.row,
                "col": item.col,
                "manual_bbox": list(item.bbox),
                "detected_bbox": None,
                "iou": None,
                "hard_failures": ["missing_detection"],
                "valid": False,
            }
            rows.append(row)
            hard_failures.append(row)
            continue
        iou = _bbox_iou(item.bbox, card.item_bbox)
        reasons = _hard_failure_reasons(item.bbox, card.item_bbox, item.tolerance_px)
        ious.append(iou)
        row = {
            "id": item.id,
            "row": item.row,
            "col": item.col,
            "manual_bbox": list(item.bbox),
            "detected_bbox": list(card.item_bbox),
            "iou": iou,
            "dx1": card.item_bbox[0] - item.bbox[0],
            "dy1": card.item_bbox[1] - item.bbox[1],
            "dx2": card.item_bbox[2] - item.bbox[2],
            "dy2": card.item_bbox[3] - item.bbox[3],
            "hard_failures": reasons,
            "valid": not reasons,
        }
        rows.append(row)
        if reasons:
            hard_failures.append(row)

    return {
        "manual_count": len(manual_items),
        "matched_count": sum(1 for row in rows if row["detected_bbox"] is not None),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "min_iou": float(np.min(ious)) if ious else None,
        "iou_below_0_75": [
            {"id": row["id"], "row": row["row"], "col": row["col"], "iou": row["iou"]}
            for row in rows
            if row["iou"] is not None and row["iou"] < 0.75
        ],
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "rows": rows,
    }


def _load_image(image: str | Path | np.ndarray) -> tuple[np.ndarray, Path | None]:
    if isinstance(image, np.ndarray):
        return _ensure_bgr_u8(image), None
    path = Path(image)
    loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(path)
    return loaded, path


def _analysis_roi(image_shape: tuple[int, ...], options: CardBodyDetectorOptions) -> BBox:
    image_h, image_w = image_shape[:2]
    x1 = int(round(image_w * options.analysis_x1_ratio))
    x2 = int(round(image_w * options.analysis_x2_ratio))
    y1 = int(round(image_h * options.analysis_y1_ratio))
    y2 = int(round(image_h * options.analysis_y2_ratio))
    return _clip_bbox((x1, y1, x2, y2), image_shape)


def _build_card_body_mask(
    image_bgr: np.ndarray,
    analysis_roi: BBox,
    options: CardBodyDetectorOptions,
) -> np.ndarray:
    hsv = cv2.cvtColor(_ensure_bgr_u8(image_bgr), cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(_ensure_bgr_u8(image_bgr), cv2.COLOR_BGR2LAB)
    _h, s, v = cv2.split(hsv)
    l, _a, _b = cv2.split(lab)
    bg_l = _estimate_background_luminance(l, s, v, analysis_roi)
    color_mask = (s > options.saturation_threshold) & (v > options.color_value_threshold)
    gray_mask = (
        ((bg_l - l.astype(np.int16)) > options.gray_delta_l_threshold)
        & (v > options.gray_value_threshold)
        & (s < options.gray_saturation_max)
    )
    roi_mask = np.zeros(image_bgr.shape[:2], dtype=bool)
    x1, y1, x2, y2 = analysis_roi
    roi_mask[y1:y2, x1:x2] = True
    mask = (color_mask | gray_mask) & roi_mask
    return (mask.astype(np.uint8) * 255)


def _estimate_background_luminance(
    l: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    roi: BBox,
) -> int:
    x1, y1, x2, y2 = roi
    l_roi = l[y1:y2, x1:x2]
    s_roi = s[y1:y2, x1:x2]
    v_roi = v[y1:y2, x1:x2]
    candidates = l_roi[(s_roi < 35) & (v_roi > 160)]
    if candidates.size < 100:
        candidates = l_roi[v_roi > 150]
    if candidates.size == 0:
        candidates = l_roi.reshape(-1)
    return int(np.percentile(candidates.astype(np.float32), 75))


def _clean_card_body_mask(
    raw_mask: np.ndarray,
    image_shape: tuple[int, ...],
    options: CardBodyDetectorOptions,
) -> np.ndarray:
    mask = raw_mask.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    filtered = np.zeros_like(mask)
    for bbox in _component_bboxes(mask):
        if _component_looks_like_body(bbox, image_shape, options):
            x1, y1, x2, y2 = bbox
            filtered[y1:y2, x1:x2] = 255
    return filtered


def _extract_body_components(
    clean_mask: np.ndarray,
    image_shape: tuple[int, ...],
    options: CardBodyDetectorOptions,
) -> list[BBox]:
    return [
        bbox
        for bbox in _component_bboxes(clean_mask)
        if _component_looks_like_body(bbox, image_shape, options)
    ]


def _component_bboxes(mask: np.ndarray) -> list[BBox]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    bboxes: list[BBox] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def _component_looks_like_body(
    bbox: BBox,
    image_shape: tuple[int, ...],
    options: CardBodyDetectorOptions,
) -> bool:
    image_h, image_w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h
    if area < options.min_component_area:
        return False
    if w > image_w * 0.58:
        return False
    if x2 > image_w * 0.985:
        return False
    if w < image_w * 0.16:
        return False
    if h < max(10, image_w * 0.018):
        return False
    if h > max(140, image_h * 0.09):
        return False
    aspect = w / max(1, h)
    return options.min_body_aspect <= aspect <= options.max_body_aspect


def _estimate_columns(
    clean_mask: np.ndarray,
    components: Sequence[BBox],
    image_shape: tuple[int, ...],
    options: CardBodyDetectorOptions,
) -> dict[str, tuple[int, int]]:
    image_h, image_w = image_shape[:2]
    component_columns = _columns_from_components(components, image_w)
    if component_columns is not None:
        return component_columns
    projection = clean_mask.mean(axis=0)
    runs = _projection_runs(projection, threshold=max(6.0, float(projection.max()) * 0.30))
    runs = [
        run for run in runs
        if run[1] - run[0] >= image_w * 0.18 and run[1] < image_w * 0.985
    ]
    if len(runs) >= 2:
        runs = sorted(runs, key=lambda item: item[1] - item[0], reverse=True)[:2]
        runs.sort(key=lambda item: item[0])
        return {"left": runs[0], "right": runs[1]}
    raise RuntimeError("failed to estimate two card body columns")


def _columns_from_components(
    components: Sequence[BBox],
    image_width: int,
) -> dict[str, tuple[int, int]] | None:
    candidates = [
        bbox for bbox in components
        if (bbox[2] - bbox[0]) >= image_width * 0.18
    ]
    if len(candidates) < 2:
        return None
    centers = sorted(((bbox[0] + bbox[2]) / 2.0 for bbox in candidates))
    gaps = [(centers[index] - centers[index - 1], index) for index in range(1, len(centers))]
    if not gaps:
        return None
    _gap, split_index = max(gaps, key=lambda item: item[0])
    split_x = (centers[split_index - 1] + centers[split_index]) / 2.0
    left = [bbox for bbox in candidates if (bbox[0] + bbox[2]) / 2.0 < split_x]
    right = [bbox for bbox in candidates if (bbox[0] + bbox[2]) / 2.0 >= split_x]
    if not left or not right:
        return None
    return {
        "left": _stable_column_range(left),
        "right": _stable_column_range(right),
    }


def _stable_column_range(bboxes: Sequence[BBox]) -> tuple[int, int]:
    x1_values = np.array([bbox[0] for bbox in bboxes], dtype=float)
    x2_values = np.array([bbox[2] for bbox in bboxes], dtype=float)
    return (
        int(round(float(np.median(x1_values)))),
        int(round(float(np.median(x2_values)))),
    )


def _row_projection(mask: np.ndarray, column_range: tuple[int, int]) -> np.ndarray:
    x1, x2 = column_range
    if x2 <= x1:
        return np.zeros(mask.shape[0], dtype=float)
    return mask[:, x1:x2].mean(axis=1).astype(float)


def _detect_body_bands(
    row_score: np.ndarray,
    image_width: int,
    options: CardBodyDetectorOptions,
) -> list[_Band]:
    smoothed = _smooth_projection(row_score, window=max(5, int(round(image_width * 0.012))))
    nonzero = smoothed[smoothed > 0]
    if nonzero.size == 0:
        return []
    threshold = max(8.0, float(np.percentile(nonzero, 35)))
    runs = _projection_runs(smoothed, threshold=threshold)
    runs = _merge_close_runs(runs, max_gap=max(3, int(round(image_width * 0.012))))
    min_h = max(10, int(round(image_width * 0.030)))
    max_h = max(min_h + 1, int(round(image_width * 0.115)))
    bands: list[_Band] = []
    for y1, y2 in runs:
        h = y2 - y1
        if h < min_h or h > max_h:
            continue
        score = float(smoothed[y1:y2].mean()) if y2 > y1 else 0.0
        bands.append(_Band(y1=y1, y2=y2, score=score))
    return bands


def _merge_row_centers(
    left_bands: Sequence[_Band],
    right_bands: Sequence[_Band],
    image_width: int,
) -> list[float]:
    centers = sorted([band.center for band in left_bands] + [band.center for band in right_bands])
    if not centers:
        return []
    tolerance = max(8.0, image_width * 0.025)
    groups: list[list[float]] = []
    for center in centers:
        if not groups or abs(center - float(np.mean(groups[-1]))) > tolerance:
            groups.append([center])
        else:
            groups[-1].append(center)
    return [float(np.mean(group)) for group in groups]


def _find_band_near(
    bands: Sequence[_Band],
    center: float,
    tolerance: float,
) -> _Band | None:
    best: tuple[float, _Band] | None = None
    for band in bands:
        distance = abs(band.center - center)
        if distance <= tolerance and (best is None or distance < best[0]):
            best = (distance, band)
    return best[1] if best is not None else None


def _build_cards(
    *,
    source_name: str,
    role: str | None,
    image_shape: tuple[int, ...],
    columns: dict[str, tuple[int, int]],
    left_bands: Sequence[_Band],
    right_bands: Sequence[_Band],
    row_centers: Sequence[float],
    median_body_h: float,
    median_row_pitch: float | None,
    clean_mask: np.ndarray,
    options: CardBodyDetectorOptions,
) -> list[DetectedCardBody]:
    if not row_centers:
        return []
    image_h, _image_w = image_shape[:2]
    body_h = max(1, int(round(median_body_h)))
    row_pitch = median_row_pitch or body_h * 1.35
    center_tolerance = max(body_h * 0.60, row_pitch * 0.18)
    cards: list[DetectedCardBody] = []
    for row, center in enumerate(row_centers):
        next_center = row_centers[row + 1] if row + 1 < len(row_centers) else None
        for col, (name, bands) in enumerate((("left", left_bands), ("right", right_bands))):
            band = _find_band_near(bands, center, center_tolerance)
            if band is None:
                y1 = int(round(center - body_h / 2.0))
                y2 = y1 + body_h
                x1, x2 = columns[name]
                body_bbox = _clip_bbox((x1, y1, x2, y2), image_shape)
                if _mask_fill_ratio(clean_mask, body_bbox) < options.min_grid_fill_body_ratio:
                    continue
                source = "grid_filled"
                confidence = 0.55
            else:
                y1, y2 = band.y1, band.y2
                source = "body_mask_projection"
                confidence = min(1.0, max(0.1, band.score / 255.0))
            x1, x2 = columns[name]
            body_bbox = _clip_bbox((x1, y1, x2, y2), image_shape)
            item_bbox = _item_bbox_from_body(
                body_bbox,
                image_shape,
                median_body_h=median_body_h,
                median_row_pitch=median_row_pitch,
                next_body_y1=int(round(next_center - body_h / 2.0)) if next_center is not None else None,
                options=options,
            )
            valid, reason = _validate_card(item_bbox, image_shape)
            cards.append(
                DetectedCardBody(
                    source_image=source_name,
                    role=role,
                    row=row,
                    col=col,
                    body_bbox=body_bbox,
                    item_bbox=item_bbox,
                    confidence=confidence,
                    source=source,
                    valid=valid,
                    invalid_reason=reason,
                )
            )
    return cards


def _mask_fill_ratio(mask: np.ndarray, bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).mean())


def _item_bbox_from_body(
    body_bbox: BBox,
    image_shape: tuple[int, ...],
    *,
    median_body_h: float,
    median_row_pitch: float | None,
    next_body_y1: int | None,
    options: CardBodyDetectorOptions,
) -> BBox:
    x1, y1, x2, y2 = body_bbox
    body_w = max(1, x2 - x1)
    body_h = max(1, y2 - y1)
    pad_left = int(round(body_w * options.pad_left_ratio))
    pad_right = int(round(body_w * options.pad_right_ratio))
    pitch_gap = max(0.0, (median_row_pitch or body_h * 1.25) - median_body_h)
    pad_top = max(
        int(round(body_h * options.pad_top_ratio)),
        int(round(pitch_gap * options.pad_top_pitch_gap_ratio)),
    )
    star_margin = min(
        int(round(median_body_h * options.star_margin_body_ratio)),
        int(round(pitch_gap * 0.85)) if pitch_gap > 0 else int(round(body_h * 0.35)),
    )
    item = (x1 - pad_left, y1 - pad_top, x2 + pad_right, y2 + max(2, star_margin))
    if next_body_y1 is not None:
        vertical_guard = max(2, int(round(body_h * options.vertical_guard_ratio)))
        item = (item[0], item[1], item[2], min(item[3], next_body_y1 - vertical_guard))
    return _clip_bbox(item, image_shape)


def _validate_card(item_bbox: BBox, image_shape: tuple[int, ...]) -> tuple[bool, str | None]:
    image_h, image_w = image_shape[:2]
    x1, y1, x2, y2 = item_bbox
    reasons: list[str] = []
    if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
        reasons.append("outside_image")
    if x2 <= x1 or y2 <= y1:
        reasons.append("nonpositive_bbox")
    return not reasons, ";".join(reasons) if reasons else None


def _median_body_height(left_bands: Sequence[_Band], right_bands: Sequence[_Band]) -> float:
    heights = [band.height for band in list(left_bands) + list(right_bands)]
    if not heights:
        return 1.0
    return float(np.median(np.array(heights, dtype=float)))


def _median_row_pitch(row_centers: Sequence[float]) -> float | None:
    if len(row_centers) < 2:
        return None
    diffs = np.diff(np.array(row_centers, dtype=float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _median_column_width(columns: dict[str, tuple[int, int]]) -> float:
    widths = [x2 - x1 for x1, x2 in columns.values()]
    return float(np.median(np.array(widths, dtype=float))) if widths else 0.0


def _projection_runs(values: np.ndarray, *, threshold: float) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate((values >= threshold).tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _merge_close_runs(runs: Sequence[tuple[int, int]], *, max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _smooth_projection(values: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values.astype(float), kernel, mode="same")


def _load_manual_items(
    expected_path: str | Path,
    *,
    image_path: str | Path | None,
    image_id: str | None,
) -> list[_ManualItem]:
    path = Path(expected_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data.get("items", []) if isinstance(data, dict) else []
    resolved_image = Path(image_path).name if image_path is not None else None
    items: list[_ManualItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        if image_id is not None and raw.get("image_id") != image_id:
            continue
        source = raw.get("source_image")
        if image_id is None and resolved_image is not None and source is not None:
            if Path(str(source)).name != resolved_image:
                continue
        bbox = raw.get("bbox")
        row = raw.get("row")
        col = raw.get("col")
        if bbox is None or row is None or col is None or len(bbox) != 4:
            continue
        items.append(
            _ManualItem(
                id=str(raw.get("id") or f"r{row}_c{col}"),
                image_id=str(raw["image_id"]) if raw.get("image_id") is not None else None,
                source_image=str(source) if source is not None else None,
                role=str(raw["role"]) if raw.get("role") is not None else None,
                row=int(row),
                col=int(col),
                bbox=tuple(int(v) for v in bbox),  # type: ignore[arg-type]
                tolerance_px=int(raw.get("tolerance_px", 8)),
            )
        )
    return items


def _hard_failure_reasons(manual: BBox, detected: BBox, tolerance: int) -> list[str]:
    mx1, my1, mx2, my2 = manual
    dx1, dy1, dx2, dy2 = detected
    reasons: list[str] = []
    if dx1 - mx1 > tolerance:
        reasons.append("possible_left_icon_cut")
    if dy1 - my1 > tolerance:
        reasons.append("possible_text_top_cut")
    if mx2 - dx2 > tolerance:
        reasons.append("possible_text_or_star_right_cut")
    if my2 - dy2 > tolerance:
        reasons.append("possible_star_bottom_cut")
    manual_w = max(1, mx2 - mx1)
    manual_h = max(1, my2 - my1)
    detected_w = max(1, dx2 - dx1)
    detected_h = max(1, dy2 - dy1)
    if detected_w > manual_w * 1.35:
        reasons.append("too_wide_possible_neighbor")
    if detected_h > manual_h * 1.35:
        reasons.append("too_tall_possible_neighbor")
    if detected_w < manual_w * 0.70:
        reasons.append("too_narrow")
    if detected_h < manual_h * 0.70:
        reasons.append("too_short")
    return reasons


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _clip_bbox(bbox: BBox, image_shape: tuple[int, ...]) -> BBox:
    image_h, image_w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), max(0, image_w - 1)))
    y1 = max(0, min(int(y1), max(0, image_h - 1)))
    x2 = max(x1 + 1, min(int(x2), image_w))
    y2 = max(y1 + 1, min(int(y2), image_h))
    return x1, y1, x2, y2


def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
