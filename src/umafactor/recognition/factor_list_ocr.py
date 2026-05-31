"""OCR helpers for stitched factor-list tiles."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import numpy as np

from ..detection.card_body_detector import DetectedCardBody, detect_card_bodies
from ..detection.factor_list import FactorListTile
from ..detection.constants import GOLD_STAR_HSV_HI, GOLD_STAR_HSV_LO
from ..detection.star_slots import detect_star_slots_from_card
from .ocr_protocol import (
    DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
    DEFAULT_OCR_MAX_UPSCALE,
    DEFAULT_OCR_MIN_HEIGHT,
    DEFAULT_OCR_MIN_WIDTH,
    DEFAULT_OCR_SHARPEN_STRENGTH,
    DEFAULT_OCR_SHEET_MAX_SIDE,
    FactorListCardDetectorMode,
    FactorListOCRLike,
    FactorListOcrCropTarget,
    FactorListOcrExecutionMode,
    FactorListOcrPreprocessMode,
)
from .name_roi import NameRoiOptions, compute_name_roi_from_body


@dataclass(frozen=True)
class OcrCanvasBatch:
    canvas: np.ndarray
    regions: list[tuple[int, int, int, int]]


@dataclass(frozen=True)
class FactorListOcrRegionDebug:
    card_bbox: tuple[int, int, int, int]
    initial_card_bbox: tuple[int, int, int, int]
    icon_exclusion_bbox: tuple[int, int, int, int]
    icon_bbox: tuple[int, int, int, int] | None
    icon_center: tuple[int, int] | None
    icon_radius: int | None
    star_roi_bbox: tuple[int, int, int, int]
    star_slot_bboxes: tuple[tuple[int, int, int, int], ...]
    text_bbox: tuple[int, int, int, int]
    preprocess_mode: FactorListOcrPreprocessMode
    fallback: bool
    rejected_icon_bboxes: tuple[tuple[int, int, int, int], ...] = ()
    expected_icon_center: tuple[int, int] | None = None
    final_icon_center: tuple[int, int] | None = None
    icon_selected_by: str = ""


@dataclass(frozen=True)
class FactorListCardDebug:
    card_bbox: tuple[int, int, int, int]
    initial_card_bbox: tuple[int, int, int, int]
    icon_bbox: tuple[int, int, int, int] | None
    icon_center: tuple[int, int] | None
    icon_radius: int | None
    fallback: bool
    rejected_icon_bboxes: tuple[tuple[int, int, int, int], ...] = ()
    expected_icon_center: tuple[int, int] | None = None
    final_icon_center: tuple[int, int] | None = None
    icon_selected_by: str = ""


@dataclass(frozen=True)
class FactorListOcrCandidate:
    tile: FactorListTile
    roi_profile: str
    preprocess_mode: str
    card_bbox: tuple[int, int, int, int]
    initial_card_bbox: tuple[int, int, int, int]
    icon_bbox: tuple[int, int, int, int] | None
    icon_center: tuple[int, int] | None
    icon_radius: int | None
    roi_bbox: tuple[int, int, int, int]
    star_roi_bbox: tuple[int, int, int, int]
    raw_crop: np.ndarray
    upscaled_crop: np.ndarray
    preprocessed_crop: np.ndarray
    fallback: bool = False
    rejected_icon_bboxes: tuple[tuple[int, int, int, int], ...] = ()
    expected_icon_center: tuple[int, int] | None = None
    final_icon_center: tuple[int, int] | None = None
    icon_selected_by: str = ""
    ocr_raw: str = ""
    ocr_score: float | None = None
    ocr_elapsed_ms: float | None = None


@dataclass(frozen=True)
class _OcrTextResult:
    text: str
    score: float | None
    elapsed_ms: float | None


@dataclass(frozen=True)
class _PreparedOcrCrop:
    tile: FactorListTile
    crop: np.ndarray


def recognize_factor_list_tile_candidates(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    ocr: FactorListOCRLike,
    *,
    roi_profiles: Sequence[str],
    preprocess_modes: Sequence[str],
    min_crop_width: int = DEFAULT_OCR_MIN_WIDTH,
    min_crop_height: int = DEFAULT_OCR_MIN_HEIGHT,
    max_upscale: float = DEFAULT_OCR_MAX_UPSCALE,
    sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH,
    contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
    ocr_execution_mode: FactorListOcrExecutionMode = "batch",
    batch_size: int = 12,
    canvas_padding: int = 24,
    sheet_max_side: int = DEFAULT_OCR_SHEET_MAX_SIDE,
    sheet_columns: int | None = None,
    card_detector: FactorListCardDetectorMode = "card-body",
) -> list[FactorListOcrCandidate]:
    candidates = build_factor_list_ocr_candidates(
        image,
        tiles,
        roi_profiles=roi_profiles,
        preprocess_modes=preprocess_modes,
        min_crop_width=min_crop_width,
        min_crop_height=min_crop_height,
        max_upscale=max_upscale,
        sharpen_strength=sharpen_strength,
        contrast_clip_limit=contrast_clip_limit,
        card_detector=card_detector,
    )
    ocr_results = _recognize_candidate_crops(
        candidates,
        ocr,
        ocr_execution_mode=ocr_execution_mode,
        batch_size=batch_size,
        canvas_padding=canvas_padding,
        sheet_max_side=sheet_max_side,
        sheet_columns=sheet_columns,
    )
    return [
        replace(
            candidate,
            ocr_raw=result.text,
            ocr_score=result.score,
            ocr_elapsed_ms=result.elapsed_ms,
        )
        for candidate, result in zip(candidates, ocr_results)
    ]


def build_factor_list_ocr_candidates(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    *,
    roi_profiles: Sequence[str],
    preprocess_modes: Sequence[str],
    min_crop_width: int = DEFAULT_OCR_MIN_WIDTH,
    min_crop_height: int = DEFAULT_OCR_MIN_HEIGHT,
    max_upscale: float = DEFAULT_OCR_MAX_UPSCALE,
    sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH,
    contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
    card_detector: FactorListCardDetectorMode = "card-body",
) -> list[FactorListOcrCandidate]:
    candidates: list[FactorListOcrCandidate] = []
    roi_profiles = tuple(roi_profiles) or ("text_band_with_margin",)
    card_debugs = build_factor_list_card_debugs(image, tiles, card_detector=card_detector)
    for tile in tiles:
        card_debug = card_debugs.get(_tile_debug_key(tile)) or factor_list_card_region_debug(image, tile)
        card_bbox = card_debug.card_bbox
        card_x0, card_y0, card_x1, card_y1 = card_bbox
        card_crop = image[card_y0:card_y1, card_x0:card_x1]
        if card_crop.size == 0:
            continue
        active_preprocess_modes = _candidate_preprocess_modes_for_color(
            tile.color,
            preprocess_modes,
        )
        for profile in roi_profiles:
            region_debug = factor_list_profile_region_debug(
                image,
                tile,
                profile=profile,
                card_debug=card_debug,
            )
            roi_bbox = region_debug.text_bbox
            roi_x0, roi_y0, roi_x1, roi_y1 = roi_bbox
            raw_crop = image[roi_y0:roi_y1, roi_x0:roi_x1]
            if raw_crop.size == 0:
                continue
            upscaled_crop = upscale_factor_list_ocr_crop(
                raw_crop,
                min_width=min_crop_width,
                min_height=min_crop_height,
                max_upscale=max_upscale,
            )
            for mode in active_preprocess_modes:
                preprocessed_crop = preprocess_upscaled_factor_list_ocr_crop(
                    upscaled_crop,
                    mode=mode,
                    sharpen_strength=sharpen_strength,
                    contrast_clip_limit=contrast_clip_limit,
                )
                candidates.append(
                    FactorListOcrCandidate(
                        tile=tile,
                        roi_profile=profile,
                        preprocess_mode=mode,
                        card_bbox=card_bbox,
                        initial_card_bbox=card_debug.initial_card_bbox,
                        icon_bbox=card_debug.icon_bbox,
                        icon_center=card_debug.icon_center,
                        icon_radius=card_debug.icon_radius,
                        roi_bbox=roi_bbox,
                        star_roi_bbox=region_debug.star_roi_bbox,
                        raw_crop=raw_crop,
                        upscaled_crop=upscaled_crop,
                        preprocessed_crop=preprocessed_crop,
                        fallback=card_debug.fallback,
                        rejected_icon_bboxes=card_debug.rejected_icon_bboxes,
                        expected_icon_center=card_debug.expected_icon_center,
                        final_icon_center=card_debug.final_icon_center,
                        icon_selected_by=card_debug.icon_selected_by,
                    )
                )
    return candidates


def recognize_factor_list_tile_names(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    ocr: FactorListOCRLike,
    *,
    crop_variant: str = "current",
    crop_target: FactorListOcrCropTarget = "name",
    preprocess_crop: bool = True,
    min_crop_width: int = DEFAULT_OCR_MIN_WIDTH,
    min_crop_height: int = DEFAULT_OCR_MIN_HEIGHT,
    max_upscale: float = DEFAULT_OCR_MAX_UPSCALE,
    sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH,
    contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
    ocr_execution_mode: FactorListOcrExecutionMode = "sequential",
    canvas_batch_size: int = 12,
    canvas_padding: int = 24,
) -> list[FactorListTile]:
    """Return tiles with raw OCR names filled.

    The output intentionally keeps OCR text as raw_name.  Dictionary correction
    and known/unknown skill handling belong to later review/database stages.
    When canvas_batch_size is 0 or negative, canvas/batch modes process all
    provided tiles at once.  The live flow calls this once per role, so that
    becomes one OCR image each for parent, ancestor1, and ancestor2.
    """

    prepared = _prepare_ocr_crops(
        image,
        tiles,
        crop_variant=crop_variant,
        crop_target=crop_target,
        preprocess_crop=preprocess_crop,
        min_crop_width=min_crop_width,
        min_crop_height=min_crop_height,
        max_upscale=max_upscale,
        sharpen_strength=sharpen_strength,
        contrast_clip_limit=contrast_clip_limit,
    )

    if ocr_execution_mode == "canvas":
        raw_names = _recognize_prepared_crops_by_canvas(
            prepared,
            ocr,
            canvas_batch_size=canvas_batch_size,
            canvas_padding=canvas_padding,
        )
    elif ocr_execution_mode == "batch":
        raw_names = _recognize_prepared_crops_by_batch(prepared, ocr, batch_size=canvas_batch_size)
    elif ocr_execution_mode == "sequential":
        raw_names = [_recognize_one_tile(entry.tile, entry.crop, ocr) for entry in prepared]
    else:
        raise ValueError(f"unknown OCR execution mode: {ocr_execution_mode}")

    return [
        replace(entry.tile, raw_name=raw_name, final_name=raw_name)
        for entry, raw_name in zip(prepared, raw_names)
    ]


def pack_ocr_canvas(
    crops: Sequence[np.ndarray],
    *,
    padding: int = 24,
    background: int = 255,
) -> OcrCanvasBatch:
    """Pack OCR crops into one vertical canvas and retain crop-local regions."""

    if not crops:
        return OcrCanvasBatch(np.zeros((1, 1, 3), dtype=np.uint8), [])

    padding = max(0, int(padding))
    prepared = [_ensure_bgr_u8(crop) for crop in crops]
    max_width = max(crop.shape[1] for crop in prepared)
    total_height = padding
    for crop in prepared:
        total_height += crop.shape[0] + padding

    canvas = np.full(
        (max(1, total_height), max(1, max_width + padding * 2), 3),
        int(np.clip(background, 0, 255)),
        dtype=np.uint8,
    )
    regions: list[tuple[int, int, int, int]] = []
    y = padding
    for crop in prepared:
        h, w = crop.shape[:2]
        x = padding
        canvas[y : y + h, x : x + w] = crop
        regions.append((x, y, x + w, y + h))
        y += h + padding
    return OcrCanvasBatch(canvas, regions)


def pack_ocr_sheet(
    crops: Sequence[np.ndarray],
    *,
    padding: int = 24,
    background: int = 255,
    max_side: int = DEFAULT_OCR_SHEET_MAX_SIDE,
    columns: int | None = None,
) -> OcrCanvasBatch:
    """Pack OCR crops into a multi-column sheet and keep exact crop regions.

    PaddleOCR detection may resize very tall canvases internally.  A compact
    sheet keeps both dimensions below max_side when possible while retaining
    enough whitespace for line-box assignment.
    """

    if not crops:
        return OcrCanvasBatch(np.zeros((1, 1, 3), dtype=np.uint8), [])

    padding = max(0, int(padding))
    max_side = max(1, int(max_side))
    prepared = [_ensure_bgr_u8(crop) for crop in crops]
    cell_width = max(crop.shape[1] for crop in prepared)
    cell_height = max(crop.shape[0] for crop in prepared)
    resolved_columns = _resolve_sheet_columns(
        len(prepared),
        cell_width=cell_width,
        cell_height=cell_height,
        padding=padding,
        max_side=max_side,
        requested_columns=columns,
    )
    rows = int(np.ceil(len(prepared) / resolved_columns))
    canvas_width = padding + resolved_columns * (cell_width + padding)
    canvas_height = padding + rows * (cell_height + padding)
    canvas = np.full(
        (max(1, canvas_height), max(1, canvas_width), 3),
        int(np.clip(background, 0, 255)),
        dtype=np.uint8,
    )

    regions: list[tuple[int, int, int, int]] = []
    for index, crop in enumerate(prepared):
        row = index // resolved_columns
        col = index % resolved_columns
        h, w = crop.shape[:2]
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        canvas[y : y + h, x : x + w] = crop
        regions.append((x, y, x + w, y + h))
    return OcrCanvasBatch(canvas, regions)


def _prepare_ocr_crops(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    *,
    crop_variant: str,
    crop_target: FactorListOcrCropTarget,
    preprocess_crop: bool,
    min_crop_width: int,
    min_crop_height: int,
    max_upscale: float,
    sharpen_strength: float,
    contrast_clip_limit: float,
) -> list[_PreparedOcrCrop]:
    prepared: list[_PreparedOcrCrop] = []
    for tile in tiles:
        crop = crop_factor_list_ocr_region(
            image,
            tile,
            target=crop_target,
            variant=crop_variant,
        )
        if preprocess_crop:
            crop = prepare_factor_list_ocr_crop(
                crop,
                mode=ocr_preprocess_mode_for_color(tile.color),
                min_width=min_crop_width,
                min_height=min_crop_height,
                max_upscale=max_upscale,
                sharpen_strength=sharpen_strength,
                contrast_clip_limit=contrast_clip_limit,
            )
        prepared.append(_PreparedOcrCrop(tile=tile, crop=crop))
    return prepared


def _recognize_prepared_crops_by_canvas(
    prepared: Sequence[_PreparedOcrCrop],
    ocr: FactorListOCRLike,
    *,
    canvas_batch_size: int,
    canvas_padding: int,
) -> list[str]:
    recognize_canvas = getattr(ocr, "recognize_canvas", None)
    if not callable(recognize_canvas):
        return _recognize_prepared_crops_by_batch(
            prepared,
            ocr,
            batch_size=canvas_batch_size,
        )

    raw_names: list[str] = []
    for chunk in _chunks(list(prepared), _effective_batch_size(prepared, canvas_batch_size)):
        packed = pack_ocr_canvas([entry.crop for entry in chunk], padding=canvas_padding)
        names = list(recognize_canvas(packed.canvas, packed.regions))
        if len(names) != len(chunk):
            names = names[: len(chunk)] + [""] * max(0, len(chunk) - len(names))
        raw_names.extend(str(name) for name in names)
    return raw_names


def _recognize_prepared_crops_by_batch(
    prepared: Sequence[_PreparedOcrCrop],
    ocr: FactorListOCRLike,
    *,
    batch_size: int,
) -> list[str]:
    recognize_many = getattr(ocr, "recognize_many", None)
    if not callable(recognize_many):
        return [_recognize_one_tile(entry.tile, entry.crop, ocr) for entry in prepared]

    raw_names: list[str] = []
    for chunk in _chunks(list(prepared), _effective_batch_size(prepared, batch_size)):
        names = list(recognize_many([entry.crop for entry in chunk]))
        if len(names) != len(chunk):
            names = names[: len(chunk)] + [""] * max(0, len(chunk) - len(names))
        raw_names.extend(str(name) for name in names)
    return raw_names


def _recognize_candidate_crops(
    candidates: Sequence[FactorListOcrCandidate],
    ocr: FactorListOCRLike,
    *,
    ocr_execution_mode: FactorListOcrExecutionMode,
    batch_size: int,
    canvas_padding: int,
    sheet_max_side: int,
    sheet_columns: int | None,
) -> list[_OcrTextResult]:
    if not candidates:
        return []
    if ocr_execution_mode == "sequential":
        return [
            _recognize_one_candidate(candidate, ocr)
            for candidate in candidates
        ]
    if ocr_execution_mode == "role_sheet":
        return _recognize_candidate_crops_by_sheet(
            candidates,
            ocr,
            batch_size=batch_size,
            canvas_padding=canvas_padding,
            sheet_max_side=sheet_max_side,
            sheet_columns=sheet_columns,
        )
    if ocr_execution_mode in {"batch", "canvas"}:
        recognize_many = getattr(ocr, "recognize_many", None)
        if not callable(recognize_many):
            return [_recognize_one_candidate(candidate, ocr) for candidate in candidates]

        raw_names: list[_OcrTextResult] = []
        for chunk in _chunks_candidates(list(candidates), _effective_batch_size(candidates, batch_size)):
            started = time.perf_counter()
            score_results = _recognize_many_with_optional_scores(
                ocr,
                [candidate.preprocessed_crop for candidate in chunk],
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if len(score_results) != len(chunk):
                score_results = score_results[: len(chunk)] + [
                    ("", None)
                ] * max(0, len(chunk) - len(score_results))
            per_item_ms = elapsed_ms / max(1, len(chunk))
            raw_names.extend(
                _OcrTextResult(text=str(text), score=score, elapsed_ms=per_item_ms)
                for text, score in score_results
            )
        return raw_names
    raise ValueError(f"unknown OCR execution mode: {ocr_execution_mode}")


def _recognize_candidate_crops_by_sheet(
    candidates: Sequence[FactorListOcrCandidate],
    ocr: FactorListOCRLike,
    *,
    batch_size: int,
    canvas_padding: int,
    sheet_max_side: int,
    sheet_columns: int | None,
) -> list[_OcrTextResult]:
    recognize_canvas = getattr(ocr, "recognize_canvas", None)
    if not callable(recognize_canvas):
        recognize_many = getattr(ocr, "recognize_many", None)
        if callable(recognize_many):
            return _recognize_candidate_crops(
                candidates,
                ocr,
                ocr_execution_mode="batch",
                batch_size=batch_size,
                canvas_padding=canvas_padding,
                sheet_max_side=sheet_max_side,
                sheet_columns=sheet_columns,
            )
        return [_recognize_one_candidate(candidate, ocr) for candidate in candidates]

    raw_names: list[_OcrTextResult] = []
    for chunk in _chunks_candidates(list(candidates), _effective_batch_size(candidates, batch_size)):
        packed = pack_ocr_sheet(
            [candidate.preprocessed_crop for candidate in chunk],
            padding=canvas_padding,
            max_side=sheet_max_side,
            columns=sheet_columns,
        )
        started = time.perf_counter()
        names = list(recognize_canvas(packed.canvas, packed.regions))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(names) != len(chunk):
            names = names[: len(chunk)] + [""] * max(0, len(chunk) - len(names))
        per_item_ms = elapsed_ms / max(1, len(chunk))
        raw_names.extend(
            _OcrTextResult(text=str(name), score=None, elapsed_ms=per_item_ms)
            for name in names
        )
    return raw_names


def _recognize_one_candidate(
    candidate: FactorListOcrCandidate,
    ocr: FactorListOCRLike,
) -> _OcrTextResult:
    started = time.perf_counter()
    recognize_with_score = getattr(ocr, "recognize_with_score", None)
    if callable(recognize_with_score):
        text, score = recognize_with_score(candidate.preprocessed_crop)
    else:
        text = _recognize_one_tile(candidate.tile, candidate.preprocessed_crop, ocr)
        score = None
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _OcrTextResult(text=str(text), score=score, elapsed_ms=elapsed_ms)


def _recognize_many_with_optional_scores(
    ocr: FactorListOCRLike,
    crops: Sequence[np.ndarray],
) -> list[tuple[str, float | None]]:
    recognize_many_with_scores = getattr(ocr, "recognize_many_with_scores", None)
    if callable(recognize_many_with_scores):
        return [(str(text), score) for text, score in recognize_many_with_scores(crops)]
    recognize_many = getattr(ocr, "recognize_many", None)
    if callable(recognize_many):
        return [(str(text), None) for text in recognize_many(crops)]
    return []


def _recognize_one_tile(
    tile: FactorListTile,
    crop: np.ndarray,
    ocr: FactorListOCRLike,
) -> str:
    if tile.color == "blue":
        return ocr.recognize_blue(crop)
    if tile.color == "red":
        return ocr.recognize_red(crop)
    if tile.color == "green":
        raw_name, _fragments = ocr.recognize_with_parts(crop)
        return raw_name
    return ocr.recognize(crop)


def _chunks(values: Sequence[_PreparedOcrCrop], size: int) -> list[Sequence[_PreparedOcrCrop]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _chunks_candidates(
    values: Sequence[FactorListOcrCandidate],
    size: int,
) -> list[Sequence[FactorListOcrCandidate]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _effective_batch_size(values: Sequence[_PreparedOcrCrop], requested_size: int) -> int:
    if requested_size <= 0:
        return max(1, len(values))
    return max(1, requested_size)


def _resolve_sheet_columns(
    count: int,
    *,
    cell_width: int,
    cell_height: int,
    padding: int,
    max_side: int,
    requested_columns: int | None,
) -> int:
    if count <= 0:
        return 1
    if requested_columns is not None and requested_columns > 0:
        return max(1, min(count, int(requested_columns)))

    best_columns: int | None = None
    best_score: tuple[int, int, int] | None = None
    for columns in range(1, count + 1):
        rows = int(np.ceil(count / columns))
        canvas_width = padding + columns * (cell_width + padding)
        canvas_height = padding + rows * (cell_height + padding)
        fits = canvas_width <= max_side and canvas_height <= max_side
        if not fits:
            continue
        score = (max(canvas_width, canvas_height), canvas_width * canvas_height, columns)
        if best_score is None or score < best_score:
            best_columns = columns
            best_score = score
    if best_columns is not None:
        return best_columns

    max_columns_by_width = max(1, (max_side - padding) // max(1, cell_width + padding))
    return max(1, min(count, max_columns_by_width))


def prepare_factor_list_ocr_crop(
    crop: np.ndarray,
    *,
    mode: FactorListOcrPreprocessMode = "dark_text",
    min_width: int = DEFAULT_OCR_MIN_WIDTH,
    min_height: int = DEFAULT_OCR_MIN_HEIGHT,
    max_upscale: float = DEFAULT_OCR_MAX_UPSCALE,
    sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH,
    contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
) -> np.ndarray:
    """Normalize a per-card OCR crop before passing it to the OCR engine.

    Live Steam captures can produce much smaller card crops than the static
    screenshots used for evaluation.  This keeps OCR input size stable without
    changing detection coordinates or the stitched output image.
    """

    if crop is None or crop.size == 0:
        return crop

    upscaled = upscale_factor_list_ocr_crop(
        crop,
        min_width=max(1, min_width),
        min_height=max(1, min_height),
        max_upscale=max(1.0, max_upscale),
    )
    return preprocess_upscaled_factor_list_ocr_crop(
        upscaled,
        mode=mode,
        sharpen_strength=sharpen_strength,
        contrast_clip_limit=contrast_clip_limit,
    )


def upscale_factor_list_ocr_crop(
    crop: np.ndarray,
    *,
    min_width: int = DEFAULT_OCR_MIN_WIDTH,
    min_height: int = DEFAULT_OCR_MIN_HEIGHT,
    max_upscale: float = DEFAULT_OCR_MAX_UPSCALE,
) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    prepared = _ensure_bgr_u8(crop)
    return _upscale_to_min_size(
        prepared,
        min_width=min_width,
        min_height=min_height,
        max_upscale=max_upscale,
    )


def preprocess_upscaled_factor_list_ocr_crop(
    upscaled_crop: np.ndarray,
    *,
    mode: str,
    sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH,
    contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
) -> np.ndarray:
    if upscaled_crop is None or upscaled_crop.size == 0:
        return upscaled_crop
    prepared = _ensure_bgr_u8(upscaled_crop)
    if mode == "raw_upscaled":
        return prepared
    if mode in {"gray_sharpen", "dark_text"}:
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_bgr = _apply_luminance_clahe(gray_bgr, clip_limit=contrast_clip_limit)
        return _apply_unsharp_mask(gray_bgr, strength=sharpen_strength)
    if mode in {"color_text_safe", "light_text_safe", "white_text"}:
        prepared = _apply_luminance_clahe(prepared, clip_limit=max(0.0, contrast_clip_limit * 0.5))
        return _apply_unsharp_mask(prepared, strength=sharpen_strength * 0.6)
    raise ValueError(f"unknown OCR preprocess mode: {mode}")


def ocr_preprocess_mode_for_color(color: str | None) -> FactorListOcrPreprocessMode:
    if color in {"blue", "red", "green"}:
        return "color_text_safe"
    return "gray_sharpen"


def _candidate_preprocess_modes_for_color(
    color: str | None,
    modes: Sequence[str],
) -> tuple[str, ...]:
    requested = tuple(modes) or ("raw_upscaled",)
    return tuple(dict.fromkeys(requested))


def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _upscale_to_min_size(
    image: np.ndarray,
    *,
    min_width: int,
    min_height: int,
    max_upscale: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image

    scale = max(1.0, min_width / width, min_height / height)
    scale = min(max_upscale, scale)
    if scale <= 1.01:
        return image

    new_width = max(width + 1, int(round(width * scale)))
    new_height = max(height + 1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def _apply_luminance_clahe(image: np.ndarray, *, clip_limit: float) -> np.ndarray:
    if clip_limit <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(4, 4))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _apply_unsharp_mask(image: np.ndarray, *, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)


def crop_factor_list_ocr_region(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    target: FactorListOcrCropTarget = "name",
    variant: str = "current",
) -> np.ndarray:
    x0, y0, x1, y1 = factor_list_ocr_region_bbox(
        image,
        tile,
        target=target,
        variant=variant,
    )
    return image[y0:y1, x0:x1]


def factor_list_ocr_region_bbox(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    target: FactorListOcrCropTarget = "name",
    variant: str = "current",
) -> tuple[int, int, int, int]:
    if target == "name":
        return factor_list_name_region_bbox(image, tile, variant=variant)
    if target == "card":
        return factor_list_card_region_bbox(image, tile)
    raise ValueError(f"unknown OCR crop target: {target}")


def crop_factor_list_card_region(
    image: np.ndarray,
    tile: FactorListTile,
) -> np.ndarray:
    x0, y0, x1, y1 = factor_list_card_region_bbox(image, tile)
    return image[y0:y1, x0:x1]


def factor_list_card_region_bbox(
    image: np.ndarray,
    tile: FactorListTile,
) -> tuple[int, int, int, int]:
    """Return the refined card bbox for one factor tile."""

    return factor_list_card_region_debug(image, tile).card_bbox


def factor_list_card_region_debug(
    image: np.ndarray,
    tile: FactorListTile,
) -> FactorListCardDebug:
    """Detect one card bbox from a round-icon anchor and UI-grid estimate."""

    return build_factor_list_card_debugs(image, [tile]).get(
        _tile_debug_key(tile),
        _fallback_card_debug(image, tile),
    )


def build_factor_list_card_debugs(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    *,
    card_detector: FactorListCardDetectorMode = "card-body",
) -> dict[tuple[int, int, int, int], FactorListCardDebug]:
    """Build card bboxes with the formal card-body detector."""

    tile_list = list(tiles)
    if not tile_list:
        return {}

    if card_detector != "card-body":
        raise ValueError(f"unknown card detector: {card_detector}")
    return _build_card_body_card_debugs(image, tile_list)


def _build_card_body_card_debugs(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
) -> dict[tuple[int, int, int, int], FactorListCardDebug]:
    if not tiles:
        return {}
    role = tiles[0].role if len({tile.role for tile in tiles}) == 1 else None
    try:
        run = detect_card_bodies(image, role=role)
    except Exception:
        return {}

    cards = [card for card in run.result.cards if card.valid]
    if not cards:
        return {}

    tolerance = _card_body_match_tolerance(
        run.result.median_body_h,
        run.result.median_row_pitch,
    )
    matched_cards = _match_card_body_cards_to_tiles(tiles, cards, tolerance=tolerance)
    return {
        _tile_debug_key(tile): _card_body_card_debug(image, card)
        for tile, card in matched_cards.items()
    }


def _match_card_body_cards_to_tiles(
    tiles: Sequence[FactorListTile],
    cards: Sequence[DetectedCardBody],
    *,
    tolerance: float,
) -> dict[FactorListTile, DetectedCardBody]:
    cards_by_col: dict[int, list[DetectedCardBody]] = {0: [], 1: []}
    for card in cards:
        cards_by_col.setdefault(card.col, []).append(card)
    for col_cards in cards_by_col.values():
        col_cards.sort(key=_card_body_card_center_y)

    matched: dict[FactorListTile, DetectedCardBody] = {}
    used_card_ids: set[int] = set()
    for tile in sorted(tiles, key=lambda item: (item.row_index, item.col_index, item.order)):
        target_y = _tile_center_y(tile)
        candidates = [
            (abs(_card_body_card_center_y(card) - target_y), card)
            for card in cards_by_col.get(tile.col_index, [])
            if id(card) not in used_card_ids
        ]
        if not candidates:
            continue
        distance, card = min(candidates, key=lambda item: item[0])
        if distance > tolerance:
            continue
        matched[tile] = card
        used_card_ids.add(id(card))
    return matched


def _card_body_card_debug(
    image: np.ndarray,
    card: DetectedCardBody,
) -> FactorListCardDebug:
    body_bbox = _clip_nonempty_bbox(card.body_bbox, image.shape)
    item_bbox = _clip_nonempty_bbox(card.item_bbox, image.shape)
    icon_center, icon_radius = _estimate_card_body_icon_anchor(item_bbox, body_bbox)
    icon_bbox = _circle_bbox(icon_center[0], icon_center[1], icon_radius, image.shape)
    return FactorListCardDebug(
        card_bbox=item_bbox,
        initial_card_bbox=body_bbox,
        icon_bbox=icon_bbox,
        icon_center=icon_center,
        icon_radius=icon_radius,
        fallback=False,
        expected_icon_center=icon_center,
        final_icon_center=icon_center,
        icon_selected_by="card_body",
        rejected_icon_bboxes=(),
    )


def _card_body_match_tolerance(
    median_body_h: float,
    median_row_pitch: float | None,
) -> float:
    body_h = max(1.0, float(median_body_h))
    row_pitch = float(median_row_pitch) if median_row_pitch else body_h * 1.55
    return max(8.0, min(row_pitch * 0.55, body_h * 1.35))


def _card_body_card_center_y(card: DetectedCardBody) -> float:
    _x0, y0, _x1, y1 = card.body_bbox
    return (y0 + y1) / 2.0


def _tile_center_y(tile: FactorListTile) -> float:
    _x0, y0, _x1, y1 = tile.bbox
    return (y0 + y1) / 2.0


def _estimate_card_body_icon_anchor(
    item_bbox: tuple[int, int, int, int],
    body_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], int]:
    item_x0, item_y0, _item_x1, _item_y1 = item_bbox
    body_x0, body_y0, _body_x1, body_y1 = body_bbox
    body_h = max(1, body_y1 - body_y0)
    radius = max(3, int(round(body_h * 0.28)))
    cx = int(round(body_x0 + body_h * 0.42))
    cy = int(round((body_y0 + body_y1) / 2.0))
    cx = max(item_x0 + radius, cx)
    cy = max(item_y0 + radius, cy)
    return (cx, cy), radius


def _tile_debug_key(tile: FactorListTile) -> tuple[int, int, int, int]:
    return tile.section_index, tile.order, tile.row_index, tile.col_index


def _fallback_card_debug(image: np.ndarray, tile: FactorListTile) -> FactorListCardDebug:
    bbox = _fixed_ratio_card_bbox(image, tile)
    return FactorListCardDebug(
        card_bbox=bbox,
        initial_card_bbox=bbox,
        icon_bbox=None,
        icon_center=None,
        icon_radius=None,
        fallback=True,
        icon_selected_by="fallback",
    )


def _fixed_ratio_card_bbox(
    image: np.ndarray,
    tile: FactorListTile,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = tile.bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    crop_left = int(round(width * 0.14))
    pad_right = int(round(width * (0.34 if tile.col_index == 0 else 0.20)))
    pad_top = int(round(height * 0.40))
    pad_bottom = int(round(height * 0.20))
    return _clip_nonempty_bbox(
        (
            x0 + crop_left,
            y0 - pad_top,
            x1 + pad_right,
            y1 + pad_bottom,
        ),
        image.shape,
    )


def _circle_bbox(
    cx: int,
    cy: int,
    radius: int,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    return _clip_nonempty_bbox((cx - radius, cy - radius, cx + radius, cy + radius), image_shape)


def crop_factor_list_name_region(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    variant: str = "current",
) -> np.ndarray:
    x0, y0, x1, y1 = factor_list_name_region_bbox(image, tile, variant=variant)
    return image[y0:y1, x0:x1]


def factor_list_name_region_bbox(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    variant: str = "current",
) -> tuple[int, int, int, int]:
    """Return the text-bearing upper area bbox from a factor tile."""

    return factor_list_text_region_debug(image, tile, variant=variant).text_bbox


def factor_list_text_region_debug(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    variant: str = "current",
) -> FactorListOcrRegionDebug:
    """Return structural OCR text ROI and the related debug bboxes."""

    return factor_list_profile_region_debug(image, tile, profile=_profile_from_variant(variant))


def factor_list_profile_region_debug(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    profile: str,
    card_debug: FactorListCardDebug | None = None,
) -> FactorListOcrRegionDebug:
    """Return OCR ROI for one named profile without post-hoc visual tuning."""

    card_debug = card_debug or factor_list_card_region_debug(image, tile)
    card = card_debug.card_bbox
    star_debug = detect_star_slots_from_card(image, card)
    profile = _profile_from_variant(profile)
    name_roi = compute_name_roi_from_body(
        image.shape,
        card_debug.initial_card_bbox,
        options=_name_roi_options_for_profile(profile),
    )

    preprocess_mode = ocr_preprocess_mode_for_color(tile.color)
    return FactorListOcrRegionDebug(
        card_bbox=card,
        initial_card_bbox=card_debug.initial_card_bbox,
        icon_exclusion_bbox=name_roi.icon_exclusion_bbox,
        icon_bbox=card_debug.icon_bbox,
        icon_center=card_debug.icon_center,
        icon_radius=card_debug.icon_radius,
        star_roi_bbox=star_debug.star_roi_bbox,
        star_slot_bboxes=star_debug.slot_bboxes,
        text_bbox=name_roi.text_bbox,
        preprocess_mode=preprocess_mode,
        fallback=card_debug.fallback,
        rejected_icon_bboxes=card_debug.rejected_icon_bboxes,
        expected_icon_center=card_debug.expected_icon_center,
        final_icon_center=card_debug.final_icon_center,
        icon_selected_by=card_debug.icon_selected_by,
    )


def _estimate_text_icon_exclusion_bbox(
    card_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    *,
    profile: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = card_bbox
    width = max(1, x1 - x0)
    icon_right_rel = _text_icon_right_rel(profile)
    return _clip_nonempty_bbox(
        (x0, y0, x0 + int(round(width * icon_right_rel)), y1),
        image_shape,
    )


def _estimate_visible_star_top(
    image: np.ndarray,
    star_roi_bbox: tuple[int, int, int, int],
) -> int | None:
    x0, y0, x1, y1 = star_roi_bbox
    if x1 <= x0 or y1 <= y0:
        return None
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(_ensure_bgr_u8(roi), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(GOLD_STAR_HSV_LO, dtype=np.uint8),
        np.array(GOLD_STAR_HSV_HI, dtype=np.uint8),
    )
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    return int(y0 + ys[0])


def _refine_colored_text_left(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    card_bbox: tuple[int, int, int, int],
    *,
    current_x0: int,
) -> int:
    """Move colored-card text ROI rightward to the first white-text run."""

    x0, y0, x1, y1 = _clip_nonempty_bbox(bbox, image.shape)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[1] < 8:
        return current_x0

    gray = cv2.cvtColor(_ensure_bgr_u8(crop), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    column_score = edges.mean(axis=0) / 255.0
    if column_score.size >= 5:
        kernel = np.ones(5, dtype=float) / 5.0
        column_score = np.convolve(column_score, kernel, mode="same")

    runs = _boolean_runs(column_score > 0.08)
    min_run_width = max(2, int(round(crop.shape[1] * 0.015)))
    runs = [run for run in runs if run[1] - run[0] >= min_run_width]
    if not runs:
        return current_x0

    card_x0, _card_y0, card_x1, _card_y1 = card_bbox
    card_width = max(1, card_x1 - card_x0)
    max_allowed_x0 = card_x0 + int(round(card_width * 0.38))
    minimum_text_start = int(round(crop.shape[1] * 0.16))
    chosen_run = next((run for run in runs if run[0] >= minimum_text_start), runs[0])
    detected_x0 = x0 + chosen_run[0]
    if detected_x0 > max_allowed_x0:
        return current_x0

    pad = max(2, int(round(card_width * 0.018)))
    return max(current_x0, detected_x0 - pad)


def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(values.tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _clip_bbox_to_image(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    return _clip_nonempty_bbox(bbox, image_shape)


def _clip_nonempty_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), max(0, width - 1)))
    y0 = max(0, min(int(y0), max(0, height - 1)))
    x1 = max(x0 + 1, min(int(x1), width))
    y1 = max(y0 + 1, min(int(y1), height))
    return x0, y0, x1, y1


def _profile_from_variant(variant: str) -> str:
    if variant in {"current", "body_name", "name"}:
        return "body_name"
    if variant in {"upper", "text_band_with_margin"}:
        return "text_band_with_margin"
    if variant in {"wide", "card_upper_band"}:
        return "card_upper_band"
    if variant in {"full", "tight_text_roi"}:
        return "tight_text_roi"
    raise ValueError(f"unknown OCR ROI profile: {variant}")


def _name_roi_options_for_profile(profile: str) -> NameRoiOptions:
    if profile == "body_name":
        return NameRoiOptions()
    if profile == "card_upper_band":
        return NameRoiOptions(x1_ratio=0.12, x2_margin_ratio=0.03, y1_ratio=0.03, y2_ratio=0.72)
    if profile == "text_band_with_margin":
        return NameRoiOptions(x1_ratio=0.13, x2_margin_ratio=0.03, y1_ratio=0.04, y2_ratio=0.70)
    if profile == "tight_text_roi":
        return NameRoiOptions(x1_ratio=0.18, x2_margin_ratio=0.03, y1_ratio=0.08, y2_ratio=0.58)
    raise ValueError(f"unknown OCR ROI profile: {profile}")


def _profile_bounds(profile: str) -> tuple[float, float, float, float]:
    if profile == "body_name":
        return 0.13, 0.03, 0.05, 0.68
    if profile == "card_upper_band":
        return 0.13, 0.04, 0.03, 0.72
    if profile == "text_band_with_margin":
        return 0.14, 0.035, 0.03, 0.70
    if profile == "tight_text_roi":
        return 0.18, 0.03, 0.06, 0.56
    raise ValueError(f"unknown OCR ROI profile: {profile}")


def _text_icon_right_rel(profile: str) -> float:
    if profile == "body_name":
        return 0.13
    if profile == "card_upper_band":
        return 0.12
    if profile == "text_band_with_margin":
        return 0.13
    if profile == "tight_text_roi":
        return 0.18
    raise ValueError(f"unknown OCR ROI profile: {profile}")
