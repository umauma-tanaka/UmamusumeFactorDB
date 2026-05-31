"""Standalone factor-list OCR pipeline.

This module intentionally does not know about Google Sheets.  It returns a
neutral OCR result and provides a thin adapter to the existing Submission
schema for callers that want to reuse the current write path.
"""

from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import fuzz

from ..capture.scraper_types import ScrollFrame
from ..capture.static_stitch import stitch_static_scroll_frames
from ..config import green_factor_names, load_labels, load_skill_master_names
from ..detection.constants import BASE_WIDTH
from ..detection.factor_list import (
    FactorListDetection,
    FactorListTile,
)
from ..detection.factor_list_cards import detect_factor_list_cards
from .types import FactorOcrOptions, FactorOcrResult, RecognizedFactor
from ..recognition.constants import BLUE_FACTOR_TYPES, RED_FACTOR_TYPES
from ..recognition.factor_list_ocr import (
    FactorListCardDebug,
    FactorListOcrCandidate,
    FactorListOCRLike,
    build_factor_list_ocr_candidates,
    factor_list_profile_region_debug,
    recognize_factor_list_tile_candidates,
)
from ..recognition.engines import create_factor_list_ocr
from ..recognition.factor_name_matcher import (
    FactorNameMatcher,
    clear_factor_name_matcher_caches,
    match_factor_name,
    normalize_factor_name,
)
from ..recognition.factor_name_ocr_pipeline import (
    FactorNameOcrInput,
    score_factor_name_ocr,
    select_best_factor_name_ocr,
)
from ..schema import FactorEntry, Submission, UmaFactors


@dataclass(frozen=True)
class _MatchedOcrCandidate:
    candidate: FactorListOcrCandidate
    normalized: str
    fuzzy_candidate: str | None
    fuzzy_score: float | None
    selected: bool
    needs_review: bool
    fallback_recommended: bool = False


@dataclass(frozen=True)
class _SelectedTileOcr:
    tile: FactorListTile
    matched: _MatchedOcrCandidate
    candidates: list[_MatchedOcrCandidate]


def recognize_factor_list_image(
    image: str | Path | np.ndarray,
    *,
    options: FactorOcrOptions | None = None,
    ocr: FactorListOCRLike | None = None,
) -> FactorOcrResult:
    """Recognize factor-list tiles from one already-stitched image."""

    resolved_options = options or FactorOcrOptions()
    image_bgr = _load_image_input(image)
    return _recognize_stitched_image(image_bgr, resolved_options, ocr=ocr)


def recognize_factor_list_frames(
    frames: Sequence[ScrollFrame | np.ndarray],
    *,
    options: FactorOcrOptions | None = None,
    ocr: FactorListOCRLike | None = None,
) -> FactorOcrResult:
    """Stitch scroll frames when requested, then run factor-list OCR."""

    resolved_options = options or FactorOcrOptions()
    scroll_frames = _coerce_scroll_frames(frames)
    if not scroll_frames:
        raise ValueError("frames is empty")

    if resolved_options.enable_stitch:
        stitch = stitch_static_scroll_frames(
            scroll_frames,
            use_scrollbar_hint=resolved_options.use_scrollbar_hint,
        )
        stitched_image = stitch.image
    else:
        stitched_image = scroll_frames[0].image.copy()

    return _recognize_stitched_image(stitched_image, resolved_options, ocr=ocr)


def to_submission(
    result: FactorOcrResult,
    *,
    submitter_id: str,
    image_path: str | Path = "",
) -> Submission:
    """Convert neutral factor-list OCR output to the existing Submission type."""

    submission = Submission(
        submitter_id=submitter_id,
        image_filename=Path(image_path).name if image_path else "",
    )
    role_targets = {
        "parent": submission.main,
        "ancestor1": submission.parent1,
        "ancestor2": submission.parent2,
    }

    for factor in sorted(result.factors, key=lambda item: (item.role, item.order)):
        target = role_targets[factor.role]
        _apply_factor_to_uma(target, factor)

    return submission


def get_paddle_factor_ocr(options: FactorOcrOptions) -> FactorListOCRLike:
    """Return a cached PaddleOCR adapter for the current process."""

    return create_factor_list_ocr(replace(options, ocr_mode="paddle"))


def get_factor_ocr(options: FactorOcrOptions) -> FactorListOCRLike:
    """Return a cached OCR adapter for the current process."""

    return create_factor_list_ocr(options)


def get_rapidocr_factor_ocr(options: FactorOcrOptions) -> FactorListOCRLike:
    """Return a cached RapidOCR rec-only adapter for the current process."""

    return create_factor_list_ocr(replace(options, ocr_mode="rapidocr"))


def _recognize_stitched_image(
    image_bgr: np.ndarray,
    options: FactorOcrOptions,
    *,
    ocr: FactorListOCRLike | None,
) -> FactorOcrResult:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image is empty")

    engine = ocr
    if engine is None and options.use_paddle:
        engine = get_factor_ocr(options)

    factors: list[RecognizedFactor] = []
    debug_candidates: list[_MatchedOcrCandidate] = []
    for section_index in range(3):
        try:
            detection = _detect_factor_list_section(image_bgr, section_index, options)
        except IndexError:
            continue

        selected_tiles = _recognize_tiles(image_bgr, detection, options, engine)
        debug_candidates.extend(
            candidate
            for selected in selected_tiles
            for candidate in selected.candidates
        )
        factors.extend(_to_recognized_factor(selected, image_bgr, options) for selected in selected_tiles)

    stitched_image_path = _write_debug_artifacts(image_bgr, factors, debug_candidates, options)
    return FactorOcrResult(
        factors=factors,
        stitched_image_path=stitched_image_path,
        debug_dir=options.debug_dir,
    )


def _detect_factor_list_section(
    image_bgr: np.ndarray,
    section_index: int,
    options: FactorOcrOptions,
) -> FactorListDetection:
    if options.card_detector != "card-body":
        raise ValueError(f"unsupported card detector: {options.card_detector}")
    return detect_factor_list_cards(image_bgr, section_index=section_index)


def _recognize_tiles(
    image_bgr: np.ndarray,
    detection: FactorListDetection,
    options: FactorOcrOptions,
    engine: FactorListOCRLike | None,
) -> list[_SelectedTileOcr]:
    roi_profiles = _ocr_roi_profiles(options)
    preprocess_modes = _ocr_preprocess_modes(options)
    if engine is None:
        candidates = [
            replace(candidate, ocr_raw=candidate.tile.raw_name)
            for candidate in build_factor_list_ocr_candidates(
                image_bgr,
                detection.tiles,
                roi_profiles=roi_profiles,
                preprocess_modes=preprocess_modes,
                min_crop_width=options.ocr_min_crop_width,
                min_crop_height=options.ocr_min_crop_height,
                max_upscale=options.ocr_max_upscale,
                sharpen_strength=options.ocr_sharpen_strength,
                contrast_clip_limit=options.ocr_contrast_clip_limit,
                card_detector=options.card_detector,
            )
        ]
    else:
        candidates = recognize_factor_list_tile_candidates(
            image_bgr,
            detection.tiles,
            engine,
            roi_profiles=roi_profiles,
            preprocess_modes=preprocess_modes,
            min_crop_width=options.ocr_min_crop_width,
            min_crop_height=options.ocr_min_crop_height,
            max_upscale=options.ocr_max_upscale,
            sharpen_strength=options.ocr_sharpen_strength,
            contrast_clip_limit=options.ocr_contrast_clip_limit,
            ocr_execution_mode=options.ocr_execution_mode,
            batch_size=options.ocr_batch_size,
            canvas_padding=options.ocr_canvas_padding,
            sheet_max_side=options.ocr_sheet_max_side,
            sheet_columns=options.ocr_sheet_columns,
            card_detector=options.card_detector,
        )

    grouped = _group_candidates_by_tile(candidates)
    selected_tiles = [
        _select_best_candidate(
            tile,
            grouped.get(_tile_key(tile), []) or [_fallback_candidate(image_bgr, tile)],
            options,
        )
        for tile in detection.tiles
    ]
    return _apply_low_confidence_fallback(selected_tiles, options)


def _apply_low_confidence_fallback(
    selected_tiles: list[_SelectedTileOcr],
    options: FactorOcrOptions,
) -> list[_SelectedTileOcr]:
    if options.fallback_ocr_mode == "none":
        return selected_tiles
    low_confidence = [selected for selected in selected_tiles if selected.matched.fallback_recommended]
    if not low_confidence:
        return selected_tiles
    if options.fallback_ocr_mode != "paddle":
        raise ValueError(f"unsupported fallback OCR mode: {options.fallback_ocr_mode}")

    fallback_engine = get_paddle_factor_ocr(options)
    updated: list[_SelectedTileOcr] = []
    for selected in selected_tiles:
        if not selected.matched.fallback_recommended:
            updated.append(selected)
            continue
        fallback_candidate = _recognize_candidate_with_engine(
            selected.matched.candidate,
            fallback_engine,
            preprocess_suffix="paddle_fallback",
        )
        updated.append(
            _select_best_candidate(
                selected.tile,
                [match.candidate for match in selected.candidates] + [fallback_candidate],
                options,
            )
        )
    return updated


def _to_recognized_factor(
    selected: _SelectedTileOcr,
    image_bgr: np.ndarray,
    options: FactorOcrOptions,
) -> RecognizedFactor:
    tile = selected.tile
    matched = selected.matched
    candidate = matched.candidate
    raw_name = candidate.ocr_raw.strip() if candidate.ocr_raw else ""
    normalized_name = (
        matched.fuzzy_candidate
        if not matched.needs_review and matched.fuzzy_candidate
        else matched.normalized
    )
    card_debug = _card_debug_from_candidate(candidate)
    region_debug = factor_list_profile_region_debug(
        image_bgr,
        tile,
        profile=candidate.roi_profile,
        card_debug=card_debug,
    )
    return RecognizedFactor(
        role=tile.role,
        order=tile.order,
        row=tile.row_index,
        col=tile.col_index,
        raw_name=raw_name,
        normalized_name=normalized_name or None,
        category=tile.color,
        stars=max(0, min(3, int(tile.star))),
        bbox=candidate.card_bbox,
        ocr_bbox=candidate.roi_bbox,
        ocr_confidence=candidate.ocr_score,
        match_confidence=matched.fuzzy_score,
        detection_confidence=tile.confidence,
        needs_review=matched.needs_review,
        icon_exclusion_bbox=region_debug.icon_exclusion_bbox,
        icon_bbox=candidate.icon_bbox,
        icon_center=candidate.icon_center,
        icon_radius=candidate.icon_radius,
        expected_icon_center=candidate.expected_icon_center,
        final_icon_center=candidate.final_icon_center,
        icon_selected_by=candidate.icon_selected_by,
        initial_card_bbox=candidate.initial_card_bbox,
        card_bbox_fallback=candidate.fallback,
        rejected_icon_bboxes=candidate.rejected_icon_bboxes,
        star_roi_bbox=region_debug.star_roi_bbox,
        star_slot_bboxes=region_debug.star_slot_bboxes,
        ocr_preprocess_mode=candidate.preprocess_mode,
        ocr_roi_profile=candidate.roi_profile,
        fuzzy_candidate=matched.fuzzy_candidate,
        fuzzy_score=matched.fuzzy_score,
        ocr_score=candidate.ocr_score,
        ocr_elapsed_ms=candidate.ocr_elapsed_ms,
        fallback_recommended=matched.fallback_recommended,
    )


def _card_debug_from_candidate(candidate: FactorListOcrCandidate) -> FactorListCardDebug:
    return FactorListCardDebug(
        card_bbox=candidate.card_bbox,
        initial_card_bbox=candidate.initial_card_bbox,
        icon_bbox=candidate.icon_bbox,
        icon_center=candidate.icon_center,
        icon_radius=candidate.icon_radius,
        fallback=candidate.fallback,
        rejected_icon_bboxes=candidate.rejected_icon_bboxes,
        expected_icon_center=candidate.expected_icon_center,
        final_icon_center=candidate.final_icon_center,
        icon_selected_by=candidate.icon_selected_by,
    )


def _recognize_candidate_with_engine(
    candidate: FactorListOcrCandidate,
    engine: FactorListOCRLike,
    *,
    preprocess_suffix: str,
) -> FactorListOcrCandidate:
    started = time.perf_counter()
    recognize_with_score = getattr(engine, "recognize_with_score", None)
    if callable(recognize_with_score):
        raw_name, score = recognize_with_score(candidate.preprocessed_crop)
    else:
        raw_name = engine.recognize(candidate.preprocessed_crop)
        score = None
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return replace(
        candidate,
        preprocess_mode=f"{candidate.preprocess_mode}+{preprocess_suffix}",
        ocr_raw=str(raw_name),
        ocr_score=score,
        ocr_elapsed_ms=elapsed_ms,
    )


def _ocr_roi_profiles(options: FactorOcrOptions) -> tuple[str, ...]:
    return tuple(options.ocr_roi_profiles) or (options.crop_variant,)


def _ocr_preprocess_modes(options: FactorOcrOptions) -> tuple[str, ...]:
    if not options.preprocess_crop:
        return ("raw_upscaled",)
    return tuple(options.ocr_preprocess_modes) or ("raw_upscaled",)


def _tile_key(tile: FactorListTile) -> tuple[str, int]:
    return tile.role, tile.order


def _group_candidates_by_tile(
    candidates: Sequence[FactorListOcrCandidate],
) -> dict[tuple[str, int], list[FactorListOcrCandidate]]:
    grouped: dict[tuple[str, int], list[FactorListOcrCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(_tile_key(candidate.tile), []).append(candidate)
    return grouped


def _select_best_candidate(
    tile: FactorListTile,
    candidates: Sequence[FactorListOcrCandidate],
    options: FactorOcrOptions,
) -> _SelectedTileOcr:
    matcher = FactorNameMatcher()
    scores = []
    for candidate in candidates:
        scores.append(
            score_factor_name_ocr(
                FactorNameOcrInput(
                    raw_text=candidate.ocr_raw,
                    category=candidate.tile.color,
                    roi_profile=candidate.roi_profile,
                    preprocess_mode=candidate.preprocess_mode,
                    ocr_score=candidate.ocr_score,
                    elapsed_ms=candidate.ocr_elapsed_ms,
                ),
                matcher=matcher,
                auto_threshold=options.fuzzy_match_threshold,
                review_threshold_value=options.fuzzy_review_threshold,
            )
        )

    _best_score, selected_scores = select_best_factor_name_ocr(scores)
    matched_candidates: list[_MatchedOcrCandidate] = []
    for candidate, score in zip(candidates, selected_scores):
        matched_candidates.append(
            _MatchedOcrCandidate(
                candidate=candidate,
                normalized=score.normalized_text,
                fuzzy_candidate=score.canonical_name,
                fuzzy_score=score.match_score,
                selected=score.selected,
                needs_review=score.needs_review,
                fallback_recommended=score.fallback_recommended,
            )
        )

    best_index = max(range(len(matched_candidates)), key=lambda index: int(matched_candidates[index].selected))
    best = matched_candidates[best_index]
    final_name = best.fuzzy_candidate if not best.needs_review and best.fuzzy_candidate else best.normalized
    return _SelectedTileOcr(
        tile=replace(
            tile,
            raw_name=best.candidate.ocr_raw,
            final_name=final_name,
            needs_review=best.needs_review,
        ),
        matched=best,
        candidates=matched_candidates,
    )


def _candidate_rank(match: _MatchedOcrCandidate, options: FactorOcrOptions) -> tuple[float, int, int, int]:
    score = match.fuzzy_score if match.fuzzy_score is not None else 0.0
    profile_rank = _rank_in_options(match.candidate.roi_profile, _ocr_roi_profiles(options))
    mode_rank = _rank_in_options(match.candidate.preprocess_mode, _ocr_preprocess_modes(options))
    return score, int(bool(match.normalized)), -profile_rank, -mode_rank


def _rank_in_options(value: str, values: Sequence[str]) -> int:
    try:
        return list(values).index(value)
    except ValueError:
        return len(values)


def _fallback_candidate(image_bgr: np.ndarray, tile: FactorListTile) -> FactorListOcrCandidate:
    x0, y0, x1, y1 = _clip_nonempty_bbox(tile.bbox, image_bgr.shape)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
    return FactorListOcrCandidate(
        tile=tile,
        roi_profile="fallback",
        preprocess_mode="raw_upscaled",
        card_bbox=(x0, y0, x1, y1),
        initial_card_bbox=(x0, y0, x1, y1),
        icon_bbox=None,
        icon_center=None,
        icon_radius=None,
        roi_bbox=(x0, y0, x1, y1),
        star_roi_bbox=(x0, y0, x1, y1),
        raw_crop=crop,
        upscaled_crop=crop,
        preprocessed_crop=crop,
        fallback=True,
        ocr_raw=tile.raw_name,
    )


def _apply_factor_to_uma(uma: UmaFactors, factor: RecognizedFactor) -> None:
    name = factor.normalized_name or factor.raw_name.strip()
    if not name:
        return

    star = max(0, min(3, int(factor.stars)))
    if factor.category == "blue" and not uma.blue_type:
        uma.blue_type = name
        uma.blue_star = star
    elif factor.category == "red" and not uma.red_type:
        uma.red_type = name
        uma.red_star = star
    elif factor.category == "green" and not uma.green_name:
        uma.green_name = name
        uma.green_star = star
    else:
        uma.skills.append(FactorEntry(color=factor.category or "white", name=name, star=star))


def _normalize_factor_name(raw_name: str) -> str:
    return normalize_factor_name(raw_name)

def _match_factor_name(
    normalized_name: str,
    category: str | None,
) -> tuple[str | None, float | None]:
    return match_factor_name(normalize_factor_name(normalized_name), category)


def _clear_factor_name_caches() -> None:
    clear_factor_name_matcher_caches()
    _factor_name_candidates.cache_clear()
    _all_factor_names.cache_clear()
    _skill_factor_names.cache_clear()
    _skill_master_names.cache_clear()


def _is_fuzzy_match_plausible(normalized_name: str, candidate_normalized: str) -> bool:
    query = re.sub(r"[^\wぁ-んァ-ン一-龯○ー・!！?？]", "", normalized_name)
    candidate = re.sub(r"[^\wぁ-んァ-ン一-龯○ー・!！?？]", "", candidate_normalized)
    if not query or not candidate:
        return False
    if len(query) <= 1:
        return False
    if len(candidate) >= 7 and len(query) / len(candidate) < 0.60:
        return False
    return True


def _devoice_japanese(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if ch not in {"\u3099", "\u309a"})
    return unicodedata.normalize("NFC", stripped)


@lru_cache(maxsize=8)
def _factor_name_candidates(category: str | None) -> tuple[tuple[str, str], ...]:
    names = _factor_names_for_category(category)
    return tuple((name, _normalize_factor_name(name)) for name in names)


def _factor_names_for_category(category: str | None) -> tuple[str, ...]:
    if category == "blue":
        return tuple(BLUE_FACTOR_TYPES)
    if category == "red":
        return tuple(RED_FACTOR_TYPES)
    if category == "green":
        return tuple(green_factor_names())
    if category == "white":
        return _skill_factor_names()
    return _all_factor_names()


@lru_cache(maxsize=1)
def _all_factor_names() -> tuple[str, ...]:
    labels = load_labels()
    names = labels.get("factor.name", [])
    return tuple(
        dict.fromkeys(
            [str(name) for name in names if str(name).strip()]
            + list(_skill_master_names())
        )
    )


@lru_cache(maxsize=1)
def _skill_factor_names() -> tuple[str, ...]:
    excluded = set(BLUE_FACTOR_TYPES) | set(RED_FACTOR_TYPES) | set(green_factor_names())
    return tuple(
        dict.fromkeys(
            [name for name in _all_factor_names() if name not in excluded]
            + [name for name in _skill_master_names() if name not in excluded]
        )
    )


@lru_cache(maxsize=1)
def _skill_master_names() -> tuple[str, ...]:
    return tuple(load_skill_master_names())


def _write_debug_artifacts(
    image_bgr: np.ndarray,
    factors: list[RecognizedFactor],
    debug_candidates: list[_MatchedOcrCandidate],
    options: FactorOcrOptions,
) -> Path | None:
    if options.debug_dir is None:
        return None

    debug_dir = options.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    stitched_image_path = debug_dir / "stitched.png"
    cv2.imwrite(str(stitched_image_path), image_bgr)
    (debug_dir / "factor_ocr_result.json").write_text(
        json.dumps(
            {
                "factor_count": len(factors),
                "factors": [factor.to_dict() for factor in factors],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_debug_ocr_crops(debug_dir, image_bgr, debug_candidates)

    if options.enable_overlay:
        from ..debug.overlay import write_factor_ocr_overlay

        write_factor_ocr_overlay(debug_dir / "factor_ocr_overlay.png", image_bgr, factors)
    return stitched_image_path


def _write_debug_ocr_crops(
    debug_dir: Path,
    image_bgr: np.ndarray,
    candidates: list[_MatchedOcrCandidate],
) -> None:
    crop_dir = debug_dir / "ocr_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    csv_path = debug_dir / "factor_ocr_debug.csv"
    card_csv_path = debug_dir / "factor_card_detection_debug.csv"
    rows: list[dict[str, object]] = []
    card_rows: list[dict[str, object]] = []
    seen_card_keys: set[tuple[str, int]] = set()
    seen_roi_keys: set[tuple[str, int, str]] = set()

    for match in sorted(
        candidates,
        key=lambda item: (_role_sort_key(item.candidate.tile.role), item.candidate.tile.order),
    ):
        candidate = match.candidate
        tile = candidate.tile
        stem = f"tile_{_tile_debug_index(tile):03d}_{tile.role}_{tile.order:03d}"
        tile_key = _tile_key(tile)
        roi_key = tile_key + (candidate.roi_profile,)

        if tile_key not in seen_card_keys:
            card_x0, card_y0, card_x1, card_y1 = candidate.card_bbox
            card_crop = image_bgr[card_y0:card_y1, card_x0:card_x1]
            if card_crop.size != 0:
                cv2.imwrite(str(crop_dir / f"{stem}_card.png"), card_crop)
            card_rows.append(_card_detection_debug_row(image_bgr, candidate))
            seen_card_keys.add(tile_key)

        if roi_key not in seen_roi_keys:
            cv2.imwrite(
                str(crop_dir / f"{stem}_roi_raw_{candidate.roi_profile}.png"),
                candidate.raw_crop,
            )
            cv2.imwrite(
                str(crop_dir / f"{stem}_roi_upscaled_{candidate.roi_profile}.png"),
                candidate.upscaled_crop,
            )
            star_x0, star_y0, star_x1, star_y1 = candidate.star_roi_bbox
            star_crop = image_bgr[star_y0:star_y1, star_x0:star_x1]
            if star_crop.size != 0:
                cv2.imwrite(
                    str(crop_dir / f"{stem}_star_roi_{candidate.roi_profile}.png"),
                    star_crop,
                )
            seen_roi_keys.add(roi_key)

        cv2.imwrite(
            str(
                crop_dir
                / f"{stem}_roi_preprocessed_{candidate.roi_profile}_{candidate.preprocess_mode}.png"
            ),
            candidate.preprocessed_crop,
        )
        crop_h, crop_w = candidate.raw_crop.shape[:2]
        rows.append(
            {
                "card_id": f"{tile.role}_{tile.order}",
                "role": tile.role,
                "index": tile.order,
                "card_bbox": _format_bbox(candidate.card_bbox),
                "initial_card_bbox": _format_bbox(candidate.initial_card_bbox),
                "icon_bbox": _format_optional_bbox(candidate.icon_bbox),
                "icon_center": _format_optional_point(candidate.icon_center),
                "icon_radius": "" if candidate.icon_radius is None else str(candidate.icon_radius),
                "expected_icon_center": _format_optional_point(candidate.expected_icon_center),
                "final_icon_center": _format_optional_point(candidate.final_icon_center),
                "icon_selected_by": candidate.icon_selected_by,
                "fallback": str(candidate.fallback),
                "roi_profile": candidate.roi_profile,
                "roi_bbox": _format_bbox(candidate.roi_bbox),
                "star_roi_bbox": _format_bbox(candidate.star_roi_bbox),
                "crop_size": f"{crop_w}x{crop_h}",
                "preprocess_mode": candidate.preprocess_mode,
                "profile": candidate.roi_profile,
                "raw_text": candidate.ocr_raw,
                "ocr_raw": candidate.ocr_raw,
                "normalized": match.normalized,
                "normalized_text": match.normalized,
                "canonical_name": match.fuzzy_candidate or "",
                "fuzzy_candidate": match.fuzzy_candidate or "",
                "match_score": "" if match.fuzzy_score is None else f"{match.fuzzy_score:.4f}",
                "fuzzy_score": "" if match.fuzzy_score is None else f"{match.fuzzy_score:.4f}",
                "ocr_score": ""
                if candidate.ocr_score is None
                else f"{candidate.ocr_score:.4f}",
                "elapsed_ms": ""
                if candidate.ocr_elapsed_ms is None
                else f"{candidate.ocr_elapsed_ms:.2f}",
                "selected": str(match.selected),
                "needs_review": str(match.needs_review),
                "fallback_recommended": str(match.fallback_recommended),
            }
        )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "card_id",
                "role",
                "index",
                "card_bbox",
                "initial_card_bbox",
                "icon_bbox",
                "icon_center",
                "icon_radius",
                "expected_icon_center",
                "final_icon_center",
                "icon_selected_by",
                "fallback",
                "roi_profile",
                "roi_bbox",
                "star_roi_bbox",
                "crop_size",
                "profile",
                "preprocess_mode",
                "raw_text",
                "ocr_raw",
                "normalized_text",
                "normalized",
                "canonical_name",
                "fuzzy_candidate",
                "match_score",
                "fuzzy_score",
                "ocr_score",
                "elapsed_ms",
                "selected",
                "needs_review",
                "fallback_recommended",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with card_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role",
                "index",
                "input_image_size",
                "detection_image_size",
                "card_bbox_global",
                "card_bbox_local",
                "rejected_icon_count",
                "icon_radius",
                "selected_by",
                "fallback_used",
                "expected_icon_x_from_card",
                "expected_icon_y_from_card",
                "final_icon_x",
                "final_icon_y",
                "delta_x_final_to_expected",
                "delta_y_final_to_expected",
                "text_roi",
                "star_roi",
            ],
        )
        writer.writeheader()
        writer.writerows(card_rows)

    _write_ocr_contact_sheet(debug_dir / "factor_ocr_contact_sheet.png", candidates)


def _write_ocr_contact_sheet(path: Path, candidates: list[_MatchedOcrCandidate]) -> None:
    selected = [match for match in candidates if match.selected]
    if not selected:
        return

    thumb_w = 360
    row_h = 150
    label_w = 520
    sheet = np.full(
        (row_h * len(selected), thumb_w + label_w, 3),
        255,
        dtype=np.uint8,
    )
    for row, match in enumerate(
        sorted(
            selected,
            key=lambda item: (
                _role_sort_key(item.candidate.tile.role),
                item.candidate.tile.order,
            ),
        )
    ):
        candidate = match.candidate
        crop = candidate.preprocessed_crop
        if crop is not None and crop.size:
            crop = cv2.resize(
                crop,
                _fit_size(crop.shape[1], crop.shape[0], thumb_w - 20, row_h - 20),
                interpolation=cv2.INTER_AREA,
            )
            y = row * row_h + 10
            x = 10
            sheet[y : y + crop.shape[0], x : x + crop.shape[1]] = crop

    pil_sheet = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_sheet)
    font = _load_contact_sheet_font(17)
    line_h = _contact_sheet_line_height(draw, font) + 3

    for row, match in enumerate(
        sorted(
            selected,
            key=lambda item: (
                _role_sort_key(item.candidate.tile.role),
                item.candidate.tile.order,
            ),
        )
    ):
        candidate = match.candidate
        x0 = thumb_w + 10
        y0 = row * row_h + 28
        score_text = "" if match.fuzzy_score is None else f"{match.fuzzy_score:.3f}"
        label_lines = [
            f"{candidate.tile.role} #{candidate.tile.order} {candidate.tile.color} s{candidate.tile.star}",
            f"profile={candidate.roi_profile} prep={candidate.preprocess_mode}",
            f"raw={candidate.ocr_raw[:36]}",
            f"norm={match.normalized[:36]}",
            f"canon={(match.fuzzy_candidate or '')[:36]} score={score_text}",
            f"review={match.needs_review} fallback={match.fallback_recommended}",
        ]
        for index, line in enumerate(label_lines):
            draw.text(
                (x0, y0 + index * line_h),
                line,
                font=font,
                fill=(30, 30, 30),
            )
    pil_sheet.save(path)


def _load_contact_sheet_font(size: int) -> ImageFont.ImageFont:
    font_paths = [
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/YuGothR.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def _contact_sheet_line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), "Agあ漢", font=font)
    return max(1, bbox[3] - bbox[1])


def _fit_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / max(1, width), max_height / max(1, height), 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _role_sort_key(role: str) -> int:
    return {"parent": 0, "ancestor1": 1, "ancestor2": 2}.get(role, 99)


def _tile_debug_index(tile: FactorListTile) -> int:
    return _role_sort_key(tile.role) * 1000 + tile.order


def _card_detection_debug_row(
    image_bgr: np.ndarray,
    candidate: FactorListOcrCandidate,
) -> dict[str, object]:
    tile = candidate.tile
    image_height, image_width = image_bgr.shape[:2]
    detection_width = BASE_WIDTH
    detection_height = int(round(image_height * (BASE_WIDTH / image_width))) if image_width else image_height
    card_x0, card_y0, card_x1, card_y1 = candidate.card_bbox
    expected_center = candidate.expected_icon_center
    final_center = candidate.final_icon_center or candidate.icon_center
    return {
        "role": tile.role,
        "index": tile.order,
        "input_image_size": f"{image_width}x{image_height}",
        "detection_image_size": f"{detection_width}x{detection_height}",
        "card_bbox_global": _format_bbox(candidate.card_bbox),
        "card_bbox_local": _format_bbox((0, 0, max(0, card_x1 - card_x0), max(0, card_y1 - card_y0))),
        "rejected_icon_count": str(len(candidate.rejected_icon_bboxes)),
        "icon_radius": "" if candidate.icon_radius is None else str(candidate.icon_radius),
        "selected_by": candidate.icon_selected_by,
        "fallback_used": str(
            candidate.fallback or candidate.icon_selected_by in {"fallback", "grid_expected"}
        ),
        "expected_icon_x_from_card": "" if expected_center is None else str(expected_center[0]),
        "expected_icon_y_from_card": "" if expected_center is None else str(expected_center[1]),
        "final_icon_x": "" if final_center is None else str(final_center[0]),
        "final_icon_y": "" if final_center is None else str(final_center[1]),
        "delta_x_final_to_expected": _format_delta_component(final_center, expected_center, 0),
        "delta_y_final_to_expected": _format_delta_component(final_center, expected_center, 1),
        "text_roi": _format_bbox(candidate.roi_bbox),
        "star_roi": _format_bbox(candidate.star_roi_bbox),
    }


def _format_delta_component(
    point: tuple[int, int] | None,
    expected: tuple[int, int] | None,
    index: int,
) -> str:
    if point is None or expected is None:
        return ""
    return str(int(point[index] - expected[index]))


def _format_bbox(bbox: tuple[int, int, int, int]) -> str:
    return ",".join(str(int(value)) for value in bbox)


def _format_optional_bbox(bbox: tuple[int, int, int, int] | None) -> str:
    return "" if bbox is None else _format_bbox(bbox)


def _format_optional_point(point: tuple[int, int] | None) -> str:
    return "" if point is None else ",".join(str(int(value)) for value in point)


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


def _load_image_input(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image.copy()

    path = Path(image)
    data = np.fromfile(str(path), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"image is missing or unreadable: {path}")
    return image_bgr


def _coerce_scroll_frames(frames: Sequence[ScrollFrame | np.ndarray]) -> tuple[ScrollFrame, ...]:
    coerced: list[ScrollFrame] = []
    for index, frame in enumerate(frames):
        if isinstance(frame, ScrollFrame):
            coerced.append(frame)
        else:
            coerced.append(ScrollFrame(image=frame.copy(), frame_index=index))
    return tuple(coerced)
