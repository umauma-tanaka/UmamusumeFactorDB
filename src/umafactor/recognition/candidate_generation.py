"""Candidate generation helpers for factor recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .candidate_fusion import (
    CandidateList,
    SourceMap,
    finalize_factor_candidates,
)
from .constants import (
    BLUE_FACTOR_TYPES,
    PERTURBATIONS_BLUE,
    PERTURBATIONS_RED,
    RED_FACTOR_TYPES,
)
from .image_crops import crop_from_original, display_crop_from_original
from ..templates import match_green_name, match_templates


class FactorPredictorLike(Protocol):
    def topk_in_category(
        self,
        crops: Sequence[np.ndarray],
        category_names: Sequence[str],
        k: int,
        use_multi_interp: bool = False,
    ) -> CandidateList:
        ...

    def topk_ensemble(self, crops: Sequence[np.ndarray], k: int) -> CandidateList:
        ...


class FactorOCRLike(Protocol):
    def recognize_red(self, img: np.ndarray) -> str:
        ...

    def recognize_blue(self, img: np.ndarray) -> str:
        ...

    def recognize(self, img: np.ndarray) -> str:
        ...

    def recognize_with_parts(self, img: np.ndarray) -> tuple[str, list[str]]:
        ...

    def match_to_green_factor_multi(
        self, text: str, fragments: Sequence[str], top_k: int = 5
    ) -> CandidateList:
        ...

    def match_to_factor(self, text: str, top_k: int = 5) -> CandidateList:
        ...


@dataclass(frozen=True)
class FactorCandidateRecognition:
    candidates: CandidateList
    sources: SourceMap
    top_name: str
    ocr_raw: str
    onnx_candidates: CandidateList
    ocr_candidates: CandidateList
    template_candidates: CandidateList


def predict_onnx_candidates(
    factor_pred: FactorPredictorLike,
    img_orig: np.ndarray,
    text_crop_norm: np.ndarray,
    bbox: tuple[int, int, int, int],
    ext_bbox: tuple[int, int, int, int],
    scale: float,
    *,
    is_blue_slot: bool,
    is_red_slot: bool,
) -> CandidateList:
    if is_blue_slot:
        crops = [
            crop_from_original(img_orig, ext_bbox, scale, dy, dx)
            for dy, dx in PERTURBATIONS_BLUE
        ]
        crops.append(text_crop_norm)
        return factor_pred.topk_in_category(crops, BLUE_FACTOR_TYPES, k=5)

    if is_red_slot:
        crops = [
            crop_from_original(img_orig, ext_bbox, scale, dy, dx)
            for dy, dx in PERTURBATIONS_RED
        ]
        crops.append(text_crop_norm)
        return factor_pred.topk_in_category(
            crops, RED_FACTOR_TYPES, k=5, use_multi_interp=True
        )

    text_crop_orig = crop_from_original(img_orig, bbox, scale)
    return factor_pred.topk_ensemble([text_crop_orig, text_crop_norm], k=5)


def recognize_ocr_candidates(
    ocr: FactorOCRLike | None,
    display_crop: np.ndarray,
    *,
    is_blue_slot: bool,
    is_red_slot: bool,
    green_adoptable: bool,
) -> tuple[str, CandidateList]:
    ocr_raw = ""
    ocr_fragments: list[str] = []
    if ocr is None:
        return ocr_raw, []

    if is_red_slot:
        ocr_raw = ocr.recognize_red(display_crop)
    elif is_blue_slot:
        ocr_raw = ocr.recognize_blue(display_crop)
    elif green_adoptable:
        ocr_raw, ocr_fragments = ocr.recognize_with_parts(display_crop)
    else:
        ocr_raw = ocr.recognize(display_crop)

    if green_adoptable:
        candidates = ocr.match_to_green_factor_multi(ocr_raw, ocr_fragments, top_k=5)
    else:
        candidates = ocr.match_to_factor(ocr_raw, top_k=5)
    return ocr_raw, candidates


def filter_slot_candidates(
    onnx_candidates: CandidateList,
    ocr_candidates: CandidateList,
    *,
    is_blue_slot: bool,
    is_red_slot: bool,
    green_adoptable: bool,
    green_name_set: set[str],
) -> tuple[CandidateList, CandidateList]:
    if is_blue_slot:
        ocr_candidates = [(n, s) for n, s in ocr_candidates if n in BLUE_FACTOR_TYPES]
    elif is_red_slot:
        ocr_candidates = [(n, s) for n, s in ocr_candidates if n in RED_FACTOR_TYPES]
    elif not green_adoptable:
        onnx_candidates = [(n, s) for n, s in onnx_candidates if n not in green_name_set]
    return onnx_candidates, ocr_candidates


def match_template_candidates(
    display_crop: np.ndarray,
    img_orig: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    *,
    is_red_slot: bool,
    is_blue_slot: bool,
    green_adoptable: bool,
) -> CandidateList:
    if is_red_slot:
        return match_templates(display_crop, "red")[:5]
    if is_blue_slot:
        return match_templates(display_crop, "blue")[:5]
    if not green_adoptable:
        return []

    x0, y0, x1, y1 = bbox
    name_x1 = x0 + int((x1 - x0) * 0.85)
    name_crop = display_crop_from_original(
        img_orig, (x0, y0, name_x1, y1), scale, pad_y_norm=2
    )
    return match_green_name(name_crop)[:5]


def recognize_factor_candidates(
    factor_pred: FactorPredictorLike,
    ocr: FactorOCRLike | None,
    img_orig: np.ndarray,
    text_crop_norm: np.ndarray,
    display_crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    *,
    is_blue_slot: bool,
    is_red_slot: bool,
    green_adoptable: bool,
    green_name_set: set[str],
    ext_bbox: tuple[int, int, int, int] | None = None,
) -> FactorCandidateRecognition:
    ext_bbox = bbox if ext_bbox is None else ext_bbox
    onnx_candidates = predict_onnx_candidates(
        factor_pred,
        img_orig,
        text_crop_norm,
        bbox,
        ext_bbox,
        scale,
        is_blue_slot=is_blue_slot,
        is_red_slot=is_red_slot,
    )
    ocr_raw, ocr_candidates = recognize_ocr_candidates(
        ocr,
        display_crop,
        is_blue_slot=is_blue_slot,
        is_red_slot=is_red_slot,
        green_adoptable=green_adoptable,
    )
    onnx_candidates, ocr_candidates = filter_slot_candidates(
        onnx_candidates,
        ocr_candidates,
        is_blue_slot=is_blue_slot,
        is_red_slot=is_red_slot,
        green_adoptable=green_adoptable,
        green_name_set=green_name_set,
    )
    template_candidates = match_template_candidates(
        display_crop,
        img_orig,
        bbox,
        scale,
        is_red_slot=is_red_slot,
        is_blue_slot=is_blue_slot,
        green_adoptable=green_adoptable,
    )
    final_candidates = finalize_factor_candidates(
        onnx_candidates,
        ocr_candidates,
        template_candidates,
        green_adoptable=green_adoptable,
    )

    return FactorCandidateRecognition(
        candidates=final_candidates.candidates,
        sources=final_candidates.sources,
        top_name=final_candidates.top_name,
        ocr_raw=ocr_raw,
        onnx_candidates=onnx_candidates,
        ocr_candidates=ocr_candidates,
        template_candidates=template_candidates,
    )
