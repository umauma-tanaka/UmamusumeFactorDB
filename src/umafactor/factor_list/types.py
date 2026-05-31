"""Public data types for the factor-list OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..recognition.ocr_protocol import (
    DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
    DEFAULT_OCR_MAX_UPSCALE,
    DEFAULT_OCR_MIN_HEIGHT,
    DEFAULT_OCR_MIN_WIDTH,
    DEFAULT_OCR_SHARPEN_STRENGTH,
    DEFAULT_OCR_SHEET_MAX_SIDE,
    FactorListCardDetectorMode,
    FactorListOcrCropTarget,
    FactorListOcrExecutionMode,
)


FactorOcrEngineMode = Literal["paddle", "rapidocr"]
FactorOcrPaddleMode = Literal["recognition", "ocr"]
FactorOcrRole = Literal["parent", "ancestor1", "ancestor2"]
FactorOcrFallbackMode = Literal["none", "paddle"]


@dataclass(frozen=True)
class FactorOcrOptions:
    """Options for the standalone factor-list OCR pipeline.

    The defaults match the live capture evaluation path while keeping all
    debug artifacts opt-in.  Paddle cache/model placement remains configurable
    so exe packaging can later redirect it without changing the pipeline API.
    """

    use_paddle: bool = True
    debug_dir: Path | None = None
    enable_overlay: bool = False
    enable_stitch: bool = True
    ocr_mode: FactorOcrEngineMode = "rapidocr"
    card_detector: FactorListCardDetectorMode = "card-body"
    use_scrollbar_hint: bool = False
    paddle_cache_dir: Path | None = None
    paddle_lang: str = "japan"
    paddle_mode: FactorOcrPaddleMode = "recognition"
    paddle_det_limit_side_len: int | None = 128
    paddle_det_limit_type: str | None = "min"
    paddle_det_thresh: float | None = None
    paddle_det_box_thresh: float | None = None
    paddle_det_unclip_ratio: float | None = None
    paddle_rec_score_thresh: float | None = None
    rapidocr_model_root_dir: Path | None = None
    rapidocr_text_score: float | None = None
    rapidocr_ocr_version: str = "PP-OCRv4"
    rapidocr_lang_type: str = "japan"
    rapidocr_model_type: str = "mobile"
    rapidocr_rec_img_shape: tuple[int, int, int] = (3, 48, 480)
    rapidocr_rec_batch_num: int = 12
    ocr_crop_target: FactorListOcrCropTarget = "name"
    crop_variant: str = "body_name"
    name_roi_x1_ratio: float = 0.13
    name_roi_x2_margin_ratio: float = 0.03
    name_roi_y1_ratio: float = 0.05
    name_roi_y2_ratio: float = 0.68
    ocr_roi_profiles: tuple[str, ...] = ("body_name",)
    ocr_preprocess_modes: tuple[str, ...] = (
        "raw_upscaled",
        "gray_sharpen",
        "color_text_safe",
    )
    fuzzy_match_threshold: float = 0.92
    fuzzy_review_threshold: float = 0.82
    fallback_ocr_mode: FactorOcrFallbackMode = "none"
    preprocess_crop: bool = True
    ocr_min_crop_width: int = DEFAULT_OCR_MIN_WIDTH
    ocr_min_crop_height: int = DEFAULT_OCR_MIN_HEIGHT
    ocr_max_upscale: float = DEFAULT_OCR_MAX_UPSCALE
    ocr_sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH
    ocr_contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT
    ocr_execution_mode: FactorListOcrExecutionMode = "batch"
    ocr_batch_size: int = 12
    ocr_canvas_padding: int = 24
    ocr_sheet_max_side: int = DEFAULT_OCR_SHEET_MAX_SIDE
    ocr_sheet_columns: int | None = None


@dataclass(frozen=True)
class RecognizedFactor:
    role: FactorOcrRole
    order: int
    row: int | None
    col: int | None
    raw_name: str
    normalized_name: str | None
    category: str | None
    stars: int
    bbox: tuple[int, int, int, int] | None
    ocr_bbox: tuple[int, int, int, int] | None
    ocr_confidence: float | None
    match_confidence: float | None
    detection_confidence: float | None
    needs_review: bool
    icon_exclusion_bbox: tuple[int, int, int, int] | None = None
    icon_bbox: tuple[int, int, int, int] | None = None
    icon_center: tuple[int, int] | None = None
    icon_radius: int | None = None
    expected_icon_center: tuple[int, int] | None = None
    final_icon_center: tuple[int, int] | None = None
    icon_selected_by: str = ""
    initial_card_bbox: tuple[int, int, int, int] | None = None
    card_bbox_fallback: bool = False
    rejected_icon_bboxes: tuple[tuple[int, int, int, int], ...] = ()
    star_roi_bbox: tuple[int, int, int, int] | None = None
    star_slot_bboxes: tuple[tuple[int, int, int, int], ...] = ()
    ocr_preprocess_mode: str | None = None
    ocr_roi_profile: str | None = None
    fuzzy_candidate: str | None = None
    fuzzy_score: float | None = None
    ocr_score: float | None = None
    ocr_elapsed_ms: float | None = None
    fallback_recommended: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "order": self.order,
            "row": self.row,
            "col": self.col,
            "raw_name": self.raw_name,
            "normalized_name": self.normalized_name,
            "category": self.category,
            "stars": self.stars,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "ocr_bbox": list(self.ocr_bbox) if self.ocr_bbox is not None else None,
            "ocr_confidence": self.ocr_confidence,
            "match_confidence": self.match_confidence,
            "detection_confidence": self.detection_confidence,
            "needs_review": self.needs_review,
            "icon_exclusion_bbox": list(self.icon_exclusion_bbox)
            if self.icon_exclusion_bbox is not None
            else None,
            "icon_bbox": list(self.icon_bbox) if self.icon_bbox is not None else None,
            "icon_center": list(self.icon_center) if self.icon_center is not None else None,
            "icon_radius": self.icon_radius,
            "expected_icon_center": list(self.expected_icon_center)
            if self.expected_icon_center is not None
            else None,
            "final_icon_center": list(self.final_icon_center)
            if self.final_icon_center is not None
            else None,
            "icon_selected_by": self.icon_selected_by,
            "initial_card_bbox": list(self.initial_card_bbox)
            if self.initial_card_bbox is not None
            else None,
            "card_bbox_fallback": self.card_bbox_fallback,
            "rejected_icon_bboxes": [list(bbox) for bbox in self.rejected_icon_bboxes],
            "star_roi_bbox": list(self.star_roi_bbox)
            if self.star_roi_bbox is not None
            else None,
            "star_slot_bboxes": [list(bbox) for bbox in self.star_slot_bboxes],
            "ocr_preprocess_mode": self.ocr_preprocess_mode,
            "ocr_roi_profile": self.ocr_roi_profile,
            "fuzzy_candidate": self.fuzzy_candidate,
            "fuzzy_score": self.fuzzy_score,
            "ocr_score": self.ocr_score,
            "ocr_elapsed_ms": self.ocr_elapsed_ms,
            "fallback_recommended": self.fallback_recommended,
        }


@dataclass(frozen=True)
class FactorOcrResult:
    factors: list[RecognizedFactor]
    stitched_image_path: Path | None
    debug_dir: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "factors": [factor.to_dict() for factor in self.factors],
            "stitched_image_path": str(self.stitched_image_path)
            if self.stitched_image_path is not None
            else None,
            "debug_dir": str(self.debug_dir) if self.debug_dir is not None else None,
        }
