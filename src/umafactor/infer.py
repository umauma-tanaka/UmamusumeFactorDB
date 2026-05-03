"""Compatibility facade for inference helpers.

Phase 3 splits the implementation into focused modules while preserving the
legacy ``umafactor.infer`` import surface.
"""

from __future__ import annotations

from .recognition.model_registry import (
    ModelSpec,
    ModelValidationResult,
    describe_loaded_model_io,
    describe_predictor_io,
    get_model_spec,
    get_predictor,
    iter_model_specs,
    missing_required_models,
    required_model_paths,
    validate_required_models,
)
from .recognition.onnx_runtime import (
    FACTOR_SOFTMAX_NAME,
    FACTOR_WITH_PROBS_FILENAME,
    OnnxModelIO,
    OnnxPredictor,
    Prediction,
    _ensure_factor_with_probs,
    describe_session_io,
)
from .recognition.stars import (
    STAR_CLASS_NAMES,
    STAR_EMPTY_HSV_HI,
    STAR_EMPTY_HSV_LO,
    STAR_FALLBACK_ENV,
    STAR_GOLD_HSV_HI,
    STAR_GOLD_HSV_LO,
    STAR_SLOT_SIZE,
    _allow_missing_star_classifier,
    _get_star_session,
    _predict_star_hsv,
    _prep_star_slot,
    _star_model_path,
    predict_star,
    predict_stars_batch,
)

__all__ = [
    "FACTOR_SOFTMAX_NAME",
    "FACTOR_WITH_PROBS_FILENAME",
    "ModelSpec",
    "ModelValidationResult",
    "OnnxModelIO",
    "OnnxPredictor",
    "Prediction",
    "STAR_CLASS_NAMES",
    "STAR_EMPTY_HSV_HI",
    "STAR_EMPTY_HSV_LO",
    "STAR_FALLBACK_ENV",
    "STAR_GOLD_HSV_HI",
    "STAR_GOLD_HSV_LO",
    "STAR_SLOT_SIZE",
    "_allow_missing_star_classifier",
    "_ensure_factor_with_probs",
    "_get_star_session",
    "_predict_star_hsv",
    "_prep_star_slot",
    "_star_model_path",
    "describe_loaded_model_io",
    "describe_predictor_io",
    "describe_session_io",
    "get_model_spec",
    "get_predictor",
    "iter_model_specs",
    "missing_required_models",
    "predict_star",
    "predict_stars_batch",
    "required_model_paths",
    "validate_required_models",
]
