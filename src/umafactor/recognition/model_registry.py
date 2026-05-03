"""Model registry and predictor construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import model_path

if TYPE_CHECKING:
    from .onnx_runtime import OnnxPredictor


FACTOR_SOFTMAX_NAME = "onnx::ReduceMax_639"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    label_key: str | None = None
    index_output: str = "index"
    confidence_output: str = "confidence"
    extra_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelValidationResult:
    model_name: str
    path: Path
    exists: bool
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "path": str(self.path),
            "exists": self.exists,
            "required": self.required,
        }


MODEL_SPECS: dict[str, ModelSpec] = {
    "factor": ModelSpec(
        model_name="factor",
        label_key="factor.name",
        extra_outputs=(FACTOR_SOFTMAX_NAME,),
    ),
    "factor_rank": ModelSpec(
        model_name="factor_rank",
        label_key="factor_rank.name",
    ),
    "aptitude": ModelSpec(
        model_name="aptitude",
        label_key="aptitude.name",
    ),
    "character": ModelSpec(
        model_name="character",
        label_key="character.card",
        index_output="card_index",
        confidence_output="card_confidence",
    ),
}

REQUIRED_MODEL_NAMES = ("factor", "factor_rank", "character", "star_classifier")


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {model_name}") from exc


def iter_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(MODEL_SPECS.values())


def required_model_paths() -> dict[str, Path]:
    return {name: model_path(name) for name in REQUIRED_MODEL_NAMES}


def validate_required_models(
    *,
    allow_missing_star_classifier: bool = False,
) -> list[ModelValidationResult]:
    results: list[ModelValidationResult] = []
    for name, path in required_model_paths().items():
        required = not (name == "star_classifier" and allow_missing_star_classifier)
        results.append(
            ModelValidationResult(
                model_name=name,
                path=path,
                exists=path.exists(),
                required=required,
            )
        )
    return results


def missing_required_models(
    *,
    allow_missing_star_classifier: bool = False,
) -> list[ModelValidationResult]:
    return [
        result
        for result in validate_required_models(
            allow_missing_star_classifier=allow_missing_star_classifier
        )
        if result.required and not result.exists
    ]


def _build_predictor(spec: ModelSpec) -> "OnnxPredictor":
    from .onnx_runtime import OnnxPredictor

    assert spec.label_key is not None
    return OnnxPredictor(
        spec.model_name,
        spec.label_key,
        index_output=spec.index_output,
        confidence_output=spec.confidence_output,
        extra_outputs=spec.extra_outputs,
    )


@lru_cache(maxsize=None)
def get_predictor(model_name: str) -> "OnnxPredictor":
    spec = get_model_spec(model_name)
    return _build_predictor(spec)


def describe_predictor_io(model_name: str) -> dict[str, Any]:
    predictor = get_predictor(model_name)
    return predictor.describe_io().to_dict()


def describe_loaded_model_io(model_names: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    names = model_names or tuple(MODEL_SPECS)
    return {name: describe_predictor_io(name) for name in names}
