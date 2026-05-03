from __future__ import annotations

from umafactor import infer
from umafactor.recognition import model_registry, onnx_runtime, stars


def test_infer_reexports_runtime_api() -> None:
    assert infer.Prediction is onnx_runtime.Prediction
    assert infer.OnnxPredictor is onnx_runtime.OnnxPredictor
    assert infer.OnnxModelIO is onnx_runtime.OnnxModelIO
    assert infer.FACTOR_SOFTMAX_NAME == onnx_runtime.FACTOR_SOFTMAX_NAME


def test_infer_reexports_model_registry_api() -> None:
    assert infer.get_predictor is model_registry.get_predictor
    assert infer.get_model_spec is model_registry.get_model_spec
    assert infer.validate_required_models is model_registry.validate_required_models
    assert infer.describe_loaded_model_io is model_registry.describe_loaded_model_io


def test_infer_reexports_star_classifier_api() -> None:
    assert infer.predict_star is stars.predict_star
    assert infer.predict_stars_batch is stars.predict_stars_batch
    assert infer.STAR_CLASS_NAMES == stars.STAR_CLASS_NAMES
