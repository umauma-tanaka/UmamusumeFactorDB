from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from umafactor.recognition import model_registry
from umafactor.recognition.onnx_runtime import FACTOR_SOFTMAX_NAME, OnnxModelIO


def test_model_specs_cover_expected_predictor_models() -> None:
    factor = model_registry.get_model_spec("factor")
    character = model_registry.get_model_spec("character")

    assert factor.label_key == "factor.name"
    assert factor.extra_outputs == (FACTOR_SOFTMAX_NAME,)
    assert character.label_key == "character.card"
    assert character.index_output == "card_index"
    assert character.confidence_output == "card_confidence"
    assert model_registry.REQUIRED_MODEL_NAMES == (
        "factor",
        "factor_rank",
        "character",
        "star_classifier",
    )


def test_get_model_spec_rejects_unknown_model() -> None:
    with pytest.raises(KeyError):
        model_registry.get_model_spec("missing")


def test_get_predictor_builds_from_registry_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_build_predictor(spec: model_registry.ModelSpec) -> object:
        calls.append(
            (
                (spec.model_name, spec.label_key),
                {
                    "index_output": spec.index_output,
                    "confidence_output": spec.confidence_output,
                    "extra_outputs": spec.extra_outputs,
                },
            )
        )
        return object()

    monkeypatch.setattr(model_registry, "_build_predictor", fake_build_predictor)
    model_registry.get_predictor.cache_clear()
    try:
        model_registry.get_predictor("factor")
        model_registry.get_predictor("character")
    finally:
        model_registry.get_predictor.cache_clear()

    assert calls == [
        (
            ("factor", "factor.name"),
            {
                "index_output": "index",
                "confidence_output": "confidence",
                "extra_outputs": (FACTOR_SOFTMAX_NAME,),
            },
        ),
        (
            ("character", "character.card"),
            {
                "index_output": "card_index",
                "confidence_output": "card_confidence",
                "extra_outputs": (),
            },
        ),
    ]


def test_validate_required_models_allows_star_classifier_to_be_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = {"factor", "factor_rank", "character"}

    def fake_model_path(model_name: str) -> Path:
        path = Path("outputs") / "test_model_registry" / f"{model_name}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        if model_name in existing:
            path.write_bytes(b"onnx")
        elif path.exists():
            path.unlink()
        return path

    monkeypatch.setattr(model_registry, "model_path", fake_model_path)

    strict_missing = model_registry.missing_required_models()
    optional_missing = model_registry.missing_required_models(
        allow_missing_star_classifier=True,
    )

    assert [result.model_name for result in strict_missing] == ["star_classifier"]
    assert optional_missing == []


def test_describe_loaded_model_io_uses_predictor_descriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_describe_predictor_io(model_name: str) -> dict[str, Any]:
        return {"input_name": f"{model_name}_input", "output_names": ["index"]}

    monkeypatch.setattr(model_registry, "describe_predictor_io", fake_describe_predictor_io)

    assert model_registry.describe_loaded_model_io(("factor", "character")) == {
        "factor": {"input_name": "factor_input", "output_names": ["index"]},
        "character": {"input_name": "character_input", "output_names": ["index"]},
    }


def test_model_io_to_dict_converts_tuples() -> None:
    model_io = OnnxModelIO(
        input_name="images",
        input_shape=("batch", 16, 168, 3),
        output_names=("index", "confidence"),
    )

    assert model_io.to_dict() == {
        "input_name": "images",
        "input_shape": ["batch", 16, 168, 3],
        "output_names": ["index", "confidence"],
    }
