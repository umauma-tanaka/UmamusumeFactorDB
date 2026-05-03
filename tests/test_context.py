from __future__ import annotations

from dataclasses import dataclass

from umafactor.recognition import context
from umafactor.recognition.context import build_recognition_context


@dataclass
class DummyOCR:
    _green_factor_names: list[str]


def test_build_recognition_context_loads_predictors_and_ocr(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    predictors = {
        "factor": object(),
        "factor_rank": object(),
        "character": object(),
    }
    ocr = DummyOCR(["green-a", "green-b"])

    def fake_get_predictor(model_name: str):
        calls.append(("predictor", model_name))
        return predictors[model_name]

    def fake_get_ocr() -> DummyOCR:
        calls.append(("ocr", ""))
        return ocr

    monkeypatch.setattr(context, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(context, "get_ocr", fake_get_ocr)
    monkeypatch.setattr(
        context,
        "green_factor_names",
        lambda: (_ for _ in ()).throw(AssertionError("should not load fallback names")),
    )

    result = build_recognition_context(skip_ocr=False)

    assert calls == [
        ("predictor", "factor"),
        ("predictor", "factor_rank"),
        ("predictor", "character"),
        ("ocr", ""),
    ]
    assert result.factor_pred is predictors["factor"]
    assert result.rank_pred is predictors["factor_rank"]
    assert result.char_pred is predictors["character"]
    assert result.ocr is ocr
    assert result.green_name_set == {"green-a", "green-b"}


def test_build_recognition_context_skips_ocr_and_uses_config_green_names(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []
    predictors = {
        "factor": object(),
        "factor_rank": object(),
        "character": object(),
    }

    def fake_get_predictor(model_name: str):
        calls.append(("predictor", model_name))
        return predictors[model_name]

    def fake_green_factor_names() -> list[str]:
        calls.append(("green_names", ""))
        return ["green-config"]

    monkeypatch.setattr(context, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(
        context,
        "get_ocr",
        lambda: (_ for _ in ()).throw(AssertionError("get_ocr should not be called")),
    )
    monkeypatch.setattr(context, "green_factor_names", fake_green_factor_names)

    result = build_recognition_context(skip_ocr=True)

    assert calls == [
        ("predictor", "factor"),
        ("predictor", "factor_rank"),
        ("predictor", "character"),
        ("green_names", ""),
    ]
    assert result.factor_pred is predictors["factor"]
    assert result.rank_pred is predictors["factor_rank"]
    assert result.char_pred is predictors["character"]
    assert result.ocr is None
    assert result.green_name_set == {"green-config"}
