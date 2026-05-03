from __future__ import annotations

from dataclasses import dataclass

from umafactor import pipeline
from umafactor.app import analyzer
from umafactor.app.analyzer import UmaFactorAnalyzer
from umafactor.review import ReviewQueue
from umafactor.schema import Submission, UmaFactors


@dataclass(frozen=True)
class DummyRecognition:
    umas: list[UmaFactors]
    review: ReviewQueue


def test_analyzer_orchestrates_image_analysis_dependencies() -> None:
    calls: list[str] = []
    prepared = object()
    context = object()
    review = ReviewQueue()
    umas = [UmaFactors(character="main"), UmaFactors(), UmaFactors()]
    submission = Submission(submitter_id="tester", image_filename="factor.png")

    def fake_prepare_image(
        image_path: str,
        *,
        debug_crops_dir: str | None = None,
        auto_debug: bool = True,
    ) -> object:
        calls.append("prepare")
        assert image_path == "images/factor.png"
        assert debug_crops_dir == "debug"
        assert auto_debug is False
        return prepared

    def fake_build_context(*, skip_ocr: bool = False) -> object:
        calls.append("context")
        assert skip_ocr is True
        return context

    def fake_load_unique_skill_map() -> dict[str, str]:
        calls.append("unique_map")
        return {"unique": "character"}

    def fake_recognize_factors(
        prepared_arg: object,
        context_arg: object,
        unique_skill_to_character: dict[str, str],
    ) -> DummyRecognition:
        calls.append("recognize")
        assert prepared_arg is prepared
        assert context_arg is context
        assert unique_skill_to_character == {"unique": "character"}
        return DummyRecognition(umas=umas, review=review)

    def fake_make_submission(
        *,
        submitter_id: str,
        image_path: str,
        umas: list[UmaFactors],
    ) -> Submission:
        calls.append("submission")
        assert submitter_id == "tester"
        assert image_path == "images/factor.png"
        assert umas[0].character == "main"
        return submission

    app = UmaFactorAnalyzer(
        prepare_image=fake_prepare_image,
        build_context=fake_build_context,
        load_unique_skill_map=fake_load_unique_skill_map,
        recognize_factors=fake_recognize_factors,
        make_submission=fake_make_submission,
    )

    result = app.analyze_image(
        "images/factor.png",
        "tester",
        debug_crops_dir="debug",
        auto_debug=False,
        skip_ocr=True,
    )

    assert calls == ["prepare", "context", "unique_map", "recognize", "submission"]
    assert result == (submission, review)


def test_pipeline_analyze_image_is_thin_facade(monkeypatch) -> None:
    calls: list[tuple[str, str, str | None, bool, bool]] = []
    submission = Submission(submitter_id="tester", image_filename="factor.png")
    review = ReviewQueue()

    class DummyAnalyzer:
        def analyze_image(
            self,
            image_path: str,
            submitter_id: str,
            debug_crops_dir: str | None = None,
            auto_debug: bool = True,
            skip_ocr: bool = False,
        ) -> tuple[Submission, ReviewQueue]:
            calls.append(
                (image_path, submitter_id, debug_crops_dir, auto_debug, skip_ocr)
            )
            return submission, review

    monkeypatch.setattr(pipeline, "UmaFactorAnalyzer", DummyAnalyzer)

    assert pipeline.analyze_image(
        "images/factor.png",
        "tester",
        debug_crops_dir="debug",
        auto_debug=False,
        skip_ocr=True,
    ) == (submission, review)
    assert calls == [("images/factor.png", "tester", "debug", False, True)]


def test_app_analyzer_module_function_uses_default_analyzer(monkeypatch) -> None:
    calls: list[str] = []
    submission = Submission(submitter_id="tester", image_filename="factor.png")
    review = ReviewQueue()

    class DummyAnalyzer:
        def analyze_image(self, *args, **kwargs):
            calls.append("analyze")
            assert args == ("images/factor.png", "tester")
            assert kwargs == {
                "debug_crops_dir": None,
                "auto_debug": True,
                "skip_ocr": False,
            }
            return submission, review

    monkeypatch.setattr(analyzer, "UmaFactorAnalyzer", DummyAnalyzer)

    assert analyzer.analyze_image("images/factor.png", "tester") == (
        submission,
        review,
    )
    assert calls == ["analyze"]
