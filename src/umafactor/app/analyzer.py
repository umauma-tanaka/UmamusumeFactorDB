"""Application-level image analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import load_unique_skill_to_character
from ..recognition.context import RecognitionContext, build_recognition_context
from ..recognition.factor_recognition import (
    FactorRecognitionResult,
    run_factor_recognition,
)
from ..recognition.image_preprocessing import (
    PreparedFactorImage,
    prepare_factor_image,
)
from ..review import ReviewQueue
from ..schema import Submission
from .result_builder import build_submission


class PrepareImageFn(Protocol):
    def __call__(
        self,
        image_path: str,
        *,
        debug_crops_dir: str | None = None,
        auto_debug: bool = True,
    ) -> PreparedFactorImage: ...


class BuildContextFn(Protocol):
    def __call__(self, *, skip_ocr: bool = False) -> RecognitionContext: ...


class LoadUniqueSkillMapFn(Protocol):
    def __call__(self) -> dict[str, str]: ...


class RecognizeFactorsFn(Protocol):
    def __call__(
        self,
        prepared: PreparedFactorImage,
        context: RecognitionContext,
        unique_skill_to_character: dict[str, str],
    ) -> FactorRecognitionResult: ...


class BuildSubmissionFn(Protocol):
    def __call__(
        self,
        *,
        submitter_id: str,
        image_path: str,
        umas: list,
    ) -> Submission: ...


@dataclass(frozen=True)
class UmaFactorAnalyzer:
    prepare_image: PrepareImageFn = prepare_factor_image
    build_context: BuildContextFn = build_recognition_context
    load_unique_skill_map: LoadUniqueSkillMapFn = load_unique_skill_to_character
    recognize_factors: RecognizeFactorsFn = run_factor_recognition
    make_submission: BuildSubmissionFn = build_submission

    def analyze_image(
        self,
        image_path: str,
        submitter_id: str,
        debug_crops_dir: str | None = None,
        auto_debug: bool = True,
        skip_ocr: bool = False,
    ) -> tuple[Submission, ReviewQueue]:
        prepared = self.prepare_image(
            image_path,
            debug_crops_dir=debug_crops_dir,
            auto_debug=auto_debug,
        )
        context = self.build_context(skip_ocr=skip_ocr)
        recognition = self.recognize_factors(
            prepared,
            context,
            self.load_unique_skill_map(),
        )
        submission = self.make_submission(
            submitter_id=submitter_id,
            image_path=image_path,
            umas=recognition.umas,
        )
        return submission, recognition.review


def analyze_image(
    image_path: str,
    submitter_id: str,
    debug_crops_dir: str | None = None,
    auto_debug: bool = True,
    skip_ocr: bool = False,
) -> tuple[Submission, ReviewQueue]:
    return UmaFactorAnalyzer().analyze_image(
        image_path,
        submitter_id,
        debug_crops_dir=debug_crops_dir,
        auto_debug=auto_debug,
        skip_ocr=skip_ocr,
    )
