"""Runtime dependencies for the recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_generation import FactorOCRLike, FactorPredictorLike
from .characters import CharacterPredictorLike
from .model_registry import get_predictor
from .star_rank import RankPredictorLike
from ..config import green_factor_names
from ..ocr import get_ocr


@dataclass(frozen=True)
class RecognitionContext:
    factor_pred: FactorPredictorLike
    rank_pred: RankPredictorLike
    char_pred: CharacterPredictorLike
    ocr: FactorOCRLike | None
    green_name_set: set[str]


def build_recognition_context(*, skip_ocr: bool = False) -> RecognitionContext:
    factor_pred = get_predictor("factor")
    rank_pred = get_predictor("factor_rank")
    char_pred = get_predictor("character")
    ocr = None if skip_ocr else get_ocr()
    green_name_set = set(green_factor_names() if skip_ocr else ocr._green_factor_names)

    return RecognitionContext(
        factor_pred=factor_pred,
        rank_pred=rank_pred,
        char_pred=char_pred,
        ocr=ocr,
        green_name_set=green_name_set,
    )
