from __future__ import annotations

from umafactor import pipeline
from umafactor.recognition import constants


def test_pipeline_reexports_factor_type_constants() -> None:
    assert pipeline.BLUE_FACTOR_TYPES is constants.BLUE_FACTOR_TYPES
    assert pipeline.RED_FACTOR_TYPES is constants.RED_FACTOR_TYPES


def test_pipeline_reexports_perturbation_constants() -> None:
    assert pipeline.PERTURBATIONS_BLUE is constants.PERTURBATIONS_BLUE
    assert pipeline.PERTURBATIONS_RED is constants.PERTURBATIONS_RED
    assert pipeline.PERTURBATIONS_RANK is constants.PERTURBATIONS_RANK
