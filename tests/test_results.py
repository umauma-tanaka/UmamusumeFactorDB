from __future__ import annotations

import numpy as np

from umafactor import pipeline, results
from umafactor.results import apply_review_results, build_submission
from umafactor.review import ReviewItem, ReviewQueue
from umafactor.schema import FactorEntry, Submission, UmaFactors


def _review_item(
    *,
    uma_index: int = 0,
    slot: str = "blue",
    white_index: int = 0,
    current_name: str = "current",
    current_star: int = 1,
    reviewed_name: str | None = "reviewed",
    reviewed_star: int | None = None,
) -> ReviewItem:
    roles = ["main", "parent1", "parent2"]
    return ReviewItem(
        uma_index=uma_index,
        uma_role=roles[uma_index],
        slot=slot,
        white_index=white_index,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        candidates=[(current_name, 0.8)],
        current_name=current_name,
        current_star=current_star,
        reviewed_name=reviewed_name,
        reviewed_star=reviewed_star,
    )


def test_build_submission_preserves_uma_order_and_image_basename() -> None:
    umas = [
        UmaFactors(character="main"),
        UmaFactors(character="parent1"),
        UmaFactors(character="parent2"),
    ]

    submission = build_submission(
        submitter_id="tester",
        image_path="captures/factor.png",
        umas=umas,
    )

    assert submission.submitter_id == "tester"
    assert submission.image_filename == "factor.png"
    assert submission.main is umas[0]
    assert submission.parent1 is umas[1]
    assert submission.parent2 is umas[2]


def test_pipeline_exports_apply_review_results_alias() -> None:
    assert pipeline.apply_review_results is results.apply_review_results


def test_apply_review_results_updates_reviewed_slots() -> None:
    submission = Submission(submitter_id="tester", image_filename="factor.png")
    submission.main.skills.append(FactorEntry(color="white", name="old-white", star=1))

    review = ReviewQueue(
        [
            _review_item(
                uma_index=0,
                slot="blue",
                current_star=2,
                reviewed_name="blue-reviewed",
                reviewed_star=None,
            ),
            _review_item(
                uma_index=1,
                slot="red",
                current_star=1,
                reviewed_name="red-reviewed",
                reviewed_star=3,
            ),
            _review_item(
                uma_index=2,
                slot="green",
                current_star=2,
                reviewed_name="green-reviewed",
                reviewed_star=None,
            ),
            _review_item(
                uma_index=0,
                slot="white",
                white_index=0,
                current_star=2,
                reviewed_name="white-reviewed",
                reviewed_star=4,
            ),
        ]
    )

    apply_review_results(submission, review)

    assert submission.main.blue_type == "blue-reviewed"
    assert submission.main.blue_star == 2
    assert submission.parent1.red_type == "red-reviewed"
    assert submission.parent1.red_star == 3
    assert submission.parent2.green_name == "green-reviewed"
    assert submission.parent2.green_star == 2
    assert submission.main.skills[0].name == "white-reviewed"
    assert submission.main.skills[0].star == 4


def test_apply_review_results_skips_missing_name_and_out_of_range_white() -> None:
    submission = Submission(submitter_id="tester", image_filename="factor.png")
    submission.main.blue_type = "kept-blue"
    submission.main.blue_star = 1
    submission.main.skills.append(FactorEntry(color="white", name="kept-white", star=2))

    review = ReviewQueue(
        [
            _review_item(slot="blue", reviewed_name=None, reviewed_star=3),
            _review_item(
                slot="white",
                white_index=9,
                current_star=1,
                reviewed_name="ignored-white",
                reviewed_star=5,
            ),
        ]
    )

    apply_review_results(submission, review)

    assert submission.main.blue_type == "kept-blue"
    assert submission.main.blue_star == 1
    assert submission.main.skills[0].name == "kept-white"
    assert submission.main.skills[0].star == 2
