"""Build and update application-level recognition results."""

from __future__ import annotations

import os
from collections.abc import Sequence

from ..review import ReviewQueue
from ..schema import Submission, UmaFactors


def build_submission(
    *,
    submitter_id: str,
    image_path: str,
    umas: Sequence[UmaFactors],
) -> Submission:
    return Submission(
        submitter_id=submitter_id,
        image_filename=os.path.basename(image_path),
        main=umas[0],
        parent1=umas[1],
        parent2=umas[2],
    )


def apply_review_results(submission: Submission, review: ReviewQueue) -> None:
    """Apply reviewed ReviewItem values back to a Submission."""
    umas = [submission.main, submission.parent1, submission.parent2]
    for item in review.items:
        if item.reviewed_name is None:
            continue
        uma = umas[item.uma_index]
        star = item.reviewed_star if item.reviewed_star is not None else item.current_star
        if item.slot == "blue":
            uma.blue_type = item.reviewed_name
            uma.blue_star = star
        elif item.slot == "red":
            uma.red_type = item.reviewed_name
            uma.red_star = star
        elif item.slot == "green":
            uma.green_name = item.reviewed_name
            uma.green_star = star
        elif item.slot == "white":
            if 0 <= item.white_index < len(uma.skills):
                uma.skills[item.white_index].name = item.reviewed_name
                uma.skills[item.white_index].star = star
