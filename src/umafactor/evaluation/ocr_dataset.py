"""Ground-truth loading and metrics for stitched factor OCR experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ExpectedOcrFactor:
    order: int
    name: str
    star: int


class OcrDetectedFactorLike(Protocol):
    order: int
    raw_name: str
    star: int


_SIMILARITY_BUCKET_LABELS = (
    "100%",
    "95-99%",
    "90-94%",
    "80-89%",
    "60-79%",
    "1-59%",
    "0%",
    "blank",
)


def load_expected_ocr_factors(path: Path) -> list[ExpectedOcrFactor]:
    expected: list[ExpectedOcrFactor] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != 2:
                raise ValueError(f"expected 2 columns at row {len(expected) + 1}: {row}")
            name = row[0].strip()
            try:
                star = int(row[1])
            except ValueError as exc:
                raise ValueError(f"invalid star value at row {len(expected) + 1}: {row}") from exc
            expected.append(ExpectedOcrFactor(order=len(expected), name=name, star=star))
    return expected


def evaluate_ocr_factors(
    expected: Sequence[ExpectedOcrFactor],
    detected: Sequence[OcrDetectedFactorLike],
    *,
    evaluate_names: bool = False,
) -> dict[str, Any]:
    pair_count = min(len(expected), len(detected))
    star_correct = 0
    name_correct = 0
    name_evaluated = 0
    blank_name_count = 0
    name_similarity_sum = 0.0
    name_similarity_min: float | None = None
    name_similarity_buckets = {label: 0 for label in _SIMILARITY_BUCKET_LABELS}
    failures: list[dict[str, Any]] = []

    for index in range(pair_count):
        exp = expected[index]
        det = detected[index]
        detected_name = (det.raw_name or "").strip()
        if evaluate_names or detected_name:
            name_evaluated += 1
            similarity = _name_similarity(detected_name, exp.name)
            name_similarity_sum += similarity
            name_similarity_min = (
                similarity
                if name_similarity_min is None
                else min(name_similarity_min, similarity)
            )
            name_similarity_buckets[_name_similarity_bucket(detected_name, similarity)] += 1
            if not detected_name:
                blank_name_count += 1
                failures.append(
                    {
                        "kind": "name",
                        "order": index,
                        "expected": exp.name,
                        "actual": "",
                        "similarity": similarity,
                    }
                )
            elif detected_name == exp.name:
                name_correct += 1
            else:
                failures.append(
                    {
                        "kind": "name",
                        "order": index,
                        "expected": exp.name,
                        "actual": detected_name,
                        "similarity": similarity,
                    }
                )
        if det.star == exp.star:
            star_correct += 1
        else:
            failures.append(
                {
                    "kind": "star",
                    "order": index,
                    "expected": exp.star,
                    "actual": det.star,
                }
            )

    missing_count = max(0, len(expected) - len(detected))
    extra_count = max(0, len(detected) - len(expected))
    if missing_count:
        failures.append(
            {
                "kind": "missing",
                "order": pair_count,
                "expected": missing_count,
                "actual": 0,
            }
        )
    if extra_count:
        failures.append(
            {
                "kind": "extra",
                "order": len(expected),
                "expected": 0,
                "actual": extra_count,
            }
        )

    return {
        "expected_count": len(expected),
        "detected_count": len(detected),
        "count_delta": len(detected) - len(expected),
        "pair_count": pair_count,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "star_correct": star_correct,
        "star_accuracy": star_correct / pair_count if pair_count else 0.0,
        "name_evaluated_count": name_evaluated,
        "blank_name_count": blank_name_count,
        "name_correct": name_correct,
        "name_accuracy": name_correct / name_evaluated if name_evaluated else None,
        "name_similarity_mean": name_similarity_sum / name_evaluated
        if name_evaluated
        else None,
        "name_similarity_min": name_similarity_min,
        "name_similarity_buckets": name_similarity_buckets,
        "name_similarity_bucket_percentages": {
            label: count / name_evaluated if name_evaluated else None
            for label, count in name_similarity_buckets.items()
        },
        "failures": failures,
    }


def _name_similarity(actual: str, expected: str) -> float:
    if not actual and not expected:
        return 1.0
    return SequenceMatcher(None, actual, expected).ratio()


def _name_similarity_bucket(actual: str, similarity: float) -> str:
    if not actual:
        return "blank"
    if similarity >= 1.0:
        return "100%"
    if similarity >= 0.95:
        return "95-99%"
    if similarity >= 0.90:
        return "90-94%"
    if similarity >= 0.80:
        return "80-89%"
    if similarity >= 0.60:
        return "60-79%"
    if similarity > 0.0:
        return "1-59%"
    return "0%"
