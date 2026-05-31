"""Candidate scoring for factor-name OCR."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .factor_name_matcher import (
    FactorNameMatcher,
    auto_accept_threshold,
    review_threshold,
)


@dataclass(frozen=True)
class FactorNameOcrInput:
    raw_text: str
    category: str | None
    roi_profile: str
    preprocess_mode: str
    ocr_score: float | None = None
    elapsed_ms: float | None = None


@dataclass(frozen=True)
class FactorNameOcrScore:
    raw_text: str
    normalized_text: str
    canonical_name: str | None
    match_score: float | None
    selected: bool
    needs_review: bool
    fallback_recommended: bool
    roi_profile: str
    preprocess_mode: str
    ocr_score: float | None
    elapsed_ms: float | None


def score_factor_name_ocr(
    item: FactorNameOcrInput,
    *,
    matcher: FactorNameMatcher,
    auto_threshold: float,
    review_threshold_value: float,
) -> FactorNameOcrScore:
    match = matcher.match(item.raw_text, item.category)
    normalized = match.normalized_text
    score = match.match_score
    auto_threshold = auto_accept_threshold(normalized, auto_threshold)
    review_cutoff = review_threshold(normalized, review_threshold_value)
    accepted = score is not None and score >= auto_threshold
    reviewable = score is not None and score >= review_cutoff
    return FactorNameOcrScore(
        raw_text=item.raw_text,
        normalized_text=normalized,
        canonical_name=match.canonical_name,
        match_score=score,
        selected=False,
        needs_review=not accepted,
        fallback_recommended=not reviewable,
        roi_profile=item.roi_profile,
        preprocess_mode=item.preprocess_mode,
        ocr_score=item.ocr_score,
        elapsed_ms=item.elapsed_ms,
    )


def select_best_factor_name_ocr(
    scores: list[FactorNameOcrScore],
) -> tuple[FactorNameOcrScore, list[FactorNameOcrScore]]:
    if not scores:
        raise ValueError("scores is empty")
    best_index = max(range(len(scores)), key=lambda index: _rank(scores[index]))
    selected = [
        replace(score, selected=(index == best_index))
        for index, score in enumerate(scores)
    ]
    return selected[best_index], selected


def _rank(score: FactorNameOcrScore) -> tuple[float, float, int, int]:
    match_score = score.match_score if score.match_score is not None else 0.0
    ocr_score = score.ocr_score if score.ocr_score is not None else 0.0
    return (
        match_score,
        ocr_score,
        int(bool(score.normalized_text)),
        -len(score.raw_text),
    )
