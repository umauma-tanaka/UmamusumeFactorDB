"""Candidate fusion for factor recognition results."""

from __future__ import annotations

CandidateList = list[tuple[str, float]]
SourceMap = dict[str, str]


def merge_candidates(
    onnx_cands: CandidateList,
    ocr_cands: CandidateList,
    limit: int = 8,
    ocr_weight: float = 1.25,
    ocr_strong_threshold: float = 0.7,
    onnx_weight: float = 1.0,
    both_bonus: float = 0.15,
) -> tuple[CandidateList, SourceMap]:
    """ONNX と OCR の候補を統合スコアでマージする（旧 v1、後方互換用）。"""
    combined: dict[str, tuple[float, str]] = {}
    for name, s in onnx_cands:
        combined[name] = (s * onnx_weight, "onnx")
    for name, s in ocr_cands:
        if name in combined:
            prev_score = combined[name][0]
            new_score = max(prev_score, s * ocr_weight) + both_bonus
            combined[name] = (new_score, "both")
        else:
            combined[name] = (s * ocr_weight, "ocr")

    ordered = sorted(combined.items(), key=lambda kv: -kv[1][0])

    if ocr_cands and ocr_cands[0][1] >= ocr_strong_threshold:
        top_ocr_name = ocr_cands[0][0]
        ordered = [(n, v) for n, v in ordered if n == top_ocr_name] + [
            (n, v) for n, v in ordered if n != top_ocr_name
        ]

    sources = {n: v[1] for n, v in ordered}
    merged = [(n, min(1.0, v[0])) for n, v in ordered][:limit]
    return merged, sources


def merge_candidates_v2(
    onnx_cands: CandidateList,
    ocr_cands: CandidateList,
    template_cands: CandidateList | None = None,
    *,
    limit: int = 8,
    onnx_weight: float = 1.0,
    ocr_weight: float = 1.25,
    template_weight: float = 0.85,
    ocr_strong_threshold: float = 0.7,
    both_bonus: float = 0.15,
    triple_bonus: float = 0.30,
) -> tuple[CandidateList, SourceMap]:
    """ONNX / OCR / Template を重み付き投票でマージする（新 v2）。"""
    template_cands = template_cands or []

    contributions: dict[str, dict[str, float]] = {}
    for name, s in onnx_cands:
        contributions.setdefault(name, {})["onnx"] = s * onnx_weight
    for name, s in ocr_cands:
        contributions.setdefault(name, {})["ocr"] = s * ocr_weight
    for name, s in template_cands:
        contributions.setdefault(name, {})["template"] = s * template_weight

    combined: dict[str, tuple[float, str]] = {}
    for name, srcs in contributions.items():
        n_sources = len(srcs)
        base = max(srcs.values())
        if n_sources >= 3:
            score = base + triple_bonus
            tag = "triple"
        elif n_sources == 2:
            score = base + both_bonus
            tag = "+".join(sorted(srcs.keys()))
        else:
            score = base
            tag = next(iter(srcs.keys()))
        combined[name] = (score, tag)

    ordered = sorted(combined.items(), key=lambda kv: -kv[1][0])

    if ocr_cands and ocr_cands[0][1] >= ocr_strong_threshold:
        top_ocr_name = ocr_cands[0][0]
        ordered = [(n, v) for n, v in ordered if n == top_ocr_name] + [
            (n, v) for n, v in ordered if n != top_ocr_name
        ]

    sources = {n: v[1] for n, v in ordered}
    merged = [(n, min(1.0, v[0])) for n, v in ordered][:limit]
    return merged, sources
