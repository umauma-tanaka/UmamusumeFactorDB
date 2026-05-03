from __future__ import annotations

import pytest

from umafactor.recognition.candidate_fusion import (
    finalize_factor_candidates,
    merge_candidates,
    merge_candidates_v2,
)


def test_merge_candidates_keeps_onnx_only_order_and_sources() -> None:
    merged, sources = merge_candidates(
        [("A", 0.8), ("B", 0.5)],
        [],
    )

    assert merged == [("A", 0.8), ("B", 0.5)]
    assert sources == {"A": "onnx", "B": "onnx"}


def test_merge_candidates_weights_and_clamps_ocr_only() -> None:
    merged, sources = merge_candidates(
        [],
        [("A", 0.9)],
    )

    assert merged == [("A", 1.0)]
    assert sources == {"A": "ocr"}


def test_merge_candidates_adds_bonus_when_sources_agree() -> None:
    merged, sources = merge_candidates(
        [("A", 0.6), ("B", 0.7)],
        [("A", 0.6)],
    )

    assert merged[0][0] == "A"
    assert merged[0][1] == pytest.approx(0.9)
    assert sources["A"] == "both"
    assert sources["B"] == "onnx"


def test_merge_candidates_promotes_strong_ocr_top1() -> None:
    merged, sources = merge_candidates(
        [("A", 0.95)],
        [("B", 0.7)],
    )

    assert merged[0][0] == "B"
    assert merged[0][1] == pytest.approx(0.875)
    assert merged[1][0] == "A"
    assert merged[1][1] == pytest.approx(0.95)
    assert sources == {"B": "ocr", "A": "onnx"}


def test_merge_candidates_limit_does_not_trim_sources() -> None:
    merged, sources = merge_candidates(
        [("A", 0.9), ("B", 0.8), ("C", 0.7)],
        [],
        limit=2,
    )

    assert merged == [("A", 0.9), ("B", 0.8)]
    assert sources == {"A": "onnx", "B": "onnx", "C": "onnx"}


def test_finalize_factor_candidates_promotes_strong_template_top1() -> None:
    result = finalize_factor_candidates(
        [("A", 0.95)],
        [],
        [("B", 0.90)],
        green_adoptable=False,
    )

    assert result.candidates == [("B", 0.90), ("A", 0.95)]
    assert result.sources == {"A": "onnx", "B": "template"}
    assert result.top_name == "B"


def test_finalize_factor_candidates_uses_green_template_threshold() -> None:
    result = finalize_factor_candidates(
        [("A", 0.95)],
        [],
        [("B", 0.94)],
        green_adoptable=True,
    )

    assert result.candidates == [("A", 0.95)]
    assert result.sources == {"A": "onnx"}
    assert result.top_name == "A"


def test_finalize_factor_candidates_returns_empty_top_name_without_candidates() -> None:
    result = finalize_factor_candidates([], [], [], green_adoptable=False)

    assert result.candidates == []
    assert result.sources == {}
    assert result.top_name == ""


def test_merge_candidates_v2_combines_three_sources() -> None:
    merged, sources = merge_candidates_v2(
        [("A", 0.8)],
        [("A", 0.7)],
        [("A", 0.9)],
    )

    assert merged == [("A", 1.0)]
    assert sources == {"A": "triple"}


def test_merge_candidates_v2_names_two_source_tags_deterministically() -> None:
    merged, sources = merge_candidates_v2(
        [("A", 0.8)],
        [],
        [("A", 0.8)],
    )

    assert merged[0][0] == "A"
    assert merged[0][1] == pytest.approx(0.95)
    assert sources == {"A": "onnx+template"}
