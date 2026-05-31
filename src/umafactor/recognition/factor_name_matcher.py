"""Factor-name normalization and fuzzy matching."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from rapidfuzz import fuzz

from ..config import green_factor_names, load_labels, load_skill_master_names
from .constants import BLUE_FACTOR_TYPES, RED_FACTOR_TYPES


_OCR_VARIANT_MAP = str.maketrans(
    {
        "\u8d4f": "\u8cde",  # 赏 -> 賞
        "\u6b65": "\u6b69",  # 步 -> 歩
        "\u6a31": "\u685c",  # 櫻 -> 桜
        "\uff65": "\u30fb",  # ･ -> ・
        "\u00b7": "\u30fb",  # · -> ・
    }
)


@dataclass(frozen=True)
class FactorNameMatch:
    raw_text: str
    normalized_text: str
    canonical_name: str | None
    match_score: float | None


class FactorNameMatcher:
    """Match normalized OCR text to the factor-name master."""

    def match(self, raw_text: str, category: str | None) -> FactorNameMatch:
        normalized = normalize_factor_name(raw_text)
        candidate, score = match_factor_name(normalized, category)
        return FactorNameMatch(
            raw_text=raw_text,
            normalized_text=normalized,
            canonical_name=candidate,
            match_score=score,
        )


def normalize_factor_name(raw_name: str) -> str:
    text = unicodedata.normalize("NFKC", raw_name or "")
    text = re.sub(r"\s+", "", text)
    text = text.translate(_OCR_VARIANT_MAP)
    text = text.replace("エリサベス", "エリザベス")
    text = text.replace("チャンピオンス", "チャンピオンズ")
    text = text.replace("熱華", "烈華")
    text = text.replace("ヴイクトリア", "ヴィクトリア")
    text = text.replace("\u30a6\u30a3\u30af\u30c8\u30a6\u30de\u30a4\u30eb", "\u30f4\u30a3\u30af\u30c8\u30ea\u30a2\u30de\u30a4\u30eb")
    text = text.replace("\u30f4\u30a3\u30af\u30c8\u30a6\u30de\u30a4\u30eb", "\u30f4\u30a3\u30af\u30c8\u30ea\u30a2\u30de\u30a4\u30eb")
    text = text.replace("\u30a6\u30a3\u30af\u30c8\u30ea\u30a2", "\u30f4\u30a3\u30af\u30c8\u30ea\u30a2")
    text = text.replace("\u56de\u30ea", "\u56de\u308a")
    text = text.replace("\u4e0a\u304c\u30ea", "\u4e0a\u304c\u308a")
    text = re.sub(r"[\u2605\u2606\u2b50]", "", text)
    text = re.sub(r"^(?:[A-Z]{1,3})?RANK", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[0Oo\uff2f\u3007\u25ef\u25cb][\u30fc\u30fb\u3001\u3002\u3040-\u30ffA-Za-z0-9]{0,4}$", "\u25cb", text)
    text = re.sub(r"(?<=[A-Za-z])[\u30fc\u30fb\u3001\u3002\u3040-\u30ff]{1,4}$", "", text)
    return text.strip()


def match_factor_name(
    normalized_name: str,
    category: str | None,
) -> tuple[str | None, float | None]:
    if not normalized_name:
        return None, None

    best_name: str | None = None
    best_normalized = ""
    best_score = 0.0
    for candidate, candidate_normalized in _factor_name_candidates(category):
        if not candidate_normalized:
            continue
        if not _is_fuzzy_match_plausible(normalized_name, candidate_normalized):
            continue
        score = _factor_name_similarity(normalized_name, candidate_normalized)
        if score > best_score or (
            abs(score - best_score) < 1e-9 and len(candidate_normalized) > len(best_normalized)
        ):
            best_name = candidate
            best_normalized = candidate_normalized
            best_score = float(score)

    if best_name is None:
        return None, None
    return best_name, best_score


def auto_accept_threshold(normalized_name: str, base_threshold: float) -> float:
    """Raise the threshold for short names because they are easy to over-match."""

    length = len(normalized_name)
    if length <= 3:
        return max(base_threshold, 0.97)
    if length <= 5:
        return max(base_threshold, 0.94)
    return base_threshold


def review_threshold(normalized_name: str, base_threshold: float) -> float:
    length = len(normalized_name)
    if length <= 3:
        return max(base_threshold, 0.90)
    return base_threshold


def clear_factor_name_matcher_caches() -> None:
    _factor_name_candidates.cache_clear()
    _all_factor_names.cache_clear()
    _skill_factor_names.cache_clear()
    _skill_master_names.cache_clear()


def _factor_name_similarity(normalized_name: str, candidate_normalized: str) -> float:
    score = max(
        fuzz.WRatio(normalized_name, candidate_normalized),
        fuzz.ratio(normalized_name, candidate_normalized),
        fuzz.partial_ratio(normalized_name, candidate_normalized) * 0.95,
        fuzz.ratio(
            _devoice_japanese(normalized_name),
            _devoice_japanese(candidate_normalized),
        )
        * 0.95,
    ) / 100.0
    length_ratio = min(len(candidate_normalized), len(normalized_name)) / max(
        1,
        max(len(candidate_normalized), len(normalized_name)),
    )
    if len(candidate_normalized) < len(normalized_name) or length_ratio < 0.80:
        score *= length_ratio
    score = max(score, _prefix_similarity_boost(normalized_name, candidate_normalized))
    return float(score)


def _prefix_similarity_boost(normalized_name: str, candidate_normalized: str) -> float:
    if not normalized_name or not candidate_normalized:
        return 0.0
    if normalized_name.startswith(candidate_normalized):
        extra = len(normalized_name) - len(candidate_normalized)
        if extra == 0:
            return 1.0
        if extra <= max(3, int(round(len(candidate_normalized) * 0.45))):
            return max(0.0, 0.985 - extra * 0.015)
        if extra <= max(6, len(candidate_normalized)) and _looks_like_trailing_ocr_noise(
            normalized_name[len(candidate_normalized) :]
        ):
            return 0.945 if len(candidate_normalized) <= 5 else 0.965
    if candidate_normalized.startswith(normalized_name):
        missing = len(candidate_normalized) - len(normalized_name)
        if missing <= max(2, int(round(len(candidate_normalized) * 0.30))):
            return max(0.0, 0.94 - missing * 0.02)
    return 0.0


def _looks_like_trailing_ocr_noise(text: str) -> bool:
    if not text:
        return False
    if re.search(r"[\u3400-\u9fff]", text):
        return False
    noise_chars = set("ー・･·。、,.・!！?？()（）[]【】「」『』-~〜")
    return all(
        char in noise_chars
        or "ぁ" <= char <= "ん"
        or "ァ" <= char <= "ン"
        or char.isascii()
        for char in text
    )


def _is_fuzzy_match_plausible(normalized_name: str, candidate_normalized: str) -> bool:
    query = _searchable_text(normalized_name)
    candidate = _searchable_text(candidate_normalized)
    if not query or not candidate:
        return False
    if len(query) <= 1:
        return False
    if len(candidate) >= 7 and len(query) / len(candidate) < 0.60:
        return False
    return True


def _searchable_text(text: str) -> str:
    return "".join(
        char
        for char in text
        if char.isalnum()
        or "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u9fff"
        or char in {"\u25cb", "\u25ce", "\u30fb", "!", "?", "\u3001"}
    )


def _devoice_japanese(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if ch not in {"\u3099", "\u309a"})
    return unicodedata.normalize("NFC", stripped)


@lru_cache(maxsize=8)
def _factor_name_candidates(category: str | None) -> tuple[tuple[str, str], ...]:
    names = _factor_names_for_category(category)
    return tuple((name, normalize_factor_name(name)) for name in names)


def _factor_names_for_category(category: str | None) -> tuple[str, ...]:
    if category == "blue":
        return tuple(BLUE_FACTOR_TYPES)
    if category == "red":
        return tuple(RED_FACTOR_TYPES)
    if category == "green":
        return tuple(green_factor_names())
    if category == "white":
        return _skill_factor_names()
    return _all_factor_names()


@lru_cache(maxsize=1)
def _all_factor_names() -> tuple[str, ...]:
    labels = load_labels()
    names = labels.get("factor.name", [])
    return _unique(
        [str(name) for name in names if str(name).strip()]
        + list(_skill_master_names())
    )


@lru_cache(maxsize=1)
def _skill_factor_names() -> tuple[str, ...]:
    excluded = set(BLUE_FACTOR_TYPES) | set(RED_FACTOR_TYPES) | set(green_factor_names())
    return _unique(
        [name for name in _all_factor_names() if name not in excluded]
        + [name for name in _skill_master_names() if name not in excluded]
    )


@lru_cache(maxsize=1)
def _skill_master_names() -> tuple[str, ...]:
    return tuple(load_skill_master_names())


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value.strip()))
