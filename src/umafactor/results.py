"""Compatibility facade for recognition result builders."""

from __future__ import annotations

from .app.result_builder import apply_review_results, build_submission

__all__ = ["apply_review_results", "build_submission"]
