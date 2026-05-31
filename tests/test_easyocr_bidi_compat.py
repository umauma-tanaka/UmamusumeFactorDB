from __future__ import annotations

import bidi

from umafactor.ocr import _ensure_easyocr_bidi_compat


def test_easyocr_bidi_compat_exposes_get_display(monkeypatch) -> None:
    monkeypatch.delattr(bidi, "get_display", raising=False)

    _ensure_easyocr_bidi_compat()

    assert callable(bidi.get_display)
