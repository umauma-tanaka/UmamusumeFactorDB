from __future__ import annotations

from umafactor.recognition.rapid_ocr_adapter import _extract_texts, _normalize_text


class _FakeRapidOutput:
    txts = (" スピード ", None, "◎")


def test_extract_texts_from_rapidocr_output() -> None:
    assert _extract_texts(_FakeRapidOutput()) == [" スピード ", "◎"]


def test_normalize_text_removes_spacing_only() -> None:
    assert _normalize_text(" ス ピード ◎ ") == "スピード◎"
