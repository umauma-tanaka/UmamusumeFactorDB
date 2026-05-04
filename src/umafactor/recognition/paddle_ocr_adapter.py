"""PaddleOCR adapter for factor-list OCR crops."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..config import PROJECT_ROOT


PaddleMode = Literal["recognition", "ocr"]


class PaddleFactorOCR:
    """Expose PaddleOCR through the same interface as the factor-list OCR flow.

    The factor-list pipeline passes one detected card or name region at a time.
    Full OCR mode is the default for card crops because PaddleOCR can detect
    the text line inside the card without seeing the full stitched image.
    """

    def __init__(
        self,
        *,
        lang: str = "japan",
        mode: PaddleMode = "recognition",
        cache_dir: Path | None = None,
        text_det_limit_side_len: int | None = None,
        text_det_limit_type: str | None = None,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_det_unclip_ratio: float | None = None,
        text_rec_score_thresh: float | None = None,
    ) -> None:
        self.mode = mode
        _prepare_paddle_cache(cache_dir)

        if mode == "recognition":
            from paddleocr import TextRecognition

            self._engine = TextRecognition(
                model_name="PP-OCRv5_server_rec",
                enable_mkldnn=False,
            )
        elif mode == "ocr":
            from paddleocr import PaddleOCR

            ocr_kwargs = _compact_kwargs(
                {
                    "text_det_limit_side_len": text_det_limit_side_len,
                    "text_det_limit_type": text_det_limit_type,
                    "text_det_thresh": text_det_thresh,
                    "text_det_box_thresh": text_det_box_thresh,
                    "text_det_unclip_ratio": text_det_unclip_ratio,
                    "text_rec_score_thresh": text_rec_score_thresh,
                }
            )
            self._engine = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                **ocr_kwargs,
            )
        else:
            raise ValueError(f"unknown PaddleOCR mode: {mode}")

    def recognize(self, img_bgr: np.ndarray) -> str:
        return self._recognize(img_bgr)

    def recognize_blue(self, img_bgr: np.ndarray) -> str:
        return self._recognize(img_bgr)

    def recognize_red(self, img_bgr: np.ndarray) -> str:
        return self._recognize(img_bgr)

    def recognize_with_parts(self, img_bgr: np.ndarray) -> tuple[str, list[str]]:
        text = self._recognize(img_bgr)
        return text, [text] if text else []

    def _recognize(self, img_bgr: np.ndarray) -> str:
        result = self._engine.predict(img_bgr)
        texts = _extract_texts(result)
        return _normalize_text("".join(texts))


def _prepare_paddle_cache(cache_dir: Path | None) -> None:
    root = cache_dir if cache_dir is not None else PROJECT_ROOT / "paddleocr_cache"
    root = root.resolve()
    # PaddlePaddle 3.3.1 on Windows can fail in the detection pipeline through
    # oneDNN attribute conversion.  Disable it before importing paddleocr.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(root / "paddlex"))
    os.environ.setdefault("PADDLE_HOME", str(root / "paddle"))
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "modelscope"))


def _compact_kwargs(values: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}


def _extract_texts(result: Any) -> list[str]:
    texts: list[str] = []
    _collect_texts(_to_payload(result), texts)
    return texts


def _to_payload(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, str)):
        return value

    json_attr = getattr(value, "json", None)
    if json_attr is not None:
        payload = json_attr() if callable(json_attr) else json_attr
        return payload

    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return to_json()

    return value


def _collect_texts(value: Any, texts: list[str]) -> None:
    value = _to_payload(value)

    if isinstance(value, str):
        if value:
            texts.append(value)
        return

    if isinstance(value, dict):
        for key in ("rec_texts", "texts"):
            item = value.get(key)
            if isinstance(item, list):
                texts.extend(str(text) for text in item if str(text))

        for key in ("rec_text", "text"):
            item = value.get(key)
            if isinstance(item, str) and item:
                texts.append(item)

        for key in ("res", "data", "ocr_res", "results"):
            if key in value:
                _collect_texts(value[key], texts)
        return

    if isinstance(value, (list, tuple)):
        if _looks_like_legacy_ocr_line(value):
            texts.append(str(value[1][0]))
            return
        for item in value:
            _collect_texts(item, texts)


def _looks_like_legacy_ocr_line(value: list[Any] | tuple[Any, ...]) -> bool:
    return (
        len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 1
        and isinstance(value[1][0], str)
    )


def _normalize_text(value: str) -> str:
    text = re.sub(r"\s+", "", value)
    text = text.replace("◯", "○").replace("〇", "○").replace("Ｏ", "○").replace("0", "○")
    text = re.sub(r"^(?:[A-Z]{1,3})?RANK", "", text, flags=re.IGNORECASE)
    return re.sub(r"[★☆⭐]+$", "", text)
