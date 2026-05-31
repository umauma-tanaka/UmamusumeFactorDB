"""RapidOCR adapter for factor-list OCR crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .rapidocr_textline import (
    RapidOcrTextLineOptions,
    RapidOcrTextLineRecognizer,
    _extract_scores,
    _extract_texts,
    _normalize_text,
)


@dataclass(frozen=True)
class RapidOcrOptions:
    model_root_dir: Path | None = None
    text_score: float | None = None
    ocr_version: str = "PP-OCRv4"
    lang_type: str = "japan"
    model_type: str = "mobile"
    rec_img_shape: tuple[int, int, int] = (3, 48, 480)
    rec_batch_num: int = 12


class RapidFactorOCR:
    """Run RapidOCR as a fast text-recognition-only engine.

    This adapter intentionally calls RapidOCR with ``use_det=False`` and
    ``use_rec=True``.  The factor-list pipeline already extracts the name ROI,
    so detection would only add latency and extra failure modes.
    """

    def __init__(
        self,
        *,
        model_root_dir: Path | None = None,
        text_score: float | None = None,
        ocr_version: str = "PP-OCRv4",
        lang_type: str = "japan",
        model_type: str = "mobile",
        rec_img_shape: tuple[int, int, int] = (3, 48, 480),
        rec_batch_num: int = 12,
    ) -> None:
        self._recognizer = RapidOcrTextLineRecognizer(
            RapidOcrTextLineOptions(
                model_root_dir=model_root_dir,
                text_score=text_score,
                ocr_version=ocr_version,
                lang_type=lang_type,
                model_type=model_type,
                rec_img_shape=rec_img_shape,
                rec_batch_num=rec_batch_num,
            )
        )

    def recognize(self, img_bgr: np.ndarray) -> str:
        return self._recognize(img_bgr)

    def recognize_blue(self, img_bgr: np.ndarray) -> str:
        return self._recognize(img_bgr)

    def recognize_red(self, img_bgr: np.ndarray) -> str:
        return self._recognize(img_bgr)

    def recognize_with_parts(self, img_bgr: np.ndarray) -> tuple[str, list[str]]:
        text = self._recognize(img_bgr)
        return text, [text] if text else []

    def recognize_many(self, images_bgr: Sequence[np.ndarray]) -> list[str]:
        return [result.text for result in self.recognize_many_with_scores(images_bgr)]

    def recognize_with_score(self, img_bgr: np.ndarray) -> tuple[str, float | None]:
        result = self._recognizer.recognize(img_bgr)
        return result.text, result.score

    def recognize_many_with_scores(
        self,
        images_bgr: Sequence[np.ndarray],
    ) -> list[tuple[str, float | None]]:
        return [
            (result.text, result.score)
            for result in self._recognizer.recognize_many(images_bgr)
        ]

    def _recognize(self, img_bgr: np.ndarray) -> str:
        return self._recognizer.recognize(img_bgr).text
