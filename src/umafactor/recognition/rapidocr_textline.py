"""RapidOCR recognition-only wrapper for single-line factor-name crops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import PROJECT_ROOT


@dataclass(frozen=True)
class RapidOcrTextLineOptions:
    model_root_dir: Path | None = None
    text_score: float | None = None
    ocr_version: str = "PP-OCRv4"
    lang_type: str = "japan"
    model_type: str = "mobile"
    rec_img_shape: tuple[int, int, int] = (3, 48, 480)
    rec_batch_num: int = 12


@dataclass(frozen=True)
class TextLineOcrResult:
    text: str
    score: float | None


class RapidOcrTextLineRecognizer:
    """Run RapidOCR without text detection/classification."""

    def __init__(self, options: RapidOcrTextLineOptions | None = None) -> None:
        _require_rapidocr_dependency()
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        opts = options or RapidOcrTextLineOptions()
        root = (opts.model_root_dir or PROJECT_ROOT / "rapidocr_models").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.text_score = opts.text_score
        self._engine = RapidOCR(
            params={
                "Global.model_root_dir": str(root),
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.use_rec": True,
                "Global.log_level": "error",
                "Det.lang_type": LangDet.MULTI,
                "Rec.lang_type": LangRec(opts.lang_type),
                "Rec.model_type": ModelType(opts.model_type),
                "Rec.ocr_version": OCRVersion(opts.ocr_version),
                "Rec.rec_img_shape": list(opts.rec_img_shape),
                "Rec.rec_batch_num": int(opts.rec_batch_num),
            }
        )

    def recognize(self, img_bgr: np.ndarray) -> TextLineOcrResult:
        output = self._engine(
            img_bgr,
            use_det=False,
            use_cls=False,
            use_rec=True,
            text_score=self.text_score,
        )
        texts = _extract_texts(output)
        scores = _extract_scores(output)
        score = max(scores) if scores else None
        return TextLineOcrResult(text=_normalize_text("".join(texts)), score=score)

    def recognize_many(self, images_bgr: Sequence[np.ndarray]) -> list[TextLineOcrResult]:
        if not images_bgr:
            return []
        try:
            from rapidocr.ch_ppocr_rec import TextRecInput

            output = self._engine.text_rec(TextRecInput(img=[_ensure_bgr_u8(image) for image in images_bgr]))
            texts = _extract_texts(output)
            scores = _extract_scores(output)
            results: list[TextLineOcrResult] = []
            for index in range(len(images_bgr)):
                text = texts[index] if index < len(texts) else ""
                score = scores[index] if index < len(scores) else None
                results.append(TextLineOcrResult(text=_normalize_text(text), score=score))
            return results
        except Exception:
            return [self.recognize(image) for image in images_bgr]


def _require_rapidocr_dependency() -> None:
    if find_spec("rapidocr") is not None:
        return
    raise RuntimeError(
        "RapidOCR requires the 'rapidocr' package, but it is not installed. "
        "Install it in the same Python environment: python -m pip install rapidocr"
    )


def _extract_texts(output: Any) -> list[str]:
    if output is None:
        return []

    txts = getattr(output, "txts", None)
    if txts is not None:
        return [str(text) for text in txts if text is not None]

    if isinstance(output, dict):
        value = output.get("txts") or output.get("rec_texts") or output.get("texts")
        if isinstance(value, (list, tuple)):
            return [str(text) for text in value if text is not None]
        if isinstance(value, str):
            return [value]

    if isinstance(output, (list, tuple)):
        texts: list[str] = []
        for item in output:
            texts.extend(_extract_texts(item))
        return texts

    return []


def _extract_scores(output: Any) -> list[float]:
    if output is None:
        return []

    scores = getattr(output, "scores", None)
    if scores is not None:
        return [float(score) for score in scores if score is not None]

    if isinstance(output, dict):
        value = output.get("scores") or output.get("rec_scores")
        if isinstance(value, (list, tuple)):
            return [float(score) for score in value if score is not None]
        if isinstance(value, (int, float)):
            return [float(value)]

    if isinstance(output, (list, tuple)):
        values: list[float] = []
        for item in output:
            values.extend(_extract_scores(item))
        return values

    return []


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        import cv2

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        import cv2

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)
