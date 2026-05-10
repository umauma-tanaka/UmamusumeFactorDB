"""PaddleOCR adapter for factor-list OCR crops."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from ..config import PROJECT_ROOT


PaddleMode = Literal["recognition", "ocr"]
DEFAULT_TEXT_DET_LIMIT_SIDE_LEN = 128
DEFAULT_TEXT_DET_LIMIT_TYPE = "min"


@dataclass(frozen=True)
class PaddleOcrTextLine:
    text: str
    bbox: tuple[float, float, float, float] | None = None


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
        _require_paddlepaddle_dependency()

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
                    "text_det_limit_side_len": (
                        text_det_limit_side_len
                        if text_det_limit_side_len is not None
                        else DEFAULT_TEXT_DET_LIMIT_SIDE_LEN
                    ),
                    "text_det_limit_type": text_det_limit_type or DEFAULT_TEXT_DET_LIMIT_TYPE,
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

    def recognize_many(self, images_bgr: Sequence[np.ndarray]) -> list[str]:
        if not images_bgr:
            return []
        try:
            result = self._engine.predict(list(images_bgr))
        except Exception:
            return [self._recognize(image) for image in images_bgr]

        payload_items = _payload_sequence(result)
        if payload_items is not None and len(payload_items) == len(images_bgr):
            return [_normalize_text("".join(_extract_texts(item))) for item in payload_items]
        return [self._recognize(image) for image in images_bgr]

    def recognize_canvas(
        self,
        canvas_bgr: np.ndarray,
        regions: Sequence[tuple[int, int, int, int]],
    ) -> list[str]:
        if not regions:
            return []

        result = self._engine.predict(canvas_bgr)
        lines = _extract_text_lines(result)
        assigned: list[list[str]] = [[] for _region in regions]

        boxed_lines = [line for line in lines if line.bbox is not None]
        if boxed_lines:
            for line in boxed_lines:
                assert line.bbox is not None
                region_index = _region_index_for_bbox(line.bbox, regions)
                if region_index is not None:
                    assigned[region_index].append(line.text)
        else:
            texts = [line.text for line in lines if line.text]
            if len(texts) == len(regions):
                assigned = [[text] for text in texts]

        return [_normalize_text("".join(texts)) for texts in assigned]

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


def _require_paddlepaddle_dependency() -> None:
    if find_spec("paddle") is not None:
        return
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_hint = ""
    if sys.version_info >= (3, 14):
        python_hint = (
            f" Current Python is {python_version}; if pip cannot find a paddlepaddle "
            "wheel, create a Python 3.13 or 3.12 virtual environment for OCR."
        )
    raise RuntimeError(
        "PaddleOCR requires the 'paddlepaddle' package, but it is not installed. "
        "Install it in the same Python environment, for example: "
        "python -m pip install paddlepaddle"
        f"{python_hint}"
    )


def _compact_kwargs(values: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}


def _extract_texts(result: Any) -> list[str]:
    texts: list[str] = []
    _collect_texts(_to_payload(result), texts)
    return texts


def _extract_text_lines(result: Any) -> list[PaddleOcrTextLine]:
    lines: list[PaddleOcrTextLine] = []
    _collect_text_lines(_to_payload(result), lines)
    return lines


def _collect_text_lines(value: Any, lines: list[PaddleOcrTextLine]) -> None:
    value = _to_payload(value)

    if isinstance(value, str):
        if value:
            lines.append(PaddleOcrTextLine(value))
        return

    if isinstance(value, dict):
        texts = _dict_text_list(value)
        boxes = _dict_box_list(value)
        if texts:
            for index, text in enumerate(texts):
                bbox = _bbox_from_any(boxes[index]) if boxes is not None and index < len(boxes) else None
                if text:
                    lines.append(PaddleOcrTextLine(str(text), bbox))

        for key in ("rec_text", "text"):
            item = value.get(key)
            if isinstance(item, str) and item:
                bbox = _bbox_from_any(
                    value.get("bbox")
                    or value.get("box")
                    or value.get("points")
                    or value.get("poly")
                    or value.get("polygon")
                )
                lines.append(PaddleOcrTextLine(item, bbox))

        for key in ("res", "data", "ocr_res", "results"):
            if key in value:
                _collect_text_lines(value[key], lines)
        return

    if isinstance(value, (list, tuple)):
        if _looks_like_legacy_ocr_line(value):
            lines.append(PaddleOcrTextLine(str(value[1][0]), _bbox_from_any(value[0])))
            return
        for item in value:
            _collect_text_lines(item, lines)


def _dict_text_list(value: dict[str, Any]) -> list[str]:
    for key in ("rec_texts", "texts"):
        item = value.get(key)
        if isinstance(item, list):
            return [str(text) for text in item if str(text)]
    return []


def _dict_box_list(value: dict[str, Any]) -> Sequence[Any] | None:
    for key in ("rec_boxes", "rec_polys", "boxes", "dt_boxes", "dt_polys", "polys"):
        item = value.get(key)
        if isinstance(item, (list, tuple, np.ndarray)):
            return item
    return None


def _bbox_from_any(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size < 4:
        return None
    if arr.ndim == 1 and arr.size >= 4:
        x0, y0, x1, y1 = arr[:4]
        return float(x0), float(y0), float(x1), float(y1)
    arr = arr.reshape(-1, 2)
    xs = arr[:, 0]
    ys = arr[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _region_index_for_bbox(
    bbox: tuple[float, float, float, float],
    regions: Sequence[tuple[int, int, int, int]],
) -> int | None:
    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    for index, (rx0, ry0, rx1, ry1) in enumerate(regions):
        if rx0 <= center_x <= rx1 and ry0 <= center_y <= ry1:
            return index

    intersections = [
        _intersection_area(bbox, (float(rx0), float(ry0), float(rx1), float(ry1)))
        for rx0, ry0, rx1, ry1 in regions
    ]
    if not intersections or max(intersections) <= 0:
        return None
    return int(np.argmax(intersections))


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    return width * height


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


def _payload_sequence(value: Any) -> list[Any] | None:
    payload = _to_payload(value)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return list(payload)
    if isinstance(payload, (dict, str)):
        return None
    try:
        return list(payload)
    except TypeError:
        return None


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
