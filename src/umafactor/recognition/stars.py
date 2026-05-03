"""Star classifier inference helpers."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import model_path


STAR_CLASS_NAMES = ["empty", "gold"]
STAR_SLOT_SIZE = 28
STAR_FALLBACK_ENV = "UMAFACTOR_ALLOW_MISSING_STAR_CLASSIFIER"
STAR_GOLD_HSV_LO = (15, 120, 180)
STAR_GOLD_HSV_HI = (40, 255, 255)
STAR_EMPTY_HSV_LO = (0, 10, 200)
STAR_EMPTY_HSV_HI = (45, 90, 255)


def _allow_missing_star_classifier() -> bool:
    return os.environ.get(STAR_FALLBACK_ENV, "").lower() in {"1", "true", "yes", "on"}


def _star_model_path() -> Path:
    return model_path("star_classifier")


def _prep_star_slot(img_bgr: np.ndarray) -> np.ndarray:
    import cv2

    h, w = img_bgr.shape[:2]
    if (h, w) != (STAR_SLOT_SIZE, STAR_SLOT_SIZE):
        img_bgr = cv2.resize(
            img_bgr,
            (STAR_SLOT_SIZE, STAR_SLOT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
    return img_bgr.astype(np.uint8)


def _predict_star_hsv(slot_img_bgr: np.ndarray) -> tuple[str, float]:
    """Explicit opt-in HSV fallback for environments missing star_classifier.onnx."""
    import cv2

    slot = _prep_star_slot(slot_img_bgr)
    hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
    gold_mask = cv2.inRange(
        hsv,
        np.array(STAR_GOLD_HSV_LO, dtype=np.uint8),
        np.array(STAR_GOLD_HSV_HI, dtype=np.uint8),
    )
    empty_mask = cv2.inRange(
        hsv,
        np.array(STAR_EMPTY_HSV_LO, dtype=np.uint8),
        np.array(STAR_EMPTY_HSV_HI, dtype=np.uint8),
    )
    gold_score = float(gold_mask.mean()) / 255.0
    empty_score = float(empty_mask.mean()) / 255.0
    label = "gold" if gold_score >= 0.02 and gold_score >= empty_score * 0.45 else "empty"
    confidence = min(1.0, 0.5 + abs(gold_score - empty_score) * 4.0)
    return label, confidence


@lru_cache(maxsize=1)
def _get_star_session() -> ort.InferenceSession:
    path = _star_model_path()
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def predict_star(slot_img_bgr: np.ndarray) -> tuple[str, float]:
    if not _star_model_path().exists() and _allow_missing_star_classifier():
        return _predict_star_hsv(slot_img_bgr)
    session = _get_star_session()
    batch = _prep_star_slot(slot_img_bgr)[None, ...]
    outs = session.run(["index", "confidence"], {"images": batch})
    idx = int(outs[0][0])
    conf = float(outs[1][0])
    return STAR_CLASS_NAMES[idx], conf


def predict_stars_batch(slot_imgs: list[np.ndarray]) -> list[tuple[str, float]]:
    if not slot_imgs:
        return []
    if not _star_model_path().exists() and _allow_missing_star_classifier():
        return [_predict_star_hsv(img) for img in slot_imgs]
    session = _get_star_session()
    batch = np.stack([_prep_star_slot(img) for img in slot_imgs], axis=0)
    outs = session.run(["index", "confidence"], {"images": batch})
    idxs = outs[0].tolist()
    confs = outs[1].tolist()
    return [(STAR_CLASS_NAMES[int(index)], float(conf)) for index, conf in zip(idxs, confs)]
