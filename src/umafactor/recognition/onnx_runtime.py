"""ONNX Runtime predictor wrapper and model I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from ..config import MODEL_INPUT_SIZES, load_labels, model_path


@dataclass
class Prediction:
    index: int
    label: str
    confidence: float


@dataclass(frozen=True)
class OnnxModelIO:
    input_name: str
    input_shape: tuple[Any, ...]
    output_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "output_names": list(self.output_names),
        }


FACTOR_SOFTMAX_NAME = "onnx::ReduceMax_639"
FACTOR_WITH_PROBS_FILENAME = "prediction_with_probs.onnx"


def _ensure_factor_with_probs(src_path: Path) -> Path:
    """Create the derived factor model with softmax output when absent."""
    derived = src_path.parent / FACTOR_WITH_PROBS_FILENAME
    if derived.exists():
        return derived
    model = onnx.load(str(src_path))
    probs_vi = onnx.helper.make_tensor_value_info(
        FACTOR_SOFTMAX_NAME,
        onnx.TensorProto.FLOAT,
        ["batch", 820],
    )
    model.graph.output.extend([probs_vi])
    onnx.save(model, str(derived))
    return derived


def describe_session_io(session: ort.InferenceSession) -> OnnxModelIO:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_meta = inputs[0]
    return OnnxModelIO(
        input_name=input_meta.name,
        input_shape=tuple(input_meta.shape),
        output_names=tuple(output.name for output in outputs),
    )


class OnnxPredictor:
    def __init__(
        self,
        model_name: str,
        label_key: str,
        index_output: str = "index",
        confidence_output: str = "confidence",
        extra_outputs: tuple[str, ...] = (),
    ) -> None:
        self.model_name = model_name
        path = model_path(model_name)
        if model_name == "factor":
            path = _ensure_factor_with_probs(path)
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.model_io = describe_session_io(self.session)
        self.input_name = self.model_io.input_name
        self.index_output = index_output
        self.confidence_output = confidence_output
        self.extra_outputs = extra_outputs
        self.labels = load_labels()[label_key]
        self.expected_hw = MODEL_INPUT_SIZES[model_name]

    def describe_io(self) -> OnnxModelIO:
        return self.model_io

    def _preprocess(self, img_hwc_bgr: np.ndarray) -> np.ndarray:
        import cv2

        eh, ew = self.expected_hw
        if img_hwc_bgr.shape[:2] != (eh, ew):
            img_hwc_bgr = cv2.resize(img_hwc_bgr, (ew, eh), interpolation=cv2.INTER_LINEAR)
        return img_hwc_bgr.astype(np.uint8)[None, ...]

    def predict(self, img_hwc_bgr: np.ndarray) -> Prediction:
        batch = self._preprocess(img_hwc_bgr)
        outputs = self.session.run(
            [self.index_output, self.confidence_output],
            {self.input_name: batch},
        )
        idx = int(outputs[0][0])
        conf = float(outputs[1][0])
        label = self.labels[idx] if 0 <= idx < len(self.labels) else f"<out_of_range:{idx}>"
        return Prediction(index=idx, label=label, confidence=conf)

    def predict_with_perturbation(
        self,
        img_hwc_bgr: np.ndarray,
        perturbations: list[tuple[int, int]],
    ) -> Prediction:
        import cv2

        if not perturbations:
            return self.predict(img_hwc_bgr)

        eh, ew = self.expected_hw
        h, w = img_hwc_bgr.shape[:2]
        if (h, w) != (eh, ew):
            base = cv2.resize(img_hwc_bgr, (ew, eh), interpolation=cv2.INTER_LINEAR)
        else:
            base = img_hwc_bgr

        label_score: dict[int, float] = {}
        label_count: dict[int, int] = {}
        for dy, dx in perturbations:
            if dy == 0 and dx == 0:
                shifted = base
            else:
                affine = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(
                    base,
                    affine,
                    (ew, eh),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            batch = shifted.astype(np.uint8)[None, ...]
            outs = self.session.run(
                [self.index_output, self.confidence_output],
                {self.input_name: batch},
            )
            idx = int(outs[0][0])
            conf = float(outs[1][0])
            label_score[idx] = label_score.get(idx, 0.0) + conf
            label_count[idx] = label_count.get(idx, 0) + 1

        best_idx = max(label_score.keys(), key=lambda key: label_score[key])
        avg_conf = label_score[best_idx] / label_count[best_idx]
        label = (
            self.labels[best_idx] if 0 <= best_idx < len(self.labels) else f"<oor:{best_idx}>"
        )
        return Prediction(index=best_idx, label=label, confidence=avg_conf)

    def predict_probs(self, img_hwc_bgr: np.ndarray) -> np.ndarray:
        if not self.extra_outputs:
            raise RuntimeError(f"{self.model_name} は probs 出力を持ちません")
        batch = self._preprocess(img_hwc_bgr)
        outputs = self.session.run(list(self.extra_outputs), {self.input_name: batch})
        return outputs[0][0]

    def predict_in_category(
        self,
        img_hwc_bgr: np.ndarray,
        allowed_labels: list[str],
    ) -> Prediction:
        probs = self.predict_probs(img_hwc_bgr)
        allowed_idxs = [self.labels.index(label) for label in allowed_labels]
        sub_probs = probs[allowed_idxs]
        best_in = int(np.argmax(sub_probs))
        global_idx = allowed_idxs[best_in]
        return Prediction(
            index=global_idx,
            label=allowed_labels[best_in],
            confidence=float(sub_probs[best_in]),
        )

    def predict_in_category_best_of(
        self,
        img_list: list[np.ndarray],
        allowed_labels: list[str],
    ) -> Prediction:
        if not img_list:
            raise ValueError("img_list is empty")
        allowed_idxs = [self.labels.index(label) for label in allowed_labels]
        best: Prediction | None = None
        for img in img_list:
            probs = self.predict_probs(img)
            sub = probs[allowed_idxs]
            best_in = int(np.argmax(sub))
            conf = float(sub[best_in])
            if best is None or conf > best.confidence:
                best = Prediction(
                    index=allowed_idxs[best_in],
                    label=allowed_labels[best_in],
                    confidence=conf,
                )
        assert best is not None
        return best

    def predict_in_category_multi_interp(
        self,
        img_list: list[np.ndarray],
        allowed_labels: list[str],
        interps: tuple[int, ...] = (1, 3),
    ) -> Prediction:
        import cv2

        if not img_list:
            raise ValueError("img_list is empty")
        allowed_idxs = [self.labels.index(label) for label in allowed_labels]
        eh, ew = self.expected_hw
        best: Prediction | None = None
        for img in img_list:
            for interp in interps:
                resized = cv2.resize(img, (ew, eh), interpolation=interp)
                batch = resized.astype(np.uint8)[None, ...]
                outs = self.session.run(list(self.extra_outputs), {self.input_name: batch})
                probs = outs[0][0]
                sub = probs[allowed_idxs]
                best_in = int(np.argmax(sub))
                conf = float(sub[best_in])
                if best is None or conf > best.confidence:
                    best = Prediction(
                        index=allowed_idxs[best_in],
                        label=allowed_labels[best_in],
                        confidence=conf,
                    )
        assert best is not None
        return best

    def predict_ensemble(self, img_list: list[np.ndarray]) -> Prediction:
        if not img_list:
            raise ValueError("img_list is empty")
        probs_sum = np.zeros_like(self.predict_probs(img_list[0]))
        for img in img_list:
            probs_sum += self.predict_probs(img)
        probs_avg = probs_sum / len(img_list)
        best = int(np.argmax(probs_avg))
        return Prediction(
            index=best,
            label=self.labels[best] if 0 <= best < len(self.labels) else f"<oor:{best}>",
            confidence=float(probs_avg[best]),
        )

    def topk_ensemble(
        self,
        img_list: list[np.ndarray],
        k: int = 8,
    ) -> list[tuple[str, float]]:
        if not img_list:
            raise ValueError("img_list is empty")
        probs_sum = np.zeros_like(self.predict_probs(img_list[0]))
        for img in img_list:
            probs_sum += self.predict_probs(img)
        probs_avg = probs_sum / len(img_list)
        top_idx = np.argsort(-probs_avg)[:k]
        return [
            (self.labels[index], float(probs_avg[index]))
            for index in top_idx
            if 0 <= index < len(self.labels)
        ]

    def topk_in_category(
        self,
        img_list: list[np.ndarray],
        allowed_labels: list[str],
        k: int = 8,
        use_multi_interp: bool = False,
        interps: tuple[int, ...] = (1, 3),
    ) -> list[tuple[str, float]]:
        import cv2

        if not img_list:
            raise ValueError("img_list is empty")
        allowed_idxs = [self.labels.index(label) for label in allowed_labels]
        if not use_multi_interp:
            probs_sum = np.zeros_like(self.predict_probs(img_list[0]))
            for img in img_list:
                probs_sum += self.predict_probs(img)
            probs_avg = probs_sum / len(img_list)
            sub = probs_avg[allowed_idxs]
        else:
            eh, ew = self.expected_hw
            sub = np.zeros(len(allowed_labels))
            for img in img_list:
                for interp in interps:
                    resized = cv2.resize(img, (ew, eh), interpolation=interp)
                    batch = resized.astype(np.uint8)[None, ...]
                    outs = self.session.run(list(self.extra_outputs), {self.input_name: batch})
                    probs = outs[0][0]
                    sub_i = probs[allowed_idxs]
                    sub = np.maximum(sub, sub_i)
        top_idx = np.argsort(-sub)[:k]
        return [(allowed_labels[index], float(sub[index])) for index in top_idx]
