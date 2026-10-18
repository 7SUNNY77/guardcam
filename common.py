"""Общие константы и препроцессинг — одинаковые для обучения и инференса."""
from __future__ import annotations

import numpy as np

# Порядок алфавитный — совпадает с torchvision.datasets.ImageFolder.
CLASSES = ["animal", "background", "person"]
IMG_SIZE = 128
PAD_VALUE = 114
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Вписывает картинку в квадрат size×size без искажения пропорций."""
    import cv2

    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def preprocess(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """Список BGR-кропов -> батч float32 формы (N, 3, S, S)."""
    import cv2

    batch = []
    for crop in crops_bgr:
        img = letterbox(crop)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
    return np.stack(batch).astype(np.float32)


def crop_with_margin(frame: np.ndarray, box, margin: float = 0.15) -> np.ndarray:
    """Вырезает box=(x, y, w, h) с запасом по краям, чтобы объект попадал целиком."""
    x, y, w, h = box
    H, W = frame.shape[:2]
    dx, dy = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
    return frame[y0:y1, x0:x1]


class Classifier:
    """Обёртка над ONNX-моделью: кропы -> вероятности классов."""

    def __init__(self, model_path: str, threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        logits = self.session.run(None, {self.input_name: preprocess(crops_bgr)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
