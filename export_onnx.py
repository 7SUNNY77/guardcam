"""Экспорт в ONNX + квантование в int8 + сравнение размера/скорости/точности.

    python export_onnx.py --ckpt models/best.pt --data data

Результат: models/classifier.onnx (fp32) и models/classifier_int8.onnx.
Int8-версия — то, что потом поедет на слабое железо (Raspberry Pi, Rockchip NPU).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import CLASSES, IMG_SIZE, Classifier, preprocess
from train import build_model


def list_images(root: Path) -> list[tuple[Path, int]]:
    return [(p, i) for i, c in enumerate(CLASSES) for p in sorted((root / c).glob("*.jpg"))]


def evaluate_onnx(path: Path, items, batch: int = 64) -> dict:
    clf = Classifier(str(path), threads=1)
    correct = np.zeros(len(CLASSES))
    total = np.zeros(len(CLASSES))
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        probs = clf([cv2.imread(str(p)) for p, _ in chunk])
        for (_, y), pr in zip(chunk, probs):
            total[y] += 1
            correct[y] += pr.argmax() == y
    crop = cv2.imread(str(items[0][0]))
    clf([crop])  # прогрев
    t0 = time.perf_counter()
    for _ in range(200):
        clf([crop])
    return {
        "size_mb": path.stat().st_size / 2**20,
        "ms_per_crop_1thread": (time.perf_counter() - t0) / 200 * 1000,
        "balanced_acc": float((correct / total.clip(min=1)).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--calib", type=int, default=300, help="картинок для калибровки int8")
    args = ap.parse_args()

    out_dir = Path(args.ckpt).parent
    fp32, int8 = out_dir / "classifier.onnx", out_dir / "classifier_int8.onnx"

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), str(fp32),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print("fp32 ->", fp32)

    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    calib_items = list_images(Path(args.data) / "train")
    random.Random(0).shuffle(calib_items)
    calib_items = calib_items[:args.calib]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_items)

        def get_next(self):
            item = next(self.it, None)
            return None if item is None else {"input": preprocess([cv2.imread(str(item[0]))])}

    prep = out_dir / "classifier_prep.onnx"
    quant_pre_process(str(fp32), str(prep))
    # Веса int8, активации 16 бит (A16W8). Полный int8 (A8W8) у MobileNetV3 после
    # обычной пост-квантизации проседает с ~0.90 до ~0.45–0.63 balanced acc:
    # у активаций огромные выбросы (ReLU ранних блоков и SE-блоков, SQNR 2–5 дБ),
    # 8 бит на них не хватает. Для полного int8 нужно quantization-aware training.
    quantize_static(str(prep), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt16, weight_type=QuantType.QInt8, per_channel=True,
                    extra_options={"UseQDQContribOps": True})
    prep.unlink(missing_ok=True)
    print("int8 ->", int8)

    val_items = list_images(Path(args.data) / "val")
    for name, path in (("fp32", fp32), ("int8", int8)):
        r = evaluate_onnx(path, val_items)
        print(f"{name}: {r['size_mb']:.2f} МБ | {r['ms_per_crop_1thread']:.2f} мс/кроп (1 поток) | "
              f"balanced acc {r['balanced_acc']:.3f}")


if __name__ == "__main__":
    main()
