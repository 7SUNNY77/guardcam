"""Шаг 1 пайплайна: дешёвый детектор движения (без нейросети).

Находит области, где что-то движется, и отдаёт их рамки. Нейросеть потом
смотрит только на эти кусочки — отсюда и экономия вычислений.

Также умеет нарезать кропы из ваших записей для дообучения:
    python motion.py --source video.mp4 --save-crops data/own_unsorted --show
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common import crop_with_margin


def _merge_boxes(boxes: list[list[int]], gap: int) -> list[list[int]]:
    """Склеивает пересекающиеся или близкие рамки (человек часто рвётся на куски)."""
    boxes = [b[:] for b in boxes]
    merged = True
    while merged:
        merged = False
        out: list[list[int]] = []
        while boxes:
            x, y, w, h = boxes.pop()
            i = 0
            while i < len(boxes):
                bx, by, bw, bh = boxes[i]
                if (x - gap < bx + bw and bx - gap < x + w and
                        y - gap < by + bh and by - gap < y + h):
                    nx, ny = min(x, bx), min(y, by)
                    w, h = max(x + w, bx + bw) - nx, max(y + h, by + bh) - ny
                    x, y = nx, ny
                    boxes.pop(i)
                    merged = True
                else:
                    i += 1
            out.append([x, y, w, h])
        boxes = out
    return boxes


class MotionDetector:
    def __init__(self, proc_width: int = 640, min_area_frac: float = 0.0015,
                 history: int = 500, var_threshold: float = 32, warmup_frames: int = 30):
        self.proc_width = proc_width
        self.min_area_frac = min_area_frac
        self.warmup_frames = warmup_frames
        self.frames_seen = 0
        self.bg = cv2.createBackgroundSubtractorMOG2(history, var_threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __call__(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        H, W = frame.shape[:2]
        scale = self.proc_width / W if W > self.proc_width else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        small = cv2.GaussianBlur(small, (5, 5), 0)

        mask = self.bg.apply(small)
        self.frames_seen += 1
        if self.frames_seen < self.warmup_frames:  # модель фона ещё учится
            return []

        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # тени (127) отбрасываем
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)   # убираем шум/дождь
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * small.shape[0] * small.shape[1]
        boxes = [list(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) >= min_area]
        boxes = _merge_boxes(boxes, gap=int(0.02 * small.shape[1]))

        inv = 1.0 / scale
        return [(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for x, y, w, h in boxes]


def open_source(source: str) -> cv2.VideoCapture:
    """0/1/... — вебкамера, rtsp://... — IP-камера, иначе путь к видеофайлу."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть источник: {source}")
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="0 | rtsp://... | video.mp4")
    ap.add_argument("--save-crops", help="папка для сохранения кропов движения")
    ap.add_argument("--every", type=int, default=5, help="сохранять кропы раз в N кадров")
    ap.add_argument("--min-side", type=int, default=24, help="мин. размер кропа в пикселях")
    ap.add_argument("--show", action="store_true", help="показывать окно с рамками")
    args = ap.parse_args()

    cap = open_source(args.source)
    det = MotionDetector()
    out_dir = Path(args.save_crops) if args.save_crops else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_frame = n_saved = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frame += 1
        boxes = det(frame)

        if out_dir and n_frame % args.every == 0:
            for box in boxes:
                if min(box[2], box[3]) < args.min_side:
                    continue
                cv2.imwrite(str(out_dir / f"{int(time.time())}_{n_frame}_{n_saved}.jpg"),
                            crop_with_margin(frame, box))
                n_saved += 1

        if args.show:
            for x, y, w, h in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow("motion", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    dt = time.perf_counter() - t0
    print(f"кадров: {n_frame}, {n_frame / max(dt, 1e-6):.1f} FPS, сохранено кропов: {n_saved}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
