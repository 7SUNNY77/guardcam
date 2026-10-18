"""Боевой режим: камера -> движение -> классификатор -> уведомление.

    python watch.py --source 0 --show
    python watch.py --source "rtsp://user:pass@192.168.1.64:554/stream1"

Уведомления в Telegram включаются переменными окружения
TELEGRAM_TOKEN и TELEGRAM_CHAT_ID (бот создаётся через @BotFather).
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

import cv2

from common import CLASSES, Classifier, crop_with_margin
from motion import MotionDetector, open_source

COLORS = {"person": (0, 0, 255), "animal": (0, 200, 0), "background": (160, 160, 160)}


def send_telegram(frame, caption: str):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    import requests

    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                      data={"chat_id": chat, "caption": caption},
                      files={"photo": ("alert.jpg", jpg.tobytes())}, timeout=15)
    except Exception as e:
        print("Telegram недоступен:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="models/classifier.onnx",
                    help="на ПК fp32 быстрее; classifier_int8.onnx — в 3 раза меньше, для слабых устройств")
    ap.add_argument("--alert-on", nargs="+", default=["person"], choices=CLASSES)
    ap.add_argument("--threshold", type=float, default=0.6, help="мин. вероятность класса")
    ap.add_argument("--confirm", type=int, default=3,
                    help="сколько срабатываний подряд (из окна 5 кадров) нужно для тревоги")
    ap.add_argument("--cooldown", type=float, default=60, help="секунд между тревогами")
    ap.add_argument("--min-side", type=int, default=20)
    ap.add_argument("--snapshots", default="alerts")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cap = open_source(args.source)
    motion = MotionDetector()
    clf = Classifier(args.model)
    Path(args.snapshots).mkdir(parents=True, exist_ok=True)

    recent = {c: deque(maxlen=5) for c in args.alert_on}
    last_alert = 0.0
    n_frames = n_crops = 0
    t_nn = 0.0
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        boxes = [b for b in motion(frame) if min(b[2], b[3]) >= args.min_side]

        t0 = time.perf_counter()
        probs = clf([crop_with_margin(frame, b) for b in boxes])
        t_nn += time.perf_counter() - t0
        n_crops += len(boxes)

        found = {c: False for c in args.alert_on}
        for box, p in zip(boxes, probs):
            label = CLASSES[int(p.argmax())]
            conf = float(p.max())
            if label in found and conf >= args.threshold:
                found[label] = True
            if args.show or label in found:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[label], 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[label], 2)

        # Временное подтверждение: одиночный «человек» на одном кадре — скорее шум.
        for c in args.alert_on:
            recent[c].append(found[c])
            if sum(recent[c]) >= args.confirm and time.time() - last_alert > args.cooldown:
                last_alert = time.time()
                stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                path = Path(args.snapshots) / f"{stamp}_{c}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"[{stamp}] ТРЕВОГА: {c} -> {path}")
                send_telegram(frame, f"Обнаружен: {c} ({stamp})")
                recent[c].clear()

        if args.show:
            cv2.imshow("guardcam", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    dt = time.perf_counter() - t_start
    print(f"кадров {n_frames} ({n_frames / max(dt, 1e-6):.1f} FPS), кропов {n_crops}, "
          f"нейросеть: {1000 * t_nn / max(n_crops, 1):.2f} мс/кроп")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
