"""Собирает стартовый датасет кропов из COCO: person / animal / background.

Скачивает только аннотации (~250 МБ) и затем только нужные картинки по одной,
а не весь COCO (18 ГБ). Полные картинки на диск не сохраняются — только кропы.

    python build_coco_dataset.py --out data --per-class 6000

Результат:
    data/train/{animal,background,person}/*.jpg
    data/val/{animal,background,person}/*.jpg
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

from common import crop_with_margin

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Животные, которых реально можно встретить во дворе.
ANIMALS = ["bird", "cat", "dog", "horse", "sheep", "cow", "bear"]
# «Движущиеся» объекты, которые не должны поднимать тревогу о человеке.
BG_OBJECTS = ["car", "bicycle", "motorcycle", "truck", "umbrella", "potted plant",
              "chair", "bench", "backpack", "handbag", "suitcase", "kite", "frisbee",
              "sports ball", "fire hydrant", "stop sign"]


def download(url: str, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dst)


def load_annotations(cache: Path, split: str) -> dict:
    zip_path = cache / "annotations_trainval2017.zip"
    json_path = cache / f"instances_{split}.json"
    if not json_path.exists():
        download(ANN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            json_path.write_bytes(z.read(f"annotations/instances_{split}.json"))
    print(f"Читаю {json_path.name} (для train2017 нужно ~2–3 ГБ RAM)...")
    return json.loads(json_path.read_text())


def random_bg_box(W, H, objects, rng, min_side=40, tries=20):
    """Случайный прямоугольник, не задевающий ни людей, ни животных."""
    for _ in range(tries):
        w = rng.randint(min_side, max(min_side, W // 3))
        h = rng.randint(min_side, max(min_side, H // 2))
        if w >= W or h >= H:
            continue
        x, y = rng.randint(0, W - w), rng.randint(0, H - h)
        if all(x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y
               for ox, oy, ow, oh in objects):
            return x, y, w, h
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    ap.add_argument("--per-class", type=int, default=6000, help="примерно кропов на класс")
    ap.add_argument("--min-side", type=int, default=24, help="мин. сторона объекта в пикселях")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    coco = load_annotations(Path(args.cache), args.split)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    by_image = defaultdict(list)   # image_id -> [(group, bbox)]
    pool = defaultdict(list)       # group/subgroup -> [image_id]
    for a in coco["annotations"]:
        name = cat_name[a["category_id"]]
        x, y, w, h = (int(v) for v in a["bbox"])
        if a.get("iscrowd") or min(w, h) < args.min_side:
            # толпы и крошечные объекты всё равно учитываем как «занятое» место для фона
            group = None
        elif name == "person":
            group = "person"
        elif name in ANIMALS:
            group = "animal"
        elif name in BG_OBJECTS:
            group = "background"
        else:
            group = None
        by_image[a["image_id"]].append((group, name, (x, y, w, h)))
        if group:
            pool["animal:" + name if group == "animal" else group].append(a["image_id"])

    # Выбираем картинки так, чтобы покрыть квоты; животных — поровну по видам.
    quotas = {"person": args.per_class, "background": args.per_class // 2}
    quotas.update({f"animal:{n}": args.per_class // len(ANIMALS) for n in ANIMALS})
    chosen: set[int] = set()
    for key, quota in quotas.items():
        ids = list(dict.fromkeys(pool[key]))
        rng.shuffle(ids)
        count = 0
        for img_id in ids:
            if count >= quota:
                break
            chosen.add(img_id)
            count += sum(1 for g, n, _ in by_image[img_id]
                         if (g == "person" and key == "person") or
                            (g == "background" and key == "background") or
                            (g == "animal" and key == f"animal:{n}"))
    print(f"Картинок к скачиванию: {len(chosen)}")

    out = Path(args.out)
    for part in ("train", "val"):
        for cls in ("animal", "background", "person"):
            (out / part / cls).mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    session = requests.Session()

    def process(img_id: int):
        im = images[img_id]
        r = session.get(im["coco_url"], timeout=30)
        r.raise_for_status()
        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        part = "val" if img_id % 10 == 0 else "train"
        local_rng = random.Random(img_id)
        saved = []
        occupied = []
        for i, (group, name, box) in enumerate(by_image[img_id]):
            if name == "person" or name in ANIMALS:
                occupied.append(box)
            if group:
                path = out / part / group / f"{img_id}_{i}.jpg"
                cv2.imwrite(str(path), crop_with_margin(frame, box))
                saved.append(group)
        # Один кусок «пустого» фона с каждой картинки — имитация ложных срабатываний.
        bg = random_bg_box(im["width"], im["height"], occupied, local_rng)
        if bg:
            cv2.imwrite(str(out / part / "background" / f"{img_id}_bg.jpg"), crop_with_margin(frame, bg, 0))
            saved.append("background")
        return saved

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(process, i) for i in chosen]
        for f in tqdm(as_completed(futures), total=len(futures), desc="кропы"):
            try:
                for g in f.result():
                    counts[g] += 1
            except Exception as e:  # битая ссылка / сеть — пропускаем
                counts["errors"] += 1
                tqdm.write(f"пропуск: {e}")

    print("Готово:", dict(counts))


if __name__ == "__main__":
    main()
