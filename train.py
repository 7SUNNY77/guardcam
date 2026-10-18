"""Обучение крошечного классификатора person / animal / background.

Модель: MobileNetV3-Small (предобучена на ImageNet), ~1 млн параметров.
Обучается на CPU: на ~15 тыс. кропов эпоха занимает несколько минут.

    python train.py --data data --epochs 15
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common import CLASSES, IMG_SIZE, MEAN, PAD_VALUE, STD


class Letterbox:
    """То же самое, что common.letterbox, только для PIL."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.pad(img.convert("RGB"), (self.size, self.size), color=(PAD_VALUE,) * 3)


class FarAway:
    """Имитация далёкого объекта: сжать до 12–40 px и растянуть обратно."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        s = random.randint(12, 40)
        return img.resize((s, s), Image.BILINEAR).resize(img.size, Image.BILINEAR)


class NightIR:
    """Имитация ночной ИК-съёмки: ч/б, низкий контраст, засветка, шум сенсора."""

    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:  # t: 3xHxW в [0, 1]
        if random.random() > self.p:
            return t
        g = (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).unsqueeze(0)
        g = g.pow(random.uniform(0.6, 1.5))                       # гамма
        g = g * random.uniform(0.5, 1.2) + random.uniform(-0.1, 0.15)
        if random.random() < 0.5:                                  # пятно ИК-подсветки в центре
            h, w = g.shape[1:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
            g = g + random.uniform(0.1, 0.4) * torch.exp(-(xx ** 2 + yy ** 2) * 2)
        g = g + torch.randn_like(g) * random.uniform(0.02, 0.08)  # шум
        return g.clamp(0, 1).expand(3, -1, -1).clone()


def make_transforms(train: bool):
    norm = transforms.Normalize(MEAN.tolist(), STD.tolist())
    if not train:
        return transforms.Compose([Letterbox(IMG_SIZE), transforms.ToTensor(), norm])
    return transforms.Compose([
        Letterbox(int(IMG_SIZE * 1.15)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        FarAway(p=0.25),
        transforms.ColorJitter(0.4, 0.4, 0.3, 0.05),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        NightIR(p=0.35),
        norm,
    ])


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    return m


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)  # [истина, предсказание]
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
    recall = cm.diagonal() / cm.sum(1).clip(min=1)
    p = CLASSES.index("person")
    not_person = [i for i in range(len(CLASSES)) if i != p]
    return {
        "cm": cm,
        "recall": dict(zip(CLASSES, recall.round(4).tolist())),
        "balanced_acc": float(recall.mean()),
        # Главная «бытовая» метрика: как часто не-человек поднимает тревогу «человек».
        "false_person_rate": float(cm[not_person, p].sum() / cm[not_person].sum().clip(min=1)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="models/best.pt")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = datasets.ImageFolder(Path(args.data) / "train", make_transforms(True))
    val_ds = datasets.ImageFolder(Path(args.data) / "val", make_transforms(False))
    assert train_ds.classes == CLASSES, f"ожидались папки {CLASSES}, найдены {train_ds.classes}"

    counts = np.bincount(train_ds.targets, minlength=len(CLASSES))
    print("train:", dict(zip(CLASSES, counts.tolist())), "| val:", len(val_ds), "| device:", device)
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts.clip(min=1)), dtype=torch.float32)

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          persistent_workers=args.workers > 0, drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers)

    model = build_model(not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)

    best = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        m = evaluate(model, val_dl, device)
        print(f"эпоха {epoch:2d} | loss {total / n:.3f} | bal_acc {m['balanced_acc']:.3f} | "
              f"ложный 'человек' {m['false_person_rate']:.3%} | recall {m['recall']} | "
              f"{time.perf_counter() - t0:.0f}s")
        if m["balanced_acc"] > best:
            best = m["balanced_acc"]
            torch.save({"model": model.state_dict(), "classes": CLASSES, "img_size": IMG_SIZE}, args.out)

    print(f"Лучшая balanced accuracy: {best:.3f} -> {args.out}")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("Матрица ошибок (строки — истина, столбцы — предсказание):", CLASSES)
    print(evaluate(model, val_dl, device)["cm"])


if __name__ == "__main__":
    main()
