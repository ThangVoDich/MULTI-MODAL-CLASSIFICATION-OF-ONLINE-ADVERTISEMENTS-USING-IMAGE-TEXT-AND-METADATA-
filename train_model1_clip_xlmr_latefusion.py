# -*- coding: utf-8 -*-
"""
Model #1: CLIP image + XLM-R text -> concat -> MLP (Late Fusion)

Fixes (Windows-safe + stable export):
- robust image loading (avoid DataLoader worker crash)
- persistent_workers OFF by default
- export val/test uses num_workers=0 (avoid worker exit unexpectedly)
- softmax(logits.float()) for numerical stability
- optional class weights + label smoothing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

import open_clip
from transformers import AutoTokenizer, AutoModel


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    # make workers deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] ocr_cache not found: {p} -> using empty cache")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_class_map(train_dir: Path) -> Dict[str, int]:
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in: {train_dir}")
    return {c: i for i, c in enumerate(classes)}


def list_split_samples(data_root: Path, split: str, class2id: Dict[str, int]) -> List[Tuple[Path, str, int]]:
    """
    Returns list of (img_path, relpath_posix, y_id)
    Assumes structure: data_root/{split}/{class_name}/*.jpg
    """
    split_dir = data_root / split
    samples = []
    for cls_name, y in class2id.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel = p.relative_to(data_root).as_posix()  # key matches ocr_cache
                samples.append((p, rel, y))

    samples.sort(key=lambda x: str(x[0]).lower())
    return samples


def compute_class_weights(train_samples: List[Tuple[Path, str, int]], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, _, y in train_samples:
        counts[y] += 1
    counts = np.maximum(counts, 1)  # avoid div0
    # inverse frequency, normalized by mean
    w = (counts.mean() / counts).astype(np.float32)
    return torch.tensor(w, dtype=torch.float32)


# -----------------------------
# Dataset / Collate
# -----------------------------
class AdsDataset(Dataset):
    def __init__(self, samples, preprocess, ocr_cache: Dict[str, str]):
        self.samples = samples
        self.preprocess = preprocess
        self.ocr_cache = ocr_cache
        self.bad_count = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, rel, y = self.samples[idx]

        # robust image load (avoid worker crash)
        try:
            img = Image.open(img_path)
            if img.mode in ["P", "RGBA"]:
                img = img.convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGB")
            img_t = self.preprocess(img)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            # return a black image tensor; keep training running
            self.bad_count += 1
            # CLIP preprocess expects PIL normally; easiest: create RGB black PIL then preprocess
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            img_t = self.preprocess(img)
            # You can uncomment to see which file is broken:
            # print(f"[WARN] bad image: {img_path} | {type(e).__name__}: {e}")

        text = self.ocr_cache.get(rel, "")
        return img_t, text, y, rel


@dataclass
class Batch:
    images: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    relpaths: list


def collate_batch(batch_list, tokenizer, max_len: int):
    images, texts, labels, rels = zip(*batch_list)
    images = torch.stack(images, dim=0)

    tok = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return Batch(images, tok["input_ids"], tok["attention_mask"], labels, list(rels))


# -----------------------------
# Model
# -----------------------------
class LateFusionMLP(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, num_classes: int, hidden=512, dropout=0.2):
        super().__init__()
        self.img_ln = nn.LayerNorm(img_dim)
        self.txt_ln = nn.LayerNorm(txt_dim)
        self.head = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, img_feat, txt_feat):
        img_feat = self.img_ln(img_feat)
        txt_feat = self.txt_ln(txt_feat)
        x = torch.cat([img_feat, txt_feat], dim=-1)
        return self.head(x)


def encode_batch(clip_model, text_model, images, input_ids, attention_mask):
    # CLIP image feature
    img_feat = clip_model.encode_image(images)
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)

    # XLM-R CLS feature
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    txt_feat = out.last_hidden_state[:, 0, :]
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

    return img_feat, txt_feat


@torch.no_grad()
def evaluate(fusion, clip_model, text_model, loader, device, use_amp: bool):
    fusion.eval()
    clip_model.eval()
    text_model.eval()

    ys, ps = [], []
    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            img_feat, txt_feat = encode_batch(clip_model, text_model, images, input_ids, attention_mask)
            logits = fusion(img_feat, txt_feat)

        pred = logits.argmax(dim=1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return float(acc), float(f1m), ys, ps


@torch.no_grad()
def predict_and_save_csv(fusion, clip_model, text_model, loader, device, use_amp: bool, out_csv: Path):
    fusion.eval()
    clip_model.eval()
    text_model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    ys, ps = [], []

    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            img_feat, txt_feat = encode_batch(clip_model, text_model, images, input_ids, attention_mask)
            logits = fusion(img_feat, txt_feat)
            probs = F.softmax(logits.float(), dim=1)  # float for stability (same idea as model2)

        pred = probs.argmax(dim=1)
        pmax = probs.max(dim=1).values

        y_true = labels.cpu().tolist()
        y_pred = pred.cpu().tolist()
        p_max = pmax.cpu().tolist()

        ys.extend(y_true)
        ps.extend(y_pred)

        for rel, yt, yp, pm in zip(batch.relpaths, y_true, y_pred, p_max):
            rows.append([rel, int(yt), int(yp), float(pm)])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "y_true", "y_pred", "p_max"])
        w.writerows(rows)

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return float(acc), float(f1m), ys, ps


def make_loader(ds, batch_size, shuffle, tokenizer, max_len, device, num_workers: int):
    collate_fn = partial(collate_batch, tokenizer=tokenizer, max_len=max_len)

    # IMPORTANT (Windows-safe): persistent_workers=False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def export_all(run_dir: Path, fusion, clip_model, text_model,
               val_ds, test_ds, tokenizer, args, device, use_amp: bool,
               best_epoch: int, best_val_macro_f1: float):
    # Export loaders with num_workers=0 to avoid "worker exited unexpectedly"
    export_val_loader = make_loader(
        val_ds, args.batch_size, False, tokenizer, args.max_len, device, num_workers=0
    )
    export_test_loader = make_loader(
        test_ds, args.batch_size, False, tokenizer, args.max_len, device, num_workers=0
    )

    # ✅ VAL export
    val_csv = run_dir / "preds_val.csv"
    val_acc, val_f1, yv, pv = predict_and_save_csv(
        fusion, clip_model, text_model, export_val_loader, device, use_amp, val_csv
    )
    with open(run_dir / "classification_report_val.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(yv, pv, digits=4))

    # ✅ TEST export
    test_csv = run_dir / "preds_test.csv"
    test_acc, test_f1, yt, pt = predict_and_save_csv(
        fusion, clip_model, text_model, export_test_loader, device, use_amp, test_csv
    )
    with open(run_dir / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(yt, pt, digits=4))

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[TEST] acc={test_acc:.4f} | macro_f1={test_f1:.4f}")
    print(f"[OK] Saved: {run_dir / 'best.pt'}")
    print(f"[OK] Saved: {run_dir / 'metrics.json'}")
    print(f"[OK] Saved: {run_dir / 'preds_val.csv'}")
    print(f"[OK] Saved: {run_dir / 'preds_test.csv'}")
    print(f"[OK] Saved: {run_dir / 'classification_report_val.txt'}")
    print(f"[OK] Saved: {run_dir / 'classification_report_test.txt'}")


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--ocr_cache", type=str, default="ocr_cache.json")

    # output
    ap.add_argument("--out_dir", type=str, default="runs/model1_latefusion")

    # train
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)  # safer default on Windows
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_stop", type=int, default=6)
    ap.add_argument("--no_amp", action="store_true")

    # backbones
    ap.add_argument("--clip_model", type=str, default="ViT-B-16")
    ap.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    ap.add_argument("--text_model", type=str, default="xlm-roberta-base")
    ap.add_argument("--freeze_clip", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")

    # stabilization knobs
    ap.add_argument("--use_class_weights", action="store_true", default=True)
    ap.add_argument("--label_smoothing", type=float, default=0.1)  # set 0 to disable
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load ocr cache
    ocr_cache = read_json(args.ocr_cache)

    # class map
    class2id = build_class_map(data_root / "train")
    num_classes = len(class2id)
    print(f"[Classes] {num_classes}")

    # models
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_model = clip_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModel.from_pretrained(args.text_model).to(device)

    if args.freeze_clip:
        for p in clip_model.parameters():
            p.requires_grad = False
        print("[OK] Freeze CLIP")

    if args.freeze_text:
        for p in text_model.parameters():
            p.requires_grad = False
        print("[OK] Freeze XLM-R")

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        img_dim = clip_model.encode_image(dummy).shape[-1]
    txt_dim = text_model.config.hidden_size

    fusion = LateFusionMLP(img_dim, txt_dim, num_classes, hidden=512, dropout=0.2).to(device)

    # data lists
    train_samples = list_split_samples(data_root, "train", class2id)
    val_samples = list_split_samples(data_root, "val", class2id)
    test_samples = list_split_samples(data_root, "test", class2id)
    print(f"[Samples] train={len(train_samples)} | val={len(val_samples)} | test={len(test_samples)}")

    train_ds = AdsDataset(train_samples, preprocess, ocr_cache)
    val_ds = AdsDataset(val_samples, preprocess, ocr_cache)
    test_ds = AdsDataset(test_samples, preprocess, ocr_cache)

    train_loader = make_loader(
        train_ds, args.batch_size, True, tokenizer, args.max_len, device, num_workers=args.num_workers
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, tokenizer, args.max_len, device, num_workers=max(0, min(args.num_workers, 2))
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False, tokenizer, args.max_len, device, num_workers=max(0, min(args.num_workers, 2))
    )

    # optimizer
    params = [p for p in list(clip_model.parameters()) + list(text_model.parameters()) + list(fusion.parameters())
              if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # loss
    weight = None
    if args.use_class_weights:
        weight = compute_class_weights(train_samples, num_classes).to(device)
        print("[OK] Using class weights")

    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=max(0.0, float(args.label_smoothing)))

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # save config
    config = vars(args).copy()
    config["num_classes"] = num_classes
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        fusion.train()
        clip_model.train()
        text_model.train()

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
        running = 0.0

        for batch in pbar:
            images = batch.images.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                img_feat, txt_feat = encode_batch(clip_model, text_model, images, input_ids, attention_mask)
                logits = fusion(img_feat, txt_feat)
                loss = crit(logits, labels)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))
            scaler.step(optim)
            scaler.update()

            running = 0.9 * running + 0.1 * float(loss.item()) if epoch > 1 else float(loss.item())
            pbar.set_postfix(loss=f"{running:.3f}")

        # eval
        val_acc, val_f1, _, _ = evaluate(fusion, clip_model, text_model, val_loader, device, use_amp)
        print(f"[VAL] acc={val_acc:.4f} | macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "fusion": fusion.state_dict(),
                    "clip": clip_model.state_dict(),
                    "text": text_model.state_dict(),
                    "class2id": class2id,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
            print("[OK] Saved best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop:
                print(f"[EARLY STOP] no improve for {args.early_stop} epochs.")
                break

    # load best and export
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    fusion.load_state_dict(ckpt["fusion"])
    # clip/text weights only if you finetune them; safe to load anyway
    try:
        clip_model.load_state_dict(ckpt["clip"])
        text_model.load_state_dict(ckpt["text"])
    except Exception as e:
        print(f"[WARN] Could not load clip/text weights fully: {e}")

    export_all(
        out_dir, fusion, clip_model, text_model,
        val_ds, test_ds, tokenizer, args, device, use_amp,
        best_epoch=best_epoch, best_val_macro_f1=best_f1
    )

    # show dataset issues
    if getattr(train_ds, "bad_count", 0) > 0:
        print(f"[WARN] bad images encountered in train: {train_ds.bad_count}")
    if getattr(val_ds, "bad_count", 0) > 0:
        print(f"[WARN] bad images encountered in val: {val_ds.bad_count}")
    if getattr(test_ds, "bad_count", 0) > 0:
        print(f"[WARN] bad images encountered in test: {test_ds.bad_count}")

    print("[DONE] Model #1 (Late Fusion) finished.")


if __name__ == "__main__":
    main()
