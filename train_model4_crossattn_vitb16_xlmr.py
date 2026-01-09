# -*- coding: utf-8 -*-
"""
Model #4: Cross-Attention Fusion (CLIP ViT-B/16 image feats + XLM-R token hidden states)

Key fixes (Windows-friendly):
1) DataLoader multiprocessing (spawn) requires Dataset/collate_fn to be picklable.
   - Replaced non-picklable closure OCR getter with top-level OCRCache callable class.

2) Clean PIL warning for palette+transparency images:
   - If an image is mode "P" with transparency, convert to RGBA first, then to RGB.

Tips:
- If you still face any odd multiprocessing issues on Windows, run with: --num_workers 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import open_clip
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -------------------------
# Repro
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# OCR cache (picklable)
# -------------------------
@dataclass
class OCRCache:
    cache: Dict[str, str]

    def __call__(self, rel_id: str) -> str:
        # exact key
        if rel_id in self.cache:
            return self.cache.get(rel_id) or ""
        # filename key
        name = Path(rel_id).name
        if name in self.cache:
            return self.cache.get(name) or ""
        # without split prefix
        rel2 = "/".join(rel_id.split("/")[1:])
        if rel2 in self.cache:
            return self.cache.get(rel2) or ""
        return ""


def build_ocr_getter(ocr_cache_path: Optional[Path]) -> Callable[[str], str]:
    if ocr_cache_path is None:
        return OCRCache({})
    with open(ocr_cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if not isinstance(cache, dict):
        raise ValueError("ocr_cache must be a JSON dict: {key: text}")
    return OCRCache(cache)


# -------------------------
# Utils: scan dataset
# -------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_classes(data_root: Path) -> List[str]:
    train_dir = data_root / "train"
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes


def list_split_samples(data_root: Path, split: str, class2id: Dict[str, int]) -> List[Tuple[Path, int, str]]:
    split_dir = data_root / split
    items: List[Tuple[Path, int, str]] = []
    for cls_name, y in class2id.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel_id = f"{split}/{cls_name}/{p.name}"
                items.append((p, y, rel_id))
    return items


# -------------------------
# Dataset
# -------------------------
@dataclass
class Batch:
    images: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    rel_ids: List[str]


class AdsDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int, str]], preprocess, ocr_get: Callable[[str], str]):
        self.samples = samples
        self.preprocess = preprocess
        self.ocr_get = ocr_get

    def __len__(self) -> int:
        return len(self.samples)

    def _safe_load_image(self, path: Path) -> Image.Image:
        """
        Load image robustly.
        Fix PIL warning: palette images with transparency -> convert to RGBA first.
        """
        try:
            with Image.open(path) as im:
                # Fix warning: palette ("P") + transparency stored as bytes
                if im.mode == "P" and ("transparency" in im.info):
                    im = im.convert("RGBA")
                return im.convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (0, 0, 0))

    def __getitem__(self, idx: int):
        img_path, y, rel_id = self.samples[idx]
        img = self._safe_load_image(img_path)
        img_tensor = self.preprocess(img)
        text = (self.ocr_get(rel_id) or "").strip()
        return img_tensor, text, y, rel_id


class Collator:
    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch) -> Batch:
        images, texts, ys, rel_ids = zip(*batch)
        images = torch.stack(images, dim=0)

        enc = self.tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        labels = torch.tensor(ys, dtype=torch.long)

        return Batch(
            images=images,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
            rel_ids=list(rel_ids),
        )


# -------------------------
# Model: Cross-Attn Fusion
# -------------------------
class CrossAttnFusion(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, hidden: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(img_dim, hidden)
        self.k = nn.Linear(txt_dim, hidden)
        self.v = nn.Linear(txt_dim, hidden)

        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, img_feat: torch.Tensor, txt_hid: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        q = self.q(img_feat).unsqueeze(1)     # [B,1,H]
        k = self.k(txt_hid)                  # [B,S,H]
        v = self.v(txt_hid)                  # [B,S,H]

        key_padding_mask = (attn_mask == 0)
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln(q + attn_out)
        x2 = self.ff(x)
        x = self.ln2(x + x2)
        return x.squeeze(1)


class Model(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, num_classes: int, hidden: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.fusion = CrossAttnFusion(img_dim, txt_dim, hidden=hidden, nhead=nhead, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, img_feat: torch.Tensor, txt_hid: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(img_feat, txt_hid, attn_mask)
        return self.head(fused)


# -------------------------
# Encode helpers (grad controlled by freeze flags)
# -------------------------
def get_clip_embed_dim(clip_model) -> int:
    for attr in ["embed_dim", "output_dim"]:
        if hasattr(clip_model, attr):
            v = getattr(clip_model, attr)
            if isinstance(v, int):
                return v
    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "output_dim"):
        return int(clip_model.visual.output_dim)
    return 512


def encode_image(clip_model, images: torch.Tensor, freeze: bool) -> torch.Tensor:
    ctx = torch.no_grad() if freeze else nullcontext()
    with ctx:
        img_feat = clip_model.encode_image(images)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
    return img_feat


def encode_text_hidden(text_model, input_ids: torch.Tensor, attention_mask: torch.Tensor, freeze: bool) -> torch.Tensor:
    ctx = torch.no_grad() if freeze else nullcontext()
    with ctx:
        out = text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_hid = out.last_hidden_state
    return txt_hid


@torch.no_grad()
def evaluate(model: nn.Module, clip_model, text_model, loader: DataLoader, device: torch.device, use_amp: bool,
             freeze_clip: bool, freeze_text: bool,
             ablate_text: bool = False, ablate_image: bool = False):
    model.eval(); clip_model.eval(); text_model.eval()
    ys, ps = [], []

    img_dim = get_clip_embed_dim(clip_model)
    txt_dim = int(getattr(text_model.config, "hidden_size", 768))

    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if ablate_image:
                img_feat = torch.zeros(images.size(0), img_dim, device=device)
            else:
                img_feat = encode_image(clip_model, images, freeze=True)

            if ablate_text:
                txt_hid = torch.zeros(images.size(0), attention_mask.size(1), txt_dim, device=device)
                attn_mask = torch.ones_like(attention_mask)
            else:
                txt_hid = encode_text_hidden(text_model, input_ids, attention_mask, freeze=True)
                attn_mask = attention_mask

            logits = model(img_feat, txt_hid, attn_mask)

        pred = logits.argmax(dim=1)
        ys.extend(labels.detach().cpu().tolist())
        ps.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return float(acc), float(f1m), ys, ps


@torch.no_grad()
def predict_save_csv(model: nn.Module, clip_model, text_model, loader: DataLoader, device: torch.device, use_amp: bool,
                     out_csv: Path, class2id: Dict[str, int]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    model.eval(); clip_model.eval(); text_model.eval()

    id2class = {v: k for k, v in class2id.items()}
    rows = []
    ys, ps = [], []

    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            img_feat = encode_image(clip_model, images, freeze=True)
            txt_hid = encode_text_hidden(text_model, input_ids, attention_mask, freeze=True)
            logits = model(img_feat, txt_hid, attention_mask)

        probs = F.softmax(logits.float(), dim=1)
        pred = probs.argmax(dim=1)

        for i in range(images.size(0)):
            y = int(labels[i].item())
            p = int(pred[i].item())
            rows.append({
                "rel_id": batch.rel_ids[i],
                "y_true": y,
                "y_pred": p,
                "y_true_name": id2class.get(y, str(y)),
                "y_pred_name": id2class.get(p, str(p)),
                "conf": float(probs[i, p].item()),
            })

        ys.extend(labels.detach().cpu().tolist())
        ps.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        else:
            w = csv.DictWriter(f, fieldnames=["rel_id", "y_true", "y_pred", "conf"])
        w.writeheader()
        w.writerows(rows)

    return float(acc), float(f1m), ys, ps


def build_param_groups(model: nn.Module, clip_model, text_model, args) -> List[dict]:
    groups = [{"params": model.parameters(), "lr": args.lr, "weight_decay": args.wd}]
    if not args.freeze_clip:
        groups.append({"params": clip_model.parameters(), "lr": args.lr_clip, "weight_decay": args.wd})
    if not args.freeze_text:
        groups.append({"params": text_model.parameters(), "lr": args.lr_text, "weight_decay": args.wd})
    return groups


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ocr_cache", type=str, default=None)
    ap.add_argument("--run_dir", type=str, default="runs/model4_crossattn")

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=64)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_clip", type=float, default=1e-5)
    ap.add_argument("--lr_text", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--freeze_clip", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--early_stop", type=int, default=6)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_class_weights", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    classes = list_classes(data_root)
    class2id = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"[Classes] {num_classes}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    clip_model = clip_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    text_model = AutoModel.from_pretrained("xlm-roberta-base").to(device)

    img_dim = get_clip_embed_dim(clip_model)
    txt_dim = int(text_model.config.hidden_size)
    model = Model(img_dim, txt_dim, num_classes, hidden=args.hidden, nhead=args.nhead, dropout=args.dropout).to(device)

    if args.freeze_clip:
        for p in clip_model.parameters():
            p.requires_grad = False
    if args.freeze_text:
        for p in text_model.parameters():
            p.requires_grad = False
    print(f"[Freeze] clip={args.freeze_clip} | text={args.freeze_text}")

    ocr_get = build_ocr_getter(Path(args.ocr_cache) if args.ocr_cache else None)

    train_samples = list_split_samples(data_root, "train", class2id)
    val_samples = list_split_samples(data_root, "val", class2id)
    test_samples = list_split_samples(data_root, "test", class2id)
    print(f"[Samples] train={len(train_samples)} | val={len(val_samples)} | test={len(test_samples)}")

    train_ds = AdsDataset(train_samples, preprocess, ocr_get)
    val_ds = AdsDataset(val_samples, preprocess, ocr_get)
    test_ds = AdsDataset(test_samples, preprocess, ocr_get)

    collate_fn = Collator(tokenizer, args.max_len)

    # Windows: avoid persistent_workers by default
    persistent = False if os.name == "nt" else (args.num_workers > 0)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=persistent,
    )

    if args.use_class_weights:
        y_train = [y for _, y, _ in train_samples]
        w = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=np.array(y_train))
        w = torch.tensor(w, dtype=torch.float32, device=device)
        print("[OK] Using class weights")
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optim = torch.optim.AdamW(build_param_groups(model, clip_model, text_model, args))
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, total_steps))

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_epoch = -1
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        clip_model.train(not args.freeze_clip)
        text_model.train(not args.freeze_text)

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for batch in pbar:
            images = batch.images.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                img_feat = encode_image(clip_model, images, freeze=args.freeze_clip)
                txt_hid = encode_text_hidden(text_model, input_ids, attention_mask, freeze=args.freeze_text)
                logits = model(img_feat, txt_hid, attention_mask)
                loss = crit(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            pbar.set_postfix(loss=float(loss.item()))

        val_acc, val_f1, _, _ = evaluate(
            model, clip_model, text_model, val_loader, device, use_amp,
            freeze_clip=args.freeze_clip, freeze_text=args.freeze_text
        )
        print(f"[VAL] acc={val_acc:.4f} | macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = ep
            bad = 0
            torch.save(
                {"model": model.state_dict(),
                 "clip": clip_model.state_dict(),
                 "text": text_model.state_dict(),
                 "class2id": class2id,
                 "args": vars(args)},
                run_dir / "best.pt"
            )
            print("[OK] Saved best.pt")
        else:
            bad += 1
            if bad >= args.early_stop:
                print(f"[EARLY STOP] no improve for {args.early_stop} epochs.")
                break

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    try:
        clip_model.load_state_dict(ckpt["clip"], strict=False)
        text_model.load_state_dict(ckpt["text"], strict=False)
    except Exception:
        pass

    test_csv = run_dir / "preds_test.csv"
    test_acc, test_f1, ys, ps = predict_save_csv(model, clip_model, text_model, test_loader, device, use_amp, test_csv, class2id)
    print(f"[TEST] acc={test_acc:.4f} | macro_f1={test_f1:.4f}")

    tacc, tf1, _, _ = evaluate(
        model, clip_model, text_model, test_loader, device, use_amp,
        freeze_clip=args.freeze_clip, freeze_text=args.freeze_text, ablate_text=True
    )
    iacc, if1, _, _ = evaluate(
        model, clip_model, text_model, test_loader, device, use_amp,
        freeze_clip=args.freeze_clip, freeze_text=args.freeze_text, ablate_image=True
    )
    print(f"[TEST(text_off)] acc={tacc:.4f} | macro_f1={tf1:.4f}")
    print(f"[TEST(img_off)]  acc={iacc:.4f} | macro_f1={if1:.4f}")

    report_txt = classification_report(ys, ps, digits=4)
    with open(run_dir / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_text_off_macro_f1": float(tf1),
        "test_img_off_macro_f1": float(if1),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved folder: {run_dir}")
    print("[DONE] Model #4 Cross-Attn finished.")


if __name__ == "__main__":
    main()
