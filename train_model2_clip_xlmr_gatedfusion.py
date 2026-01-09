# train_model2_clip_xlmr_gatedfusion.py
# Model 2: CLIP (image) + XLM-R (text) + Gated Fusion
# - Gate temperature + clamp to avoid ignoring text
# - Gate regularization to keep balance image/text
# - Class-weighted CE + label smoothing to boost macro-F1
# - Separate LR for backbone vs head
# - Robust image loading + eval_workers default 0 on Windows

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip
from transformers import AutoTokenizer, AutoModel

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid crash on truncated images


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_map(train_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class2id = {c: i for i, c in enumerate(classes)}
    id2class = {i: c for c, i in class2id.items()}
    return class2id, id2class


def list_split_samples(data_root: Path, split: str, class2id: Dict[str, int]):
    split_dir = data_root / split
    samples = []
    if not split_dir.exists():
        return samples

    for cls_name, y in class2id.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel = p.relative_to(data_root).as_posix()
                samples.append((p, rel, y))

    samples.sort(key=lambda x: str(x[0]).lower())
    return samples


# -----------------------------
# Dataset
# -----------------------------
class AdsDataset(Dataset):
    def __init__(self, samples, preprocess, ocr_cache: Dict[str, str]):
        self.samples = samples
        self.preprocess = preprocess
        self.ocr_cache = ocr_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, rel, y = self.samples[idx]

        # robust image load
        try:
            img = Image.open(img_path)
            if img.mode in ["P", "RGBA"]:
                img = img.convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGB")
        except Exception:
            # fallback: black image (avoid dataloader worker crash)
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        img_t = self.preprocess(img)
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
# Model (Gated Fusion)
# -----------------------------
class GatedFusionClassifier(nn.Module):
    """
    gate logits -> sigmoid(temp) -> clamp -> fused = g*img + (1-g)*txt
    """
    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        num_classes: int,
        d_model: int = 512,
        hidden: int = 512,
        dropout: float = 0.2,
        gate_min: float = 0.20,
        gate_temp: float = 2.0,
    ):
        super().__init__()
        self.img_ln = nn.LayerNorm(img_dim)
        self.txt_ln = nn.LayerNorm(txt_dim)

        self.img_proj = nn.Linear(img_dim, d_model)
        self.txt_proj = nn.Linear(txt_dim, d_model)

        # gate logits (NO sigmoid here)
        self.gate_fc = nn.Linear(d_model * 2, d_model)
        self.gate_min = float(gate_min)
        self.gate_temp = float(gate_temp)

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, img_feat, txt_feat):
        img_feat = self.img_ln(img_feat)
        txt_feat = self.txt_ln(txt_feat)

        img_p = self.img_proj(img_feat)
        txt_p = self.txt_proj(txt_feat)

        g_logits = self.gate_fc(torch.cat([img_p, txt_p], dim=-1))  # B x D
        g = torch.sigmoid(g_logits / self.gate_temp)

        # clamp: g ∈ [gate_min, 1-gate_min] to prevent ignoring one modality
        g = self.gate_min + (1.0 - 2.0 * self.gate_min) * g

        fused = g * img_p + (1.0 - g) * txt_p
        logits = self.head(fused)
        return logits, g


# -----------------------------
# Encode (allow finetune when not frozen)
# -----------------------------
def encode_batch(
    clip_model,
    text_model,
    images,
    input_ids,
    attention_mask,
    grad_clip: bool,
    grad_text: bool,
):
    if grad_clip:
        img_feat = clip_model.encode_image(images)
    else:
        with torch.no_grad():
            img_feat = clip_model.encode_image(images)
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)

    if grad_text:
        out = text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = out.last_hidden_state[:, 0, :]
    else:
        with torch.no_grad():
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
            img_feat, txt_feat = encode_batch(
                clip_model, text_model, images, input_ids, attention_mask,
                grad_clip=False, grad_text=False
            )
            logits, _g = fusion(img_feat, txt_feat)

        pred = logits.argmax(dim=1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return acc, f1m


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
            img_feat, txt_feat = encode_batch(
                clip_model, text_model, images, input_ids, attention_mask,
                grad_clip=False, grad_text=False
            )
            logits, g = fusion(img_feat, txt_feat)
            probs = F.softmax(logits, dim=1)

        pred = probs.argmax(dim=1)
        pmax = probs.max(dim=1).values

        y_true = labels.cpu().tolist()
        y_pred = pred.cpu().tolist()
        p_max = pmax.cpu().tolist()

        ys.extend(y_true)
        ps.extend(y_pred)

        # gate mean per sample (debug)
        g_mean = g.mean(dim=1).float().cpu().tolist()

        for rel, yt, yp, pm, gm in zip(batch.relpaths, y_true, y_pred, p_max, g_mean):
            rows.append([rel, yt, yp, float(pm), float(gm)])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "y_true", "y_pred", "p_max", "gate_mean"])
        w.writerows(rows)

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return acc, f1m, ys, ps


def export_all(run_dir: Path, best_epoch: int, best_val_f1: float, test_acc: float, test_f1: float):
    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--ocr_cache", type=str, default="ocr_cache.json")

    # output folder
    ap.add_argument("--run_dir", type=str, default="runs/model2_gatedfusion")

    # train
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_workers", type=int, default=0)  # safer on Windows
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=None)   # default = lr*0.1
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_stop", type=int, default=4)

    # fusion regularization & smoothing
    ap.add_argument("--gate_reg", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    # backbones
    ap.add_argument("--clip_model", type=str, default="ViT-B-16")
    ap.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    ap.add_argument("--text_model", type=str, default="xlm-roberta-base")

    # options
    ap.add_argument("--freeze_clip", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # load OCR
    with open(args.ocr_cache, "r", encoding="utf-8") as f:
        ocr_cache = json.load(f)

    # class map
    class2id, id2class = build_class_map(data_root / "train")
    num_classes = len(class2id)
    print("[Classes]", num_classes)

    with open(run_dir / "class2id.json", "w", encoding="utf-8") as f:
        json.dump(class2id, f, ensure_ascii=False, indent=2)

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
        clip_model.eval()
        print("[OK] Freeze CLIP")

    if args.freeze_text:
        for p in text_model.parameters():
            p.requires_grad = False
        text_model.eval()
        print("[OK] Freeze XLM-R")

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        img_dim = clip_model.encode_image(dummy).shape[-1]
    txt_dim = text_model.config.hidden_size

    fusion = GatedFusionClassifier(
        img_dim, txt_dim, num_classes,
        d_model=512, hidden=512, dropout=0.2,
        gate_min=0.20, gate_temp=2.0
    ).to(device)

    # data lists
    train_samples = list_split_samples(data_root, "train", class2id)
    val_samples = list_split_samples(data_root, "val", class2id)
    test_samples = list_split_samples(data_root, "test", class2id)
    print(f"[Samples] train={len(train_samples)} | val={len(val_samples)} | test={len(test_samples)}")

    # class weights (imbalance)
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _p, _rel, y in train_samples:
        counts[y] += 1
    w = 1.0 / torch.sqrt(counts.float().clamp(min=1))
    w = w / w.mean()
    w = w.to(device)

    train_ds = AdsDataset(train_samples, preprocess, ocr_cache)
    val_ds = AdsDataset(val_samples, preprocess, ocr_cache)
    test_ds = AdsDataset(test_samples, preprocess, ocr_cache)

    collate_fn = partial(collate_batch, tokenizer=tokenizer, max_len=args.max_len)

    def make_loader(ds, shuffle, workers: int):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=True if (workers > 0) else False,
            collate_fn=collate_fn,
        )

    train_loader = make_loader(train_ds, True, args.num_workers)
    val_loader = make_loader(val_ds, False, args.eval_workers)
    test_loader = make_loader(test_ds, False, args.eval_workers)

    # loss
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)

    # optimizer (separate LR)
    lr_backbone = args.lr_backbone if args.lr_backbone is not None else args.lr * 0.1
    param_groups = [{"params": fusion.parameters(), "lr": args.lr, "weight_decay": args.wd}]
    if not args.freeze_clip:
        param_groups.append({"params": clip_model.parameters(), "lr": lr_backbone, "weight_decay": args.wd})
    if not args.freeze_text:
        param_groups.append({"params": text_model.parameters(), "lr": lr_backbone, "weight_decay": args.wd})
    optim = torch.optim.AdamW(param_groups)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # save config
    config = vars(args).copy()
    config["num_classes"] = num_classes
    config["lr_backbone_effective"] = lr_backbone
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_f1 = -1.0
    best_epoch = -1
    bad = 0

    for ep in range(1, args.epochs + 1):
        fusion.train()
        if not args.freeze_clip:
            clip_model.train()
        if not args.freeze_text:
            text_model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for batch in pbar:
            images = batch.images.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                img_feat, txt_feat = encode_batch(
                    clip_model, text_model, images, input_ids, attention_mask,
                    grad_clip=(not args.freeze_clip),
                    grad_text=(not args.freeze_text),
                )
                logits, g = fusion(img_feat, txt_feat)

                ce = crit(logits, labels)
                reg = (g - 0.5).pow(2).mean()  # keep gate around 0.5
                loss = ce + args.gate_reg * reg

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            pbar.set_postfix(
                loss=float(loss.item()),
                ce=float(ce.item()),
                g=float(g.mean().item())
            )

        val_acc, val_f1 = evaluate(fusion, clip_model, text_model, val_loader, device, use_amp)
        print(f"[VAL] acc={val_acc:.4f} | macro_f1={val_f1:.4f}")

        torch.save(
            {"fusion": fusion.state_dict(),
             "clip": clip_model.state_dict(),
             "text": text_model.state_dict()},
            run_dir / "last.pt"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = ep
            bad = 0
            torch.save(
                {"fusion": fusion.state_dict(),
                 "clip": clip_model.state_dict(),
                 "text": text_model.state_dict(),
                 "class2id": class2id},
                run_dir / "best.pt"
            )
            print("[OK] Saved best.pt")
        else:
            bad += 1
            if bad >= args.early_stop:
                print("[EarlyStop]")
                break

    # load best
    best_path = run_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        fusion.load_state_dict(ckpt["fusion"])
        clip_model.load_state_dict(ckpt["clip"])
        text_model.load_state_dict(ckpt["text"])

    # test + save preds
    test_csv = run_dir / "preds_test.csv"
    test_acc, test_f1, ys, ps = predict_and_save_csv(
        fusion, clip_model, text_model, test_loader, device, use_amp, test_csv
    )

    report_txt = classification_report(ys, ps, digits=4, zero_division=0)
    with open(run_dir / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    export_all(run_dir, best_epoch, best_f1, test_acc, test_f1)

    print(f"[TEST] acc={test_acc:.4f} | macro_f1={test_f1:.4f}")
    print(f"[OK] Saved: {run_dir / 'best.pt'}")
    print(f"[OK] Saved: {run_dir / 'metrics.json'}")
    print(f"[OK] Saved: {run_dir / 'preds_test.csv'}")
    print(f"[OK] Saved: {run_dir / 'classification_report_test.txt'}")
    print("[DONE] Model #2 (Gated Fusion) finished.")


if __name__ == "__main__":
    main()
