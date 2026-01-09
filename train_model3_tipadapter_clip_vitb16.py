import argparse, json, csv, os, random
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm

import open_clip
from sklearn.metrics import accuracy_score, f1_score, classification_report

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -----------------------
# Utils
# -----------------------
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


class OCRLookup:
    """pickle OK on Windows workers"""
    def __init__(self, ocr_cache: Dict[str, str]):
        self.ocr = ocr_cache

    def __call__(self, rel: str) -> str:
        candidates = [
            rel,
            rel.replace("/", "\\"),
            "data/" + rel,
            "data\\" + rel.replace("/", "\\"),
            "dataset_split/" + rel,
            "dataset_split\\" + rel.replace("/", "\\"),
        ]
        for k in candidates:
            v = self.ocr.get(k, None)
            if v is not None:
                return v
        return ""


# -----------------------
# Dataset / Collate
# -----------------------
class AdsDataset(Dataset):
    def __init__(self, samples, preprocess, ocr_get: OCRLookup):
        self.samples = samples
        self.preprocess = preprocess
        self.ocr_get = ocr_get

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, rel, y = self.samples[idx]

        # robust image loading: tránh worker chết vì 1 ảnh lỗi
        try:
            img = Image.open(img_path)
            if img.mode in ["P", "RGBA"]:
                img = img.convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        img_t = self.preprocess(img)

        text = self.ocr_get(rel)
        text = "" if (not text or not str(text).strip()) else str(text).strip()
        return img_t, text, y, rel


@dataclass
class Batch:
    images: torch.Tensor
    texts: List[str]
    labels: torch.Tensor
    relpaths: List[str]


def collate_simple(batch_list):
    imgs, txts, ys, rels = zip(*batch_list)
    imgs = torch.stack(imgs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return Batch(imgs, list(txts), ys, list(rels))


# -----------------------
# Feature helpers
# -----------------------
@torch.inference_mode()
def encode_fused_features(
    clip_model,
    images: torch.Tensor,
    texts: List[str],
    device: torch.device,
    use_amp: bool,
    fusion_alpha: float,
    empty_text_policy: str = "image_only",  # image_only | zero | keep
    ablate_text: bool = False,
    ablate_image: bool = False,
):
    """
    fused = normalize( fusion_alpha*img_feat + (1-fusion_alpha)*txt_feat )

    empty_text_policy:
      - image_only: nếu OCR rỗng -> fused = img_feat
      - zero:       nếu OCR rỗng -> txt_feat = 0 (fused vẫn theo alpha)
      - keep:       OCR rỗng vẫn encode_text("") (thường không tốt)
    """
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        img_feat = clip_model.encode_image(images)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)

        if ablate_text:
            txt_feat = torch.zeros_like(img_feat)
            has_text = torch.zeros(img_feat.size(0), device=device, dtype=torch.bool)
        else:
            has_text = torch.tensor([len(t) > 0 for t in texts], device=device, dtype=torch.bool)
            if empty_text_policy == "keep":
                tokens = open_clip.tokenize(texts).to(device)
                txt_feat = clip_model.encode_text(tokens)
                txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                # encode text only for non-empty (đỡ nhiễu + nhanh hơn)
                txt_feat = torch.zeros_like(img_feat)
                if has_text.any():
                    idx = has_text.nonzero(as_tuple=True)[0]
                    texts_non_empty = [texts[i] for i in idx.tolist()]
                    tokens = open_clip.tokenize(texts_non_empty).to(device)
                    tfeat = clip_model.encode_text(tokens)
                    tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-8)
                    txt_feat[idx] = tfeat

                if empty_text_policy == "zero":
                    pass
                elif empty_text_policy == "image_only":
                    # sẽ handle ở dưới: chỗ has_text=False -> fused=img_feat
                    pass

        if ablate_image:
            img_feat = torch.zeros_like(img_feat)

        fused = fusion_alpha * img_feat + (1.0 - fusion_alpha) * txt_feat

        if (not ablate_text) and (empty_text_policy == "image_only"):
            # OCR rỗng -> dùng img_feat thuần
            fused = torch.where(has_text.unsqueeze(1), fused, img_feat)

        fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)

    return fused


@torch.inference_mode()
def build_cache_and_prototypes(
    clip_model,
    loader,
    device,
    use_amp: bool,
    num_classes: int,
    fusion_alpha: float,
    empty_text_policy: str,
):
    """
    Cache:
      K: [N,D] train fused features
      y: [N] labels
      V: [N,C] one-hot
    Prototypes:
      P: [C,D] mean of class features
    """
    clip_model.eval()

    feats = []
    ys = []
    lens = []

    for batch in tqdm(loader, desc="Build cache (train)", leave=False):
        images = batch.images.to(device, non_blocking=True)
        fused = encode_fused_features(
            clip_model, images, batch.texts, device, use_amp,
            fusion_alpha=fusion_alpha,
            empty_text_policy=empty_text_policy
        )
        feats.append(fused.detach().cpu())
        ys.append(batch.labels.detach().cpu())
        lens.extend([len(t.strip()) for t in batch.texts])

    K = torch.cat(feats, dim=0)  # [N,D] CPU
    y = torch.cat(ys, dim=0)     # [N]
    N, D = K.shape

    # prototypes
    P = torch.zeros(num_classes, D, dtype=K.dtype)
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(N):
        cls = int(y[i].item())
        P[cls] += K[i]
        counts[cls] += 1
    counts = counts.clamp(min=1).unsqueeze(1)
    P = P / counts
    P = P / (P.norm(dim=-1, keepdim=True) + 1e-8)

    # one-hot V
    V = torch.zeros(N, num_classes, dtype=torch.float16)
    V[torch.arange(N), y] = 1.0

    ocr_non_empty = sum(1 for L in lens if L > 0)
    avg_len = sum(lens) / max(1, len(lens))

    stats = {
        "train_cache_N": int(N),
        "feat_dim": int(D),
        "train_text_non_empty_rate": float(ocr_non_empty / max(1, len(lens))),
        "train_text_avg_len": float(avg_len),
    }
    return K, V, P, y, stats


@torch.inference_mode()
def tipadapter_logits(
    q: torch.Tensor,         # [B,D] GPU
    K: torch.Tensor,         # [N,D] GPU
    V: torch.Tensor,         # [N,C] GPU
    P: torch.Tensor,         # [C,D] GPU
    logit_scale: float,
    beta: float,
    cache_alpha: float,
    topk: int = 0,
    exp_clip: float = 50.0,  # clamp beta*sim để tránh overflow
):
    """
    proto logits: logit_scale * q @ P.T
    cache logits: exp(beta * q @ K.T) @ V
    final = proto + cache_alpha * cache
    """
    target_dtype = K.dtype
    if q.dtype != target_dtype:
        q = q.to(target_dtype)
    if P.dtype != target_dtype:
        P = P.to(target_dtype)
    if V.dtype != target_dtype:
        V = V.to(target_dtype)

    # proto logits
    proto = (q @ P.t()).float() * float(logit_scale)  # [B,C] float32

    # cache logits
    sim = (q @ K.t()).float()  # [B,N] float32

    if topk and topk > 0 and topk < K.shape[0]:
        sim_top, idx = sim.topk(topk, dim=1)  # [B,topk]
        x = float(beta) * sim_top
        x = torch.clamp(x, min=-exp_clip, max=exp_clip)
        aff = torch.exp(x)  # float32
        V_sel = V[idx].float()  # [B,topk,C]
        cache = (aff.unsqueeze(-1) * V_sel).sum(dim=1)  # [B,C]
    else:
        x = float(beta) * sim
        x = torch.clamp(x, min=-exp_clip, max=exp_clip)
        aff = torch.exp(x)         # [B,N]
        cache = aff @ V.float()    # [B,C]

    logits = proto + float(cache_alpha) * cache
    return logits


@torch.inference_mode()
def evaluate_split(
    clip_model,
    loader,
    device,
    use_amp: bool,
    K_gpu,
    V_gpu,
    P_gpu,
    fusion_alpha: float,
    empty_text_policy: str,
    beta: float,
    cache_alpha: float,
    topk: int,
    ablate_text: bool = False,
    ablate_image: bool = False,
):
    clip_model.eval()
    ys, ps = [], []

    try:
        logit_scale = float(clip_model.logit_scale.exp().item())
    except Exception:
        logit_scale = 10.0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch.images.to(device, non_blocking=True)
        fused = encode_fused_features(
            clip_model, images, batch.texts, device, use_amp,
            fusion_alpha=fusion_alpha,
            empty_text_policy=empty_text_policy,
            ablate_text=ablate_text,
            ablate_image=ablate_image,
        )

        logits = tipadapter_logits(
            fused, K_gpu, V_gpu, P_gpu,
            logit_scale=logit_scale,
            beta=beta,
            cache_alpha=cache_alpha,
            topk=topk
        )
        pred = logits.argmax(dim=1)

        ys.extend(batch.labels.tolist())
        ps.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return float(acc), float(f1m), ys, ps


@torch.inference_mode()
def predict_and_save_csv(
    clip_model,
    loader,
    device,
    use_amp: bool,
    K_gpu,
    V_gpu,
    P_gpu,
    fusion_alpha: float,
    empty_text_policy: str,
    beta: float,
    cache_alpha: float,
    topk: int,
    out_csv: Path
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        logit_scale = float(clip_model.logit_scale.exp().item())
    except Exception:
        logit_scale = 10.0

    rows = []
    ys, ps = [], []

    for batch in tqdm(loader, desc="Predict", leave=False):
        images = batch.images.to(device, non_blocking=True)
        fused = encode_fused_features(
            clip_model, images, batch.texts, device, use_amp,
            fusion_alpha=fusion_alpha,
            empty_text_policy=empty_text_policy,
        )

        logits = tipadapter_logits(
            fused, K_gpu, V_gpu, P_gpu,
            logit_scale=logit_scale,
            beta=beta,
            cache_alpha=cache_alpha,
            topk=topk
        )
        probs = F.softmax(logits.float(), dim=1)
        pred = probs.argmax(dim=1)
        pmax = probs.max(dim=1).values

        ytrue = batch.labels.tolist()
        ypred = pred.detach().cpu().tolist()
        pmax = pmax.detach().cpu().tolist()

        ys.extend(ytrue); ps.extend(ypred)
        for rel, yt, yp, pm in zip(batch.relpaths, ytrue, ypred, pmax):
            rows.append([rel, yt, yp, float(pm)])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "y_true", "y_pred", "p_max"])
        w.writerows(rows)

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return float(acc), float(f1m), ys, ps


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--ocr_cache", type=str, default="ocr_cache.json")
    ap.add_argument("--run_dir", type=str, default="runs/model3_tipadapter_clip_vitb16")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")

    # CLIP
    ap.add_argument("--clip_model", type=str, default="ViT-B-16")
    ap.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")

    # Fusion + Tip-Adapter hyperparams
    ap.add_argument("--fusion_alpha", type=float, default=0.7, help="weight for image in fusion")
    ap.add_argument("--empty_text_policy", type=str, default="image_only",
                    choices=["image_only", "zero", "keep"])

    ap.add_argument("--beta", type=float, default=20.0, help="sharpness for cache affinity")
    ap.add_argument("--cache_alpha", type=float, default=10.0, help="strength of cache logits")
    ap.add_argument("--topk", type=int, default=0, help="0=full cache; else topk neighbors")

    # Cache reuse
    ap.add_argument("--cache_path", type=str, default="", help="path to cache_train.pt (optional)")
    ap.add_argument("--rebuild_cache", action="store_true")

    # optional tuning on val
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--betas", type=str, default="5,10,20,30")
    ap.add_argument("--alphas", type=str, default="1,5,10,20")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ocr_cache = json.load(open(args.ocr_cache, "r", encoding="utf-8"))
    ocr_get = OCRLookup(ocr_cache)

    class2id, id2class = build_class_map(data_root / "train")
    num_classes = len(class2id)
    print("[Classes]", num_classes)

    json.dump(class2id, open(run_dir / "class2id.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    cfg = vars(args).copy(); cfg["num_classes"] = num_classes
    json.dump(cfg, open(run_dir / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()

    train_samples = list_split_samples(data_root, "train", class2id)
    val_samples = list_split_samples(data_root, "val", class2id)
    test_samples = list_split_samples(data_root, "test", class2id)
    print(f"[Samples] train={len(train_samples)} | val={len(val_samples)} | test={len(test_samples)}")

    use_amp = (device.type == "cuda") and (not args.no_amp)

    def make_loader(samples, shuffle=False):
        ds = AdsDataset(samples, preprocess, ocr_get)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=True if (args.persistent_workers and args.num_workers > 0) else False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            collate_fn=collate_simple,
        )

    train_loader = make_loader(train_samples, shuffle=False)
    val_loader = make_loader(val_samples, shuffle=False)
    test_loader = make_loader(test_samples, shuffle=False)

    # Cache path
    cache_path = Path(args.cache_path) if args.cache_path.strip() else (run_dir / "cache_train.pt")

    # Build/load cache/prototypes from train
    if cache_path.exists() and (not args.rebuild_cache):
        pack = torch.load(cache_path, map_location="cpu")
        K_cpu, V_cpu, P_cpu, y_cpu = pack["K"], pack["V"], pack["P"], pack["y"]
        cache_stats = pack.get("stats", {"train_cache_N": int(K_cpu.shape[0]), "feat_dim": int(K_cpu.shape[1])})
        print(f"[Cache] Loaded: {cache_path} | N={cache_stats.get('train_cache_N', K_cpu.shape[0])} | D={cache_stats.get('feat_dim', K_cpu.shape[1])}")
    else:
        K_cpu, V_cpu, P_cpu, y_cpu, cache_stats = build_cache_and_prototypes(
            clip_model, train_loader, device, use_amp,
            num_classes, args.fusion_alpha, args.empty_text_policy
        )
        torch.save({"K": K_cpu, "V": V_cpu, "P": P_cpu, "y": y_cpu, "stats": cache_stats}, cache_path)
        print(f"[Cache] Saved: {cache_path}")

    print(
        f"[CacheStats] N={cache_stats.get('train_cache_N', K_cpu.shape[0])} | "
        f"D={cache_stats.get('feat_dim', K_cpu.shape[1])} | "
        f"OCR_non_empty_rate={cache_stats.get('train_text_non_empty_rate', -1):.3f} | "
        f"avg_len={cache_stats.get('train_text_avg_len', -1):.1f}"
    )

    # Move to GPU
    K_gpu = K_cpu.to(device, dtype=torch.float16, non_blocking=True)
    V_gpu = V_cpu.to(device, dtype=torch.float16, non_blocking=True)
    P_gpu = P_cpu.to(device, dtype=torch.float16, non_blocking=True)

    # Optionally tune beta/alpha on VAL
    best_beta = args.beta
    best_alpha = args.cache_alpha
    best_val_f1 = -1.0

    if args.tune:
        betas = [float(x) for x in args.betas.split(",") if x.strip()]
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        print("[Tune] betas=", betas, "| alphas=", alphas)

        for b in betas:
            for a in alphas:
                vacc, vf1, _, _ = evaluate_split(
                    clip_model, val_loader, device, use_amp,
                    K_gpu, V_gpu, P_gpu,
                    fusion_alpha=args.fusion_alpha,
                    empty_text_policy=args.empty_text_policy,
                    beta=b, cache_alpha=a, topk=args.topk
                )
                print(f"  VAL beta={b} alpha={a} -> acc={vacc:.4f} f1={vf1:.4f}")
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    best_beta = b
                    best_alpha = a

        print(f"[Tune Best] beta={best_beta} | cache_alpha={best_alpha} | val_macro_f1={best_val_f1:.4f}")

    # Final eval (VAL + TEST)
    val_acc, val_f1, _, _ = evaluate_split(
        clip_model, val_loader, device, use_amp,
        K_gpu, V_gpu, P_gpu,
        fusion_alpha=args.fusion_alpha,
        empty_text_policy=args.empty_text_policy,
        beta=best_beta, cache_alpha=best_alpha, topk=args.topk
    )
    print(
        f"[VAL] acc={val_acc:.4f} | macro_f1={val_f1:.4f} | "
        f"beta={best_beta} | cache_alpha={best_alpha} | fusion_alpha={args.fusion_alpha} | "
        f"empty_text_policy={args.empty_text_policy} | topk={args.topk}"
    )

    # Save preds
    val_csv = run_dir / "preds_val.csv"
    _va, _vf, _ysv, _psv = predict_and_save_csv(
        clip_model, val_loader, device, use_amp,
        K_gpu, V_gpu, P_gpu,
        fusion_alpha=args.fusion_alpha,
        empty_text_policy=args.empty_text_policy,
        beta=best_beta, cache_alpha=best_alpha, topk=args.topk,
        out_csv=val_csv
    )

    test_csv = run_dir / "preds_test.csv"
    test_acc, test_f1, ys, ps = predict_and_save_csv(
        clip_model, test_loader, device, use_amp,
        K_gpu, V_gpu, P_gpu,
        fusion_alpha=args.fusion_alpha,
        empty_text_policy=args.empty_text_policy,
        beta=best_beta, cache_alpha=best_alpha, topk=args.topk,
        out_csv=test_csv
    )
    print(f"[TEST] acc={test_acc:.4f} | macro_f1={test_f1:.4f}")

    # Ablation
    tacc, tf1, _, _ = evaluate_split(
        clip_model, test_loader, device, use_amp,
        K_gpu, V_gpu, P_gpu,
        fusion_alpha=args.fusion_alpha,
        empty_text_policy=args.empty_text_policy,
        beta=best_beta, cache_alpha=best_alpha, topk=args.topk,
        ablate_text=True
    )
    iacc, if1, _, _ = evaluate_split(
        clip_model, test_loader, device, use_amp,
        K_gpu, V_gpu, P_gpu,
        fusion_alpha=args.fusion_alpha,
        empty_text_policy=args.empty_text_policy,
        beta=best_beta, cache_alpha=best_alpha, topk=args.topk,
        ablate_image=True
    )
    print(f"[TEST(text_off)] acc={tacc:.4f} | macro_f1={tf1:.4f}")
    print(f"[TEST(img_off)]  acc={iacc:.4f} | macro_f1={if1:.4f}")

    # Reports
    report_txt = classification_report(ys, ps, digits=4)
    with open(run_dir / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    metrics = {
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "best_beta": best_beta,
        "best_cache_alpha": best_alpha,
        "fusion_alpha": float(args.fusion_alpha),
        "empty_text_policy": args.empty_text_policy,
        "topk": int(args.topk),
        "test_text_off_macro_f1": float(tf1),
        "test_img_off_macro_f1": float(if1),
        "cache_stats": cache_stats,
        "cache_path": str(cache_path).replace("\\", "/"),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved folder: {run_dir}")
    print("[DONE] Model #3 Tip-Adapter finished.")


if __name__ == "__main__":
    main()
