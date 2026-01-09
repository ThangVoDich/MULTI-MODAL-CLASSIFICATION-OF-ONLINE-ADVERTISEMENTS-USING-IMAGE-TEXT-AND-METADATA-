
# streamlit_demo_app.py
# Demo app: compare 4 multimodal ad-classification models (CLIP + OCR-text)
# - Model1: Late fusion (CLIP image + XLM-R CLS -> concat -> MLP)
# - Model2: Gated fusion (learned gate between modalities)
# - Model3: Tip-Adapter (CLIP image+text fused -> cache/prototype logits)
# - Model4: Cross-attention (CLIP image queries XLM-R token states)

from __future__ import annotations
import json

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# -----------------------------
# Small utilities
# -----------------------------
def _safe_torch_load(path: str, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)

def _build_id2class(class2id: Dict[str, int]) -> Dict[int, str]:
    return {i: c for c, i in class2id.items()}

def _topk(probs: torch.Tensor, k: int = 3):
    k = min(k, probs.shape[-1])
    vals, idx = torch.topk(probs, k=k, dim=-1)
    return idx[0].tolist(), vals[0].tolist()

def _list_classes_from_train(data_root: str) -> List[str]:
    train_dir = Path(data_root) / "train"
    if not train_dir.exists():
        return []
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return classes

def _maybe_read_class2id_json(p: str) -> Optional[Dict[str,int]]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # accept {"class2id": {...}} or direct dict
        if isinstance(obj, dict) and "class2id" in obj and isinstance(obj["class2id"], dict):
            obj = obj["class2id"]
        if isinstance(obj, dict) and all(isinstance(k,str) and isinstance(v,int) for k,v in obj.items()):
            return obj
    except Exception:
        return None
    return None


# -----------------------------
# OCR (optional)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _get_easyocr_reader():
    try:
        import easyocr  # type: ignore
        return easyocr.Reader(["en", "vi"], gpu=torch.cuda.is_available())
    except Exception:
        return None

def ocr_text_from_image(img: Image.Image, mode: str, manual_text: str, reader, cache: Dict[str,str], cache_key: str) -> str:
    if mode == "Manual":
        return manual_text.strip()

    if mode == "From OCR cache JSON":
        return (cache.get(cache_key) or "").strip()

    # Auto OCR
    if reader is None:
        return manual_text.strip()  # fallback
    try:
        import numpy as _np
        arr = _np.array(img.convert("RGB"))
        results = reader.readtext(arr, detail=0, paragraph=True)
        if isinstance(results, list):
            txt = " ".join([str(x) for x in results]).strip()
            return txt
        return str(results).strip()
    except Exception:
        return manual_text.strip()


# -----------------------------
# Model #1 (Late fusion)
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

def get_clip_embed_dim(clip_model) -> int:
    for attr in ["embed_dim", "output_dim"]:
        if hasattr(clip_model, attr):
            v = getattr(clip_model, attr)
            if isinstance(v, int):
                return v
    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "output_dim"):
        return int(clip_model.visual.output_dim)
    return 512

def encode_batch_late(clip_model, text_model, images, input_ids, attention_mask):
    img_feat = clip_model.encode_image(images)
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    txt_feat = out.last_hidden_state[:, 0, :]
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
    return img_feat, txt_feat


# -----------------------------
# Model #2 (Gated fusion)
# -----------------------------
class GatedFusionClassifier(nn.Module):
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

        g_logits = self.gate_fc(torch.cat([img_p, txt_p], dim=-1))
        g = torch.sigmoid(g_logits / self.gate_temp)
        g = self.gate_min + (1.0 - 2.0 * self.gate_min) * g

        fused = g * img_p + (1.0 - g) * txt_p
        logits = self.head(fused)
        return logits, g


# -----------------------------
# Model #4 (Cross-attention)
# -----------------------------
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
        q = self.q(img_feat).unsqueeze(1)  # [B,1,H]
        k = self.k(txt_hid)               # [B,S,H]
        v = self.v(txt_hid)               # [B,S,H]
        key_padding_mask = (attn_mask == 0)
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln(q + attn_out)
        x2 = self.ff(x)
        x = self.ln2(x + x2)
        return x.squeeze(1)

class CrossAttnModel(nn.Module):
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

@torch.inference_mode()
def encode_text_hidden(text_model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state

@torch.inference_mode()
def encode_image_only(clip_model, images: torch.Tensor) -> torch.Tensor:
    img_feat = clip_model.encode_image(images)
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
    return img_feat


# -----------------------------
# Model #3 (Tip-Adapter)
# -----------------------------
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
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and (device.type == "cuda")):
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
                txt_feat = torch.zeros_like(img_feat)
                if has_text.any():
                    idx = has_text.nonzero(as_tuple=True)[0]
                    texts_non_empty = [texts[i] for i in idx.tolist()]
                    tokens = open_clip.tokenize(texts_non_empty).to(device)
                    tfeat = clip_model.encode_text(tokens)
                    tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-8)
                    txt_feat[idx] = tfeat

        if ablate_image:
            img_feat = torch.zeros_like(img_feat)

        fused = fusion_alpha * img_feat + (1.0 - fusion_alpha) * txt_feat
        if (not ablate_text) and (empty_text_policy == "image_only"):
            fused = torch.where(has_text.unsqueeze(1), fused, img_feat)

        fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)
    return fused

@torch.inference_mode()
def tipadapter_logits(
    q: torch.Tensor,         # [B,D]
    K: torch.Tensor,         # [N,D]
    V: torch.Tensor,         # [N,C]
    P: torch.Tensor,         # [C,D]
    logit_scale: float,
    beta: float,
    cache_alpha: float,
    topk: int = 0,
    exp_clip: float = 50.0,
):
    """
    Tip-Adapter style logits:
      logits = (q @ P^T) * logit_scale  +  cache_alpha * exp(beta * (q @ K^T)) @ V

    Important: all matmul operands must share dtype on GPU.
    This function normalizes/casts tensors to avoid Half vs Float mismatches.
    """
    target_dtype = torch.float16 if q.is_cuda else torch.float32

    q = q.to(target_dtype)
    K = K.to(target_dtype)
    P = P.to(target_dtype)
    V = V.to(target_dtype)

    # prototype (text-encoder) branch
    proto = (q @ P.t()).float() * float(logit_scale)  # [B,C]

    # cache (training-set) branch
    sim = (q @ K.t()).float()                         # [B,N]

    if topk and 0 < topk < K.shape[0]:
        # keep top-k cache keys per sample
        vals, idx = torch.topk(sim, k=topk, dim=-1)   # [B,k]
        sim = vals

        # gather V for each sample: [B,k,C]
        V_exp = V.unsqueeze(0).expand(idx.size(0), -1, -1)  # [B,N,C]
        V_sel = V_exp.gather(1, idx.unsqueeze(-1).expand(-1, -1, V.size(1)))  # [B,k,C]

        x = (float(beta) * sim).clamp(min=-float(exp_clip), max=float(exp_clip))
        aff = torch.exp(x)                                  # [B,k]
        cache = torch.bmm(aff.unsqueeze(1), V_sel.float()).squeeze(1)  # [B,C]
    else:
        x = (float(beta) * sim).clamp(min=-float(exp_clip), max=float(exp_clip))
        aff = torch.exp(x)                                  # [B,N]
        cache = aff @ V.float()                             # [B,C]

    return proto + float(cache_alpha) * cache



# -----------------------------
# Pipelines
# -----------------------------
@dataclass
class PredictResult:
    top1: str
    top1_prob: float
    topk: List[Tuple[str, float]]
    extra: Dict[str, float]

class BasePipeline:
    name: str
    def predict(self, img: Image.Image, text: str) -> PredictResult:
        raise NotImplementedError

class Pipeline1(BasePipeline):
    def __init__(self, ckpt_path: str, device: torch.device):
        ckpt = _safe_torch_load(ckpt_path, device)
        class2id = ckpt.get("class2id") or {}
        if not class2id:
            raise ValueError("Model1 checkpoint missing class2id.")
        self.class2id = class2id
        self.id2class = _build_id2class(class2id)
        args = ckpt.get("args") or {}
        clip_model_name = args.get("clip_model", "ViT-B-16")
        clip_pretrained = args.get("clip_pretrained", "laion2b_s34b_b88k")
        text_model_name = args.get("text_model", "xlm-roberta-base")
        max_len = int(args.get("max_len", 64))

        clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_model = AutoModel.from_pretrained(text_model_name)

        clip_model = clip_model.to(device)
        text_model = text_model.to(device)

        img_dim = get_clip_embed_dim(clip_model)
        txt_dim = int(text_model.config.hidden_size)
        fusion = LateFusionMLP(img_dim, txt_dim, num_classes=len(class2id), hidden=512, dropout=0.2).to(device)

        fusion.load_state_dict(ckpt["fusion"], strict=True)
        clip_model.load_state_dict(ckpt["clip"], strict=False)
        text_model.load_state_dict(ckpt["text"], strict=False)

        self.name = "Model 1 (LateFusion)"
        self.device = device
        self.clip_model = clip_model.eval()
        self.text_model = text_model.eval()
        self.fusion = fusion.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_len = max_len

    @torch.inference_mode()
    def predict(self, img: Image.Image, text: str) -> PredictResult:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        tok = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        img_feat, txt_feat = encode_batch_late(self.clip_model, self.text_model, x, input_ids, attention_mask)
        logits = self.fusion(img_feat, txt_feat)
        probs = F.softmax(logits.float(), dim=-1)
        idxs, vals = _topk(probs, k=3)
        topk = [(self.id2class[i], float(v)) for i, v in zip(idxs, vals)]
        return PredictResult(
            top1=topk[0][0],
            top1_prob=topk[0][1],
            topk=topk,
            extra={}
        )

class Pipeline2(BasePipeline):
    def __init__(self, ckpt_path: str, device: torch.device):
        ckpt = _safe_torch_load(ckpt_path, device)
        class2id = ckpt.get("class2id") or {}
        if not class2id:
            raise ValueError("Model2 checkpoint missing class2id.")
        self.class2id = class2id
        self.id2class = _build_id2class(class2id)

        # Model2 script stores clip/text model names in args (if missing, use defaults)
        # We still accept defaults here.
        args = ckpt.get("args") or {}
        clip_model_name = args.get("clip_model", "ViT-B-16")
        clip_pretrained = args.get("clip_pretrained", "laion2b_s34b_b88k")
        text_model_name = args.get("text_model", "xlm-roberta-base")
        max_len = int(args.get("max_len", 64))

        clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_model = AutoModel.from_pretrained(text_model_name)

        clip_model = clip_model.to(device)
        text_model = text_model.to(device)

        img_dim = get_clip_embed_dim(clip_model)
        txt_dim = int(text_model.config.hidden_size)
        fusion = GatedFusionClassifier(img_dim, txt_dim, num_classes=len(class2id), d_model=512, hidden=512, dropout=0.2).to(device)

        fusion.load_state_dict(ckpt["fusion"], strict=True)
        clip_model.load_state_dict(ckpt["clip"], strict=False)
        text_model.load_state_dict(ckpt["text"], strict=False)

        self.name = "Model 2 (GatedFusion)"
        self.device = device
        self.clip_model = clip_model.eval()
        self.text_model = text_model.eval()
        self.fusion = fusion.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_len = max_len

    @torch.inference_mode()
    def predict(self, img: Image.Image, text: str) -> PredictResult:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        tok = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        img_feat, txt_feat = encode_batch_late(self.clip_model, self.text_model, x, input_ids, attention_mask)
        logits, g = self.fusion(img_feat, txt_feat)
        probs = F.softmax(logits.float(), dim=-1)
        idxs, vals = _topk(probs, k=3)
        topk = [(self.id2class[i], float(v)) for i, v in zip(idxs, vals)]
        gate_mean = float(g.mean().item())
        return PredictResult(
            top1=topk[0][0],
            top1_prob=topk[0][1],
            topk=topk,
            extra={"gate_mean": gate_mean}
        )

class Pipeline4(BasePipeline):
    def __init__(self, ckpt_path: str, device: torch.device):
        ckpt = _safe_torch_load(ckpt_path, device)
        class2id = ckpt.get("class2id") or {}
        if not class2id:
            raise ValueError("Model4 checkpoint missing class2id.")
        self.class2id = class2id
        self.id2class = _build_id2class(class2id)
        args = ckpt.get("args") or {}
        max_len = int(args.get("max_len", 64))
        hidden = int(args.get("hidden", 512))
        nhead = int(args.get("nhead", 8))
        dropout = float(args.get("dropout", 0.1))

        # Model4 training script uses fixed names:
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        text_model = AutoModel.from_pretrained("xlm-roberta-base")

        clip_model = clip_model.to(device)
        text_model = text_model.to(device)

        img_dim = get_clip_embed_dim(clip_model)
        txt_dim = int(text_model.config.hidden_size)
        model = CrossAttnModel(img_dim, txt_dim, num_classes=len(class2id), hidden=hidden, nhead=nhead, dropout=dropout).to(device)

        model.load_state_dict(ckpt["model"], strict=True)
        clip_model.load_state_dict(ckpt["clip"], strict=False)
        text_model.load_state_dict(ckpt["text"], strict=False)

        self.name = "Model 4 (CrossAttn)"
        self.device = device
        self.clip_model = clip_model.eval()
        self.text_model = text_model.eval()
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_len = max_len

    @torch.inference_mode()
    def predict(self, img: Image.Image, text: str) -> PredictResult:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        tok = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        img_feat = encode_image_only(self.clip_model, x)
        txt_hid = encode_text_hidden(self.text_model, input_ids, attention_mask)
        logits = self.model(img_feat, txt_hid, attention_mask)
        probs = F.softmax(logits.float(), dim=-1)
        idxs, vals = _topk(probs, k=3)
        topk = [(self.id2class[i], float(v)) for i, v in zip(idxs, vals)]
        return PredictResult(
            top1=topk[0][0],
            top1_prob=topk[0][1],
            topk=topk,
            extra={}
        )

class Pipeline3(BasePipeline):
    def __init__(self, data_root: str, cache_path: str, device: torch.device,
                 fusion_alpha: float = 0.7, beta: float = 20.0, cache_alpha: float = 10.0, topk: int = 0,
                 empty_text_policy: str = "image_only"):
        classes = _list_classes_from_train(data_root)
        if not classes:
            raise ValueError("Model3 needs data_root with train/CLASS_NAME folders to build class list.")
        self.class2id = {c:i for i,c in enumerate(classes)}
        self.id2class = _build_id2class(self.class2id)
        self.num_classes = len(classes)

        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        clip_model = clip_model.to(device).eval()

        cache = _safe_torch_load(cache_path, device)
        K = cache["K"].to(device)
        V = cache["V"].to(device)
        P = cache["P"].to(device)

        self.name = "Model 3 (Tip-Adapter)"
        self.device = device
        self.clip_model = clip_model
        self.preprocess = preprocess

        self.K = K
        self.V = V
        self.P = P

        self.fusion_alpha = float(fusion_alpha)
        self.beta = float(beta)
        self.cache_alpha = float(cache_alpha)
        self.topk = int(topk)
        self.empty_text_policy = empty_text_policy

    @torch.inference_mode()
    def predict(self, img: Image.Image, text: str) -> PredictResult:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        q = encode_fused_features(
            self.clip_model, x, [text], self.device,
            use_amp=True, fusion_alpha=self.fusion_alpha,
            empty_text_policy=self.empty_text_policy
        )
        logit_scale = float(self.clip_model.logit_scale.exp().item())
        logits = tipadapter_logits(
            q=q, K=self.K, V=self.V, P=self.P,
            logit_scale=logit_scale,
            beta=self.beta,
            cache_alpha=self.cache_alpha,
            topk=self.topk,
        )
        probs = F.softmax(logits.float(), dim=-1)
        idxs, vals = _topk(probs, k=3)
        topk = [(self.id2class[i], float(v)) for i, v in zip(idxs, vals)]
        return PredictResult(
            top1=topk[0][0],
            top1_prob=topk[0][1],
            topk=topk,
            extra={}
        )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Ad Classifier Demo (4 Models)", layout="wide")

st.title("Demo: Multimodal Ad Classification (4 models)")
st.caption("Upload an ad image → (OCR text) → compare predictions across Model 1/2/3/4.")

with st.sidebar:
    st.header("Runtime")
    device_opt = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    if device_opt == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA not available. Falling back to CPU.")
    if device_opt == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_opt if (device_opt != "cuda" or torch.cuda.is_available()) else "cpu")
    st.write(f"Using: `{device}`")

    st.divider()
    st.header("Paths")
    colA, colB = st.columns(2)
    with colA:
        ckpt1 = st.text_input("Model1 best.pt", value="runs/model1_latefusion/best.pt")
        ckpt2 = st.text_input("Model2 best.pt", value="runs/model2_gatedfusion/best.pt")
    with colB:
        ckpt4 = st.text_input("Model4 best.pt", value="runs/model4_crossattn/best.pt")
        data_root = st.text_input("Dataset root (for Model3 classes)", value="data")
    cache3 = st.text_input("Model3 cache_train.pt", value="runs/model3_tipadapter/cache_train.pt")

    enable1 = st.checkbox("Enable Model 1", value=True)
    enable2 = st.checkbox("Enable Model 2", value=True)
    enable3 = st.checkbox("Enable Model 3", value=True)
    enable4 = st.checkbox("Enable Model 4", value=True)

    st.divider()
    st.header("OCR / Text input")
    ocr_mode = st.selectbox("Text source", ["Auto OCR (easyocr if available)", "From OCR cache JSON", "Manual"], index=0)
    ocr_cache_path = st.text_input("OCR cache json (optional)", value="ocr_cache.json")
    manual_text = st.text_area("Manual text (fallback)", value="", height=120)

    st.divider()
    st.header("Tip-Adapter params (Model3)")
    fusion_alpha = st.slider("fusion_alpha (image weight)", 0.0, 1.0, 0.7, 0.05)
    beta = st.slider("beta (sharpness)", 1.0, 50.0, 20.0, 1.0)
    cache_alpha = st.slider("cache_alpha (strength)", 0.0, 50.0, 10.0, 1.0)
    topk = st.number_input("topk neighbors (0 = full cache)", min_value=0, max_value=5000, value=0, step=10)
    empty_text_policy = st.selectbox("empty_text_policy", ["image_only", "zero", "keep"], index=0)


@st.cache_resource(show_spinner=True)
def load_pipeline_1(path: str, device: torch.device):
    return Pipeline1(path, device)

@st.cache_resource(show_spinner=True)
def load_pipeline_2(path: str, device: torch.device):
    return Pipeline2(path, device)

@st.cache_resource(show_spinner=True)
def load_pipeline_4(path: str, device: torch.device):
    return Pipeline4(path, device)

@st.cache_resource(show_spinner=True)
def load_pipeline_3(data_root: str, cache_path: str, device: torch.device,
                    fusion_alpha: float, beta: float, cache_alpha: float, topk: int, empty_text_policy: str):
    return Pipeline3(data_root=data_root, cache_path=cache_path, device=device,
                     fusion_alpha=fusion_alpha, beta=beta, cache_alpha=cache_alpha, topk=topk,
                     empty_text_policy=empty_text_policy)


def _load_ocr_cache(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            # accept {"relpath": "text"} format
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}

reader = _get_easyocr_reader()
ocr_cache = _load_ocr_cache(ocr_cache_path) if ocr_mode == "From OCR cache JSON" else {}

tab1, tab2 = st.tabs(["Single image", "Batch evaluate (folder)"])

with tab1:
    st.subheader("Single image inference")
    up = st.file_uploader("Upload an ad image", type=["png","jpg","jpeg","webp"])
    if up is None:
        st.info("Upload an image to start.")
    else:
        img = Image.open(up).convert("RGB")
        st.image(img, caption=f"{up.name}", use_container_width=True)

        # cache key: try file name first (common in ocr_cache json)
        cache_key = up.name
        text = ocr_text_from_image(img, ocr_mode, manual_text, reader, ocr_cache, cache_key)
        st.markdown("**Extracted / provided text:**")
        st.code(text if text else "(empty)")

        run = st.button("Run all enabled models")
        if run:
            pipelines: List[BasePipeline] = []
            errors = []

            if enable1:
                try:
                    if not Path(ckpt1).exists():
                        raise FileNotFoundError(f"Missing: {ckpt1}")
                    pipelines.append(load_pipeline_1(ckpt1, device))
                except Exception as e:
                    errors.append(f"Model1: {e}")

            if enable2:
                try:
                    if not Path(ckpt2).exists():
                        raise FileNotFoundError(f"Missing: {ckpt2}")
                    pipelines.append(load_pipeline_2(ckpt2, device))
                except Exception as e:
                    errors.append(f"Model2: {e}")

            if enable3:
                try:
                    if not Path(cache3).exists():
                        raise FileNotFoundError(f"Missing: {cache3}")
                    pipelines.append(load_pipeline_3(data_root, cache3, device, fusion_alpha, beta, cache_alpha, int(topk), empty_text_policy))
                except Exception as e:
                    errors.append(f"Model3: {e}")

            if enable4:
                try:
                    if not Path(ckpt4).exists():
                        raise FileNotFoundError(f"Missing: {ckpt4}")
                    pipelines.append(load_pipeline_4(ckpt4, device))
                except Exception as e:
                    errors.append(f"Model4: {e}")

            if errors:
                st.warning("Some models failed to load:\n- " + "\n- ".join(errors))

            if not pipelines:
                st.error("No model loaded. Check paths in the sidebar.")
            else:
                cols = st.columns(len(pipelines))
                for col, pipe in zip(cols, pipelines):
                    with col:
                        st.markdown(f"### {pipe.name}")
                        try:
                            res = pipe.predict(img, text)
                            st.metric("Top-1", res.top1, delta=None)
                            st.write("Top-3:")
                            for c, p in res.topk:
                                st.write(f"- `{c}`")
                            if res.extra:
                                for k, v in res.extra.items():
                                    st.write(f"**{k}**: `{v:.3f}`")
                        except Exception as e:
                            st.error(f"Inference error: {e}")

with tab2:
    st.subheader("Batch evaluate (folder path)")
    st.caption("Give a folder that contains class subfolders (e.g., data/test/CLASS/*.jpg). If no subfolders, it will run unlabeled inference only.")

    # Persist batch results across reruns (Streamlit reruns script on any widget change)
    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = None

    folder = st.text_input("Folder path", value="data/test")
    max_items = st.number_input("Max images", min_value=1, max_value=5000, value=200, step=50)

    model_choices = st.multiselect("Models to evaluate", ["Model1", "Model2", "Model3", "Model4"], default=["Model1","Model2","Model3","Model4"])
    run_batch = st.button("Run batch")

    def _collect_images(folder_path: str, limit: int):
        folder_p = Path(folder_path)
        if not folder_p.exists():
            return [], False
        # labeled if has subfolders
        subdirs = [d for d in folder_p.iterdir() if d.is_dir()]
        labeled = len(subdirs) > 0
        items = []
        if labeled:
            for cls_dir in sorted(subdirs):
                for img_p in cls_dir.rglob("*"):
                    if img_p.suffix.lower() in [".jpg",".jpeg",".png",".webp"]:
                        items.append((img_p, cls_dir.name))
        else:
            for img_p in folder_p.rglob("*"):
                if img_p.suffix.lower() in [".jpg",".jpeg",".png",".webp"]:
                    items.append((img_p, None))
        items = items[:limit]
        return items, labeled


    if run_batch:
        items, labeled = _collect_images(folder, int(max_items))
        if not items:
            st.error("No images found.")
            st.session_state["batch_results"] = None
        else:
            # -------- Load selected models --------
            if not model_choices:
                st.warning("Pick at least one model.")
                st.stop()

            def _load_pipe(name: str):
                if name == "Model1":
                    return load_pipeline_1(ckpt1, device)
                if name == "Model2":
                    return load_pipeline_2(ckpt2, device)
                if name == "Model3":
                    return load_pipeline_3(data_root, cache3, device, fusion_alpha, beta, cache_alpha, int(topk), empty_text_policy)
                if name == "Model4":
                    return load_pipeline_4(ckpt4, device)
                raise ValueError("Unknown model")

            pipes = {}
            load_errors = []
            for nm in model_choices:
                try:
                    pipes[nm] = _load_pipe(nm)
                except Exception as e:
                    load_errors.append(f"{nm}: {e}")

            if load_errors:
                st.warning("Some models failed to load:\n- " + "\n- ".join(load_errors))

            if not pipes:
                st.error("No model loaded. Check paths in the sidebar.")
                st.session_state["batch_results"] = None
                st.stop()

            # -------- OCR cache (optional) --------
            batch_cache = _load_ocr_cache(ocr_cache_path) if ocr_mode == "From OCR cache JSON" else {}

            prog = st.progress(0)
            rows = []
            y_true = []
            y_pred_map = {nm: [] for nm in pipes.keys()}

            for i, (img_path, true_cls) in enumerate(items, 1):
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                key = str(img_path.as_posix())
                txt = ""
                if ocr_mode == "From OCR cache JSON":
                    txt = (batch_cache.get(key) or batch_cache.get(img_path.name) or "").strip()
                elif ocr_mode.startswith("Auto"):
                    txt = ocr_text_from_image(
                        img,
                        "Auto OCR (easyocr if available)",
                        manual_text="",
                        reader=reader,
                        cache={},
                        cache_key="",
                    )
                else:
                    txt = ""

                row = {"path": str(img_path), "text_len": len(txt)}
                for nm, pipe in pipes.items():
                    res = pipe.predict(img, txt)
                    row[f"pred_{nm}"] = res.top1
                    row[f"top2_{nm}"] = res.topk[1][0] if len(res.topk) > 1 else ""
                    row[f"top3_{nm}"] = res.topk[2][0] if len(res.topk) > 2 else ""
                    for k, v in (res.extra or {}).items():
                        row[f"{k}_{nm}"] = v
                    if labeled and true_cls is not None:
                        y_pred_map[nm].append(res.top1)

                if labeled and true_cls is not None:
                    y_true.append(true_cls)

                rows.append(row)
                prog.progress(i / len(items))

            st.session_state["batch_results"] = {
                "rows": rows,
                "labeled": labeled,
                "y_true": y_true,
                "y_pred_map": y_pred_map,
                "models": list(pipes.keys()),
            }

    # -------------------------
    # Display (persisted)
    # -------------------------
    br = st.session_state.get("batch_results")
    if br is None:
        st.info("Run batch to see results.")
    else:
        st.success(f"Done. Processed {len(br['rows'])} images.")
        st.dataframe(br["rows"], use_container_width=True)

        labeled = bool(br.get("labeled", False))
        y_true = br.get("y_true", [])
        y_pred_map = br.get("y_pred_map", {})
        models = br.get("models", [])

        if labeled and len(y_true) > 0:
            import pandas as pd

            summary_rows = []
            for nm in models:
                preds = y_pred_map.get(nm, [])
                if len(preds) != len(y_true):
                    continue
                acc = accuracy_score(y_true, preds)
                f1m = f1_score(y_true, preds, average="macro", zero_division=0)
                correct = sum(1 for t, p in zip(y_true, preds) if t == p)
                total = len(y_true)
                summary_rows.append({
                    "Model": nm,
                    "Accuracy": acc,
                    "Macro-F1": f1m,
                    "Correct": correct,
                    "Total": total,
                    "Correct rate (%)": 100.0 * correct / max(total, 1),
                    "Correct/Total": f"{correct}/{total}",
                })

            if not summary_rows:
                st.warning("No labeled samples found (need folder structure test/CLASS_NAME/*.jpg).")
                st.stop()

            df_sum = pd.DataFrame(summary_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
            df_show = df_sum.copy()
            df_show["Accuracy"] = df_show["Accuracy"].map(lambda x: f"{x:.4f}")
            df_show["Macro-F1"] = df_show["Macro-F1"].map(lambda x: f"{x:.4f}")
            df_show["Correct rate (%)"] = df_show["Correct rate (%)"].map(lambda x: f"{x:.2f}%")

            st.subheader("Model comparison")
            st.dataframe(
                df_show[["Model", "Accuracy", "Macro-F1", "Correct/Total", "Correct rate (%)"]],
                use_container_width=True,
            )

            # persist chosen model via key so it won't "reset weirdly"
            pick = st.selectbox("Show details for", options=df_sum["Model"].tolist(), index=0, key="batch_detail_model")
            y_pred = y_pred_map[pick]

            acc = accuracy_score(y_true, y_pred)
            f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            total = len(y_true)
            acc_pct = 100.0 * correct / max(total, 1)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Macro-F1", f"{f1m:.4f}")
            c3.metric("Correct / Total", f"{correct} / {total}")
            c4.metric("Correct rate", f"{acc_pct:.2f}%")

            # Per-class accuracy
            from collections import defaultdict, Counter
            cls_total = defaultdict(int)
            cls_correct = defaultdict(int)
            for t, p in zip(y_true, y_pred):
                cls_total[t] += 1
                if t == p:
                    cls_correct[t] += 1

            per_rows = []
            for c in sorted(cls_total.keys()):
                tot = cls_total[c]
                cor = cls_correct[c]
                per_rows.append({"Class": c, "Correct": cor, "Total": tot, "Accuracy": cor / max(tot, 1)})

            df_per = pd.DataFrame(per_rows).sort_values("Accuracy", ascending=True).reset_index(drop=True)
            df_per_show = df_per.copy()
            df_per_show["Accuracy"] = df_per_show["Accuracy"].map(lambda x: f"{x*100:.2f}%")
            st.write(f"Per-class accuracy — {pick} (sorted worst → best)")
            st.dataframe(df_per_show, use_container_width=True)

            # Confusion matrix + report (top classes only)
            topN = 12
            common = [c for c, _ in Counter(y_true).most_common(topN)]
            labels = common
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            st.write(f"Confusion matrix (top {topN} by true frequency) — {pick}")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

            st.write("Classification report (top classes only)")
            st.code(classification_report(y_true, y_pred, labels=labels, zero_division=0))
