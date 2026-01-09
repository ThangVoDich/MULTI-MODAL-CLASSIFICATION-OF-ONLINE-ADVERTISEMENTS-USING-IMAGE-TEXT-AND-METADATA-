============================================================
# README – MULTI-MODAL CLASSIFICATION OF ONLINE ADVERTISEMENTS
USING IMAGE, TEXT, AND METADATA
============================================================

Authors:
- Nguyen Bao Thang 

Affiliation:
Faculty of Information Technology,
Ton Duc Thang University, Ho Chi Minh City, Vietnam

------------------------------------------------------------

## 1) Project Description
This project implements a multimodal classification system for online advertisement images.
The system integrates multiple information sources, including:
- Image content (visual layout, objects, colors)
- Text embedded in images extracted via OCR
- Metadata-level semantic cues learned implicitly through multimodal fusion

The goal is to predict a single advertisement category from a predefined label set.

Four multimodal architectures are implemented and compared:
- Model 1: Late Fusion (CLIP image + XLM-R text concatenation)
- Model 2: Gated Fusion (adaptive modality weighting)
- Model 3: Tip-Adapter (cache-based CLIP adaptation)
- Model 4: Cross-Attention Fusion (image-to-text token interaction)

## 2) System Requirements
- Python 3.9 or higher (recommended: Python 3.10)
- Windows / macOS / Linux
- GPU is optional (CPU execution is supported)

## 3) Dataset Structure
data/train/<class_name>/*.jpg|png
data/val/<class_name>/*.jpg|png
data/test/<class_name>/*.jpg|png

## 4) Environment Setup (one command per step)

### 4.1 Create virtual environment:
python -m venv .venv

### 4.2 Activate virtual environment (Windows PowerShell):
.\.venv\Scripts\activate

### 4.3 Install required dependencies:
Install PyTorch

If you have an NVIDIA GPU (for example, CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

If you are using CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Install the remaining dependencies (for all code):
pip install open-clip-torch transformers sentencepiece scikit-learn tqdm pillow numpy streamlit easyocr opencv-python pandas


## 5) OCR Preprocessing (OCR Cache Generation)
python preprocess_ocr_cache.py --data_root data --out ocr_cache.json

(Optional – enable GPU for OCR):
python preprocess_ocr_cache.py --data_root data --out ocr_cache.json --gpu

## 6) Model Training (single-line commands)

### 6.1 Model 1 – Late Fusion:
python train_model1_clip_xlmr_latefusion.py --data_root data --ocr_cache ocr_cache.json --out_dir runs/model1_latefusion --epochs 12 --batch_size 128 --lr 1e-4 --wd 1e-4 --max_len 64

### 6.2 Model 2 – Gated Fusion:
python train_model2_clip_xlmr_gatedfusion.py --data_root data --ocr_cache ocr_cache.json --run_dir runs/model2_gatedfusion --epochs 12 --batch_size 128 --lr 1e-4 --wd 1e-4 --max_len 64 --eval_workers 0

### 6.3 Model 3 – Tip-Adapter:
python train_model3_tipadapter_clip_vitb16.py --data_root data --ocr_cache ocr_cache.json --run_dir runs/model3_tipadapter --fusion_alpha 0.7 --beta 20 --cache_alpha 10 --topk 0

### 6.4 Model 4 – Cross-Attention Fusion:
python train_model4_crossattn_vitb16_xlmr.py --data_root data --ocr_cache ocr_cache.json --run_dir runs/model4_crossattn --epochs 12 --batch_size 64 --lr 1e-4 --wd 1e-4 --max_len 64 --num_workers 0

## 7) Running the Streamlit Demonstration
streamlit run appfinal.py

## 8) Output Files
- runs/model1_latefusion/best.pt
- runs/model2_gatedfusion/best.pt
- runs/model3_tipadapter/cache_train.pt
- runs/model4_crossattn/best.pt

## 9) Notes
- All commands are written in single-line format.
- OCR cache must be regenerated if dataset paths change.
- CPU-only execution is supported.


