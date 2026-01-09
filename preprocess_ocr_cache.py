import argparse, json
from pathlib import Path
from tqdm import tqdm
import easyocr

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def list_images(split_dir: Path):
    """Return list of image paths under split_dir/<class_name>/*"""
    items = []
    if not split_dir.exists():
        return items
    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append(p)
    # stable order
    items.sort(key=lambda x: str(x).lower())
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Folder gốc chứa train/val/test (VD: data)."
    )
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    ap.add_argument("--langs", type=str, nargs="+", default=["vi", "en"])
    ap.add_argument("--gpu", action="store_true", help="Dùng GPU cho EasyOCR nếu có")
    ap.add_argument("--out", type=str, default="ocr_cache.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)

    # check structure
    for sp in args.splits:
        sp_dir = data_root / sp
        if not sp_dir.exists():
            print(f"[WARN] split not found: {sp_dir}")

    reader = easyocr.Reader(args.langs, gpu=args.gpu)

    cache = {}
    total = 0

    for sp in args.splits:
        sp_dir = data_root / sp
        imgs = list_images(sp_dir)
        total += len(imgs)

        for img_path in tqdm(imgs, desc=f"OCR {sp}"):
            # key dạng: "train/Class Name/xxx.jpg"
            rel = img_path.relative_to(data_root).as_posix()

            if rel in cache:
                continue

            try:
                # paragraph=True: gom chữ thành đoạn, detail=0: chỉ lấy text
                texts = reader.readtext(str(img_path), detail=0, paragraph=True)
                text = " ".join([t.strip() for t in texts if isinstance(t, str) and t.strip()])
            except Exception:
                text = ""

            cache[rel] = text

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved OCR cache: {args.out} | samples={len(cache)}/{total}")

if __name__ == "__main__":
    main()
