"""Split a YOLO-annotated dataset into train/val/test folders (Section 5.1).

Input: a folder with images and one YOLO .txt label per image, either side by
side or in images/ and labels/ subfolders. Output layout:

    <dst>/images/{train,val,test}/  and  <dst>/labels/{train,val,test}/

Example:
    python scripts/split_dataset.py --src raw_plates --dst datasets/plates --val 0.15 --test 0.15
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(src: Path):
    img_dir = src / "images" if (src / "images").is_dir() else src
    lbl_dir = src / "labels" if (src / "labels").is_dir() else img_dir
    pairs, missing = [], 0
    for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
        lbl = lbl_dir / img.relative_to(img_dir).with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing += 1
    return pairs, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs, missing = find_pairs(args.src)
    if not pairs:
        raise SystemExit(f"no image/label pairs found in {args.src}")
    random.Random(args.seed).shuffle(pairs)
    n = len(pairs)
    n_test, n_val = int(n * args.test), int(n * args.val)
    splits = {"test": pairs[:n_test], "val": pairs[n_test:n_test + n_val],
              "train": pairs[n_test + n_val:]}
    for split, items in splits.items():
        (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, args.dst / "images" / split / img.name)
            shutil.copy2(lbl, args.dst / "labels" / split / (img.stem + ".txt"))
    print(f"{n} pairs -> " + ", ".join(f"{k}: {len(v)}" for k, v in splits.items()))
    if missing:
        print(f"warning: skipped {missing} images without a label file")


if __name__ == "__main__":
    main()
