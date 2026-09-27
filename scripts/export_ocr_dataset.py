"""Build a PaddleOCR recognition dataset from your labelled plates, to
fine-tune the recogniser on local fonts and layouts.

Input: a CSV with the columns ``image,plate`` (the same format as
``evaluate.py recognize``). Each image is run through the trained detector
and the enhancement pipeline, and the crops are written in PaddleOCR's
recognition format:

    <out>/images/xxx.png
    <out>/rec_gt_train.txt   ("images/xxx.png\\tLEB1234" per line)
    <out>/rec_gt_val.txt

    python scripts/export_ocr_dataset.py --weights weights/plate_yolo11.pt \
        --labels datasets/plates/all_plates.csv --out datasets/ocr_rec

See docs/DEPLOYMENT.md ("Fine-tune the OCR") for the training command.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2

import _path  # noqa: F401
from lpr.detector import YOLODetector, crop_box
from lpr.enhancement import PlateEnhancer
from lpr.geometry import deskew
from lpr.ocr import clean_plate_text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-enhance", action="store_true",
                    help="export raw crops (the pipeline enhances before OCR by default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    det = YOLODetector(args.weights, conf=args.conf)
    enh = PlateEnhancer()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels)
    with labels_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    samples, skipped = [], 0
    for i, row in enumerate(rows):
        p = Path(row["image"])
        img = cv2.imread(str(p if p.is_absolute() else labels_path.parent / p))
        text = clean_plate_text(row["plate"])
        dets = det.detect(img) if img is not None else []
        if not dets or not text:
            skipped += 1
            continue
        crop, _ = deskew(crop_box(img, dets[0].box, 0.05))
        out = crop if args.no_enhance else enh.enhance(crop)
        name = f"images/{i:06d}.png"
        cv2.imwrite(str(args.out / name), out)
        samples.append(f"{name}\t{text}")

    random.Random(args.seed).shuffle(samples)
    n_val = int(len(samples) * args.val)
    (args.out / "rec_gt_val.txt").write_text("\n".join(samples[:n_val]) + "\n")
    (args.out / "rec_gt_train.txt").write_text("\n".join(samples[n_val:]) + "\n")
    (args.out / "plate_dict.txt").write_text(
        "\n".join("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + "\n")
    print(f"{len(samples)} crops written ({n_val} val), {skipped} images skipped")


if __name__ == "__main__":
    main()
