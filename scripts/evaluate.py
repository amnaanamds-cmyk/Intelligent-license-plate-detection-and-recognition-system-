"""Evaluate detection and recognition (Sections 6-7).

Detection: runs Ultralytics validation and reports precision, recall, F1 and
mAP@50 / mAP@50-95 on the chosen split.

    python scripts/evaluate.py detect --weights weights/plate_yolo11.pt --split test

Recognition: reads a CSV with the columns ``image,plate`` (the ground-truth
plate string for each image) and reports exact-match accuracy, character
accuracy and CER. With --ablation the run is repeated with and without the
enhancement pipeline, which measures the effect claimed in Section 7.

    python scripts/evaluate.py recognize --weights weights/plate_yolo11.pt \
        --labels datasets/plates/test_plates.csv --ablation
"""
import argparse
import csv
import json
from pathlib import Path

import _path  # noqa: F401


def eval_detection(args):
    from ultralytics import YOLO

    from lpr.metrics import f1_score

    metrics = YOLO(args.weights).val(data=args.data, split=args.split,
                                     imgsz=args.imgsz, device=args.device, plots=True)
    p, r = float(metrics.box.mp), float(metrics.box.mr)
    report = {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1_score(p, r), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    }
    print(json.dumps(report, indent=2))
    return report


def _read_labels(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = path.parent
    return [((base / r["image"]) if not Path(r["image"]).is_absolute() else Path(r["image"]),
             r["plate"]) for r in rows]


def eval_recognition(args):
    import cv2

    from lpr.metrics import recognition_report
    from lpr.pipeline import LicensePlatePipeline

    samples = _read_labels(Path(args.labels))
    pipe = LicensePlatePipeline.from_weights(args.weights, conf=args.conf,
                                             device=args.device)
    modes = [("enhanced", True), ("raw", False)] if args.ablation else [("enhanced", True)]
    reports, failures = {}, []
    for name, enhance in modes:
        pipe.use_enhancement = enhance
        preds, refs = [], []
        for img_path, ref in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"warning: cannot read {img_path}")
                continue
            readings = pipe.process(img)
            pred = readings[0].plate_text if readings else ""
            preds.append(pred)
            refs.append(ref)
            if name == "enhanced" and pred != "".join(c for c in ref.upper() if c.isalnum()):
                failures.append({"image": str(img_path), "reference": ref, "predicted": pred})
        reports[name] = recognition_report(preds, refs).as_dict()
    print(json.dumps(reports, indent=2))
    if args.failures:
        Path(args.failures).write_text(json.dumps(failures, indent=2))
        print(f"{len(failures)} failures written to {args.failures}")
    return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect")
    d.add_argument("--weights", required=True)
    d.add_argument("--data", default="configs/data.yaml")
    d.add_argument("--split", default="test", choices=["train", "val", "test"])
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--device", default=None)
    d.set_defaults(func=eval_detection)

    r = sub.add_parser("recognize")
    r.add_argument("--weights", required=True)
    r.add_argument("--labels", required=True, help="CSV with columns image,plate")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--device", default=None)
    r.add_argument("--ablation", action="store_true",
                   help="also evaluate without the enhancement pipeline")
    r.add_argument("--failures", default=None, help="write misread samples to this JSON file")
    r.set_defaults(func=eval_recognition)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
