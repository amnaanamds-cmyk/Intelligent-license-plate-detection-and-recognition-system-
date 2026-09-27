"""Train a vehicle-attribute classifier for the verification branch (Section 8.2).

The dataset is an image-classification folder with one subfolder per class,
for example body types:

    datasets/body_type/train/{sedan,suv,hatchback,pickup,van}/*.jpg
    datasets/body_type/val/{sedan,suv,hatchback,pickup,van}/*.jpg

    python scripts/train_attribute_classifier.py --data datasets/body_type \
        --export weights/body_type_cls.pt

A compact classification backbone (yolo11n-cls) keeps the added latency low,
as the paper requires. Use the same script for the make/model group
classifier with a different dataset folder.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--export", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    name = args.name or Path(args.data).name + "_cls"
    YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                           batch=args.batch, device=args.device,
                           project="runs/classify", name=name, exist_ok=True)
    best = Path("runs/classify") / name / "weights" / "best.pt"
    Path(args.export).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export)
    print(f"best weights copied to {args.export}")


if __name__ == "__main__":
    main()
