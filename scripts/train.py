"""Train the YOLOv11 license-plate detector (Section 5.2).

Example:
    python scripts/train.py --data configs/data.yaml --model yolo11s.pt --epochs 100

The Ultralytics defaults already include the augmentations described in the
paper (mosaic, HSV colour jitter, scale jitter). They are set explicitly here
so they are easy to find and change.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11{n,s,m,l,x}.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience")
    ap.add_argument("--device", default=None, help="'0', '0,1' or 'cpu'")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="plate_yolo11")
    ap.add_argument("--export", default="weights/plate_yolo11.pt",
                    help="where to copy best.pt after training")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=args.device,
        project=args.project, name=args.name, exist_ok=True,
        # augmentation (Section 5.2)
        mosaic=1.0, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, perspective=0.0005, fliplr=0.0,  # flipping mirrors the characters
        close_mosaic=10, plots=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists() and args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.export)
        print(f"best weights copied to {args.export}")


if __name__ == "__main__":
    main()
