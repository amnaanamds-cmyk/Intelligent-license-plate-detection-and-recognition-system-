"""Run the full pipeline on images, a folder, or a video file / webcam.

    python scripts/infer.py --weights weights/plate_yolo11.pt --source car.jpg
    python scripts/infer.py --weights weights/plate_yolo11.pt --source images/ --save outputs/
    python scripts/infer.py --weights weights/plate_yolo11.pt --source 0          # webcam

With vehicle-plate verification (Section 8):

    python scripts/infer.py --weights weights/plate_yolo11.pt --source car.jpg \
        --vehicle-weights yolo11n.pt --registry configs/registry_example.json
"""
import argparse
import json
from pathlib import Path

import cv2

import _path  # noqa: F401
from lpr.pipeline import LicensePlatePipeline, draw_readings

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_images(source: str):
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                yield f.name, cv2.imread(str(f))
    else:
        yield p.name, cv2.imread(str(p))


def iter_video(source: str, stride: int):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield f"frame_{i:06d}.jpg", frame
            i += 1
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained plate detector weights")
    ap.add_argument("--source", required=True, help="image, folder, video file or webcam index")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-enhance", action="store_true", help="skip the enhancement pipeline")
    ap.add_argument("--save", default=None, help="folder for annotated images")
    ap.add_argument("--stride", type=int, default=5, help="process every Nth video frame")
    ap.add_argument("--vehicle-weights", default=None, help="COCO YOLOv11 weights, e.g. yolo11n.pt")
    ap.add_argument("--registry", default=None, help="registration records (.json/.csv)")
    ap.add_argument("--body-type-weights", default=None)
    ap.add_argument("--make-model-weights", default=None)
    args = ap.parse_args()

    pipe = LicensePlatePipeline.from_weights(
        args.weights, conf=args.conf, device=args.device,
        use_enhancement=not args.no_enhance,
        vehicle_weights=args.vehicle_weights, registry_path=args.registry,
        body_type_weights=args.body_type_weights, make_model_weights=args.make_model_weights,
    )
    src = Path(args.source)
    is_video = args.source.isdigit() or src.suffix.lower() in VID_EXTS
    frames = iter_video(args.source, args.stride) if is_video else iter_images(args.source)
    out_dir = Path(args.save) if args.save else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name, img in frames:
        if img is None:
            print(json.dumps({"source": name, "error": "unreadable image"}))
            continue
        readings = pipe.process(img)
        print(json.dumps({"source": name, "plates": [r.as_dict() for r in readings]}))
        if out_dir:
            cv2.imwrite(str(out_dir / name), draw_readings(img, readings))


if __name__ == "__main__":
    main()
