"""Export the trained detector to a faster deployment format.

    python scripts/export.py --weights weights/plate_yolo11.pt --format onnx
    python scripts/export.py --weights weights/plate_yolo11.pt --format engine --half   # NVIDIA TensorRT
    python scripts/export.py --weights weights/plate_yolo11.pt --format openvino        # Intel CPU / iGPU

Then point ``detector.weights`` in configs/system.yaml at the exported file.
Ultralytics loads it the same way as the .pt file, tracking included.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx",
                    choices=["onnx", "engine", "openvino", "torchscript", "ncnn", "tflite"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (GPU formats)")
    ap.add_argument("--int8", action="store_true", help="INT8 quantisation (needs --data)")
    ap.add_argument("--data", default=None, help="dataset yaml for INT8 calibration")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    kw = {"format": args.format, "imgsz": args.imgsz, "half": args.half,
          "int8": args.int8, "device": args.device}
    if args.data:
        kw["data"] = args.data
    path = YOLO(args.weights).export(**kw)
    print(f"exported: {path}")


if __name__ == "__main__":
    main()
