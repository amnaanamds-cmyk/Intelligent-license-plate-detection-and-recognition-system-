"""YOLOv11 license-plate detector (Section 5.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    confidence: float
    class_id: int = 0
    label: str = "license_plate"

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return (max(0, min(x1, width)), max(0, min(y1, height)),
            max(0, min(x2, width)), max(0, min(y2, height)))


def crop_box(image: np.ndarray, box, pad_ratio: float = 0.0) -> np.ndarray:
    """Crop ``box`` from ``image``. A small padding keeps characters that
    touch the edge of the box."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    if pad_ratio:
        px = (x2 - x1) * pad_ratio
        py = (y2 - y1) * pad_ratio
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    x1, y1, x2, y2 = clip_box((x1, y1, x2, y2), w, h)
    return image[y1:y2, x1:x2]


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class YOLODetector:
    """Thin wrapper around an Ultralytics YOLO model.

    It is used for the plate detector (the trained 1-class model) and for the
    vehicle detector in the verification branch (a COCO model restricted to
    vehicle classes through ``classes``).
    """

    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.7,
                 imgsz: int = 640, device: str | None = None,
                 classes: list[int] | None = None):
        from ultralytics import YOLO  # imported here so tests do not need it

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.classes = classes

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, classes=self.classes, verbose=False,
        )
        h, w = image.shape[:2]
        names = self.model.names
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, confs, clss):
                detections.append(Detection(
                    box=clip_box(box, w, h), confidence=float(c),
                    class_id=int(k), label=str(names.get(int(k), k)),
                ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
