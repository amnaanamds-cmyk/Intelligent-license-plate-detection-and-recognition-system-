# Intelligent License Plate Detection and Recognition System

**YOLOv11 + PaddleOCR for Various Plate Formats**

Abdal Ahmad, Department of Computer Science, Government Post Graduate College Charsadda
Supervisor: Lecturer Izaz Ullah

This repository implements the two-stage LPR pipeline described in the paper,
along with the proposed vehicle-plate verification extension (Section 8):

```
                          ┌─────────────── plate branch (Sections 5.2-5.4) ────────────────┐
 input image ──► YOLOv11 ─┤ crop ► CLAHE ► bilateral ► normalise ► unsharp ► resize ► PaddleOCR ├─► plate string
                          └────────────────────────────────────────────────────────────────┘        │
            (optional)                                                                               ▼
            └─► YOLOv11 (COCO vehicles) ► vehicle crop ► colour / body type / make-model ──► compare ◄── registry lookup
                                                                                              │
                                                                            MATCH / MISMATCH (+ attributes)
```

## Reported detector results (Table 1)

| Metric | Value |
|---|---|
| Precision | 98.3% |
| Recall | 99.3% |
| F1-score | 98.8% |
| mAP@50 | 99.5% |

These numbers come from the authors' 2,200-image dataset. The dataset and
trained weights are not in this repository. Use `scripts/evaluate.py` to
reproduce the metrics on your own copy.

## Repository layout

```
lpr/
  enhancement.py   CLAHE → bilateral → normalisation → unsharp mask → resize (5.3)
  detector.py      YOLOv11 wrapper, box helpers (5.2)
  ocr.py           PaddleOCR wrapper (2.x & 3.x), multi-line ordering, text cleaning (5.4)
  pipeline.py      end-to-end pipeline + drawing
  verification.py  vehicle attributes, registration records, comparator (8)
  metrics.py       F1, CER, exact-match / character accuracy (6)
app/gui.py         PyQt5 desktop application (5.5)
scripts/
  split_dataset.py              train/val/test split of a YOLO-format dataset (5.1)
  train.py                      train the YOLOv11 plate detector (5.2)
  evaluate.py                   detection metrics + recognition metrics / enhancement ablation (6-7)
  infer.py                      CLI inference on images, folders, videos or a webcam
  train_attribute_classifier.py body-type / make-model classifiers for verification (8.2)
configs/
  data.yaml                     YOLO dataset config
  registry_example.json         example registration records
tests/                          unit tests (need only numpy + OpenCV + pytest)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU: install the paddlepaddle-gpu build and CUDA-enabled PyTorch that match your CUDA version
```

## 1. Prepare the dataset (Section 5.1)

Label each image in YOLO format with a single class, `0 = license_plate`
(for example with LabelImg, CVAT or Roboflow). Then split it:

```bash
python scripts/split_dataset.py --src raw_plates --dst datasets/plates --val 0.15 --test 0.15
```

## 2. Train the detector (Section 5.2)

```bash
python scripts/train.py --data configs/data.yaml --model yolo11s.pt --epochs 100 --device 0
# best weights → weights/plate_yolo11.pt; loss/PR/F1 curves → runs/detect/plate_yolo11/
```

Horizontal flipping is turned off because it mirrors the characters. Mosaic,
HSV colour jitter and scale jitter stay on, as in the paper.

## 3. Evaluate (Sections 6-7)

```bash
# precision / recall / F1 / mAP@50 / mAP@50-95
python scripts/evaluate.py detect --weights weights/plate_yolo11.pt --split test

# end-to-end recognition, with and without the enhancement pipeline
#   labels CSV:  image,plate   (image paths relative to the CSV)
python scripts/evaluate.py recognize --weights weights/plate_yolo11.pt \
    --labels datasets/plates/test_plates.csv --ablation --failures outputs/failures.json
```

`--ablation` gives a numerical check of the claim in Section 7 that the
enhancement stages reduce OCR errors on low-light and skewed plates.

## 4. Inference

```bash
python scripts/infer.py --weights weights/plate_yolo11.pt --source car.jpg --save outputs/
python scripts/infer.py --weights weights/plate_yolo11.pt --source traffic.mp4 --stride 5
```

Each image produces one JSON line: plate string, raw OCR text, OCR and
detection confidence, and the bounding box.

## 5. Desktop application (Section 5.5)

```bash
python app/gui.py --weights weights/plate_yolo11.pt
```

Upload an image and click **Detect & recognise**. The app shows the annotated
image, the recognised plate string, a table of all detected plates, and each
enhancement stage for the selected plate. The *Enhancement pipeline* checkbox
compares results with and without enhancement.

## 6. Vehicle-plate verification (Section 8)

Plate cloning or swapping cannot be detected by reading the plate alone. The
verification branch compares what the camera sees with the registration
record for the recognised plate:

1. A COCO-pretrained YOLOv11 (`yolo11n.pt`) finds vehicles (car, motorcycle,
   bus, truck). Each plate is assigned to the vehicle box that contains it.
2. **Colour** is estimated without training, from HSV pixel votes over the
   centre of the vehicle body. **Body type** and **make/model group** use
   optional `yolo11n-cls` classifiers:
   ```bash
   python scripts/train_attribute_classifier.py --data datasets/body_type --export weights/body_type_cls.pt
   ```
3. The plate is looked up in the registry (a JSON or CSV file here, the
   licensing authority's database in production).
4. A rule-based comparator checks only the attributes whose prediction
   confidence passes a per-attribute threshold. Colours that are easy to
   confuse (white/silver, grey/black, …) count as consistent. The result is
   one of `match`, `mismatch` (with the attributes that differ),
   `not_registered` or `insufficient_evidence`.

```bash
python scripts/infer.py --weights weights/plate_yolo11.pt --source car.jpg \
    --vehicle-weights yolo11n.pt --registry configs/registry_example.json \
    --body-type-weights weights/body_type_cls.pt
python app/gui.py --weights weights/plate_yolo11.pt \
    --vehicle-weights yolo11n.pt --registry configs/registry_example.json
```

A mismatch is shown in red for an **operator to review**. It is not an
automatic violation, because the attribute classifiers can be wrong
(Section 8.2). Re-sprays, lighting and classifier errors all cause false
flags. Use `AttributeComparator(min_mismatches=2)` to flag a plate only when
two attributes disagree.

## Tests

```bash
pip install numpy opencv-python pytest
python -m pytest -q
```

The tests cover the enhancement stages, OCR output parsing and multi-line
ordering, the metrics (including the F1 value in Table 1), the verification
logic, and the pipeline wiring using stub models. YOLO and PaddleOCR are not
needed to run them.
