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
  geometry.py      deskew (rotation correction) for tilted plates
  postprocess.py   plate-format rules: O/0, B/8, S/5 correction, ignore province text
  tracking.py      multi-frame voting: one event per vehicle from many reads
  stream.py        RTSP/webcam/video workers, reconnect, event handling
  storage.py       SQLite event log + snapshot evidence + watchlist, retention purge
  alerts.py        watchlist (exact + fuzzy) and mismatch alerts, webhook delivery
  config.py        YAML config with defaults
service/api.py     REST API (FastAPI) for toll / parking / gate integration
app/gui.py         PyQt5 desktop application (5.5)
scripts/
  split_dataset.py              train/val/test split of a YOLO-format dataset (5.1)
  train.py                      train the YOLOv11 plate detector (5.2)
  evaluate.py                   detection metrics + recognition metrics / enhancement ablation (6-7)
  infer.py                      CLI inference on images, folders, videos or a webcam
  train_attribute_classifier.py body-type / make-model classifiers for verification (8.2)
  serve.py                      run cameras + REST API as a service
  export.py                     ONNX / TensorRT / OpenVINO export
  export_ocr_dataset.py         build a PaddleOCR fine-tuning set from your plates
configs/
  system.yaml                   deployment config (cameras, formats, alerts, storage, API)
  data.yaml                     YOLO dataset config
  registry_example.json         example registration records
docs/DEPLOYMENT.md              step-by-step guide to a real-world installation
Dockerfile, docker-compose.yml  containerised service
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

## 7. Real-world deployment (live cameras, database, alerts, API)

For a working installation, run the system as a service instead of one
image at a time:

```
RTSP camera ─► FrameSource (auto-reconnect, newest frame only)
            ─► YOLOv11 + ByteTrack (a track ID per plate)
            ─► deskew ► enhancement ► PaddleOCR ► plate-format correction
            ─► vote over the track's reads ─► ONE event per vehicle
            ─► SQLite + snapshots ─► watchlist / mismatch alerts ─► webhook
            ─► REST API (/events, /recognize, /watchlist, /cameras)
```

```bash
# 1. put trained weights in weights/, 2. edit configs/system.yaml (cameras, plate templates, api_key)
docker compose up -d --build              # or: python scripts/serve.py --config configs/system.yaml
curl -H "X-API-Key: <key>" "http://localhost:8000/events?plate=LEB"
curl -H "X-API-Key: <key>" -X POST localhost:8000/watchlist -H 'Content-Type: application/json' \
     -d '{"plate": "LEB1234", "reason": "reported stolen"}'
curl -H "X-API-Key: <key>" -F image=@car.jpg "localhost:8000/recognize?save=true"
```

Why these parts matter on real roads:
* **Multi-frame voting.** One blurred frame no longer decides the result. A
  car is seen in 10-40 frames, and the confidence-weighted vote (including a
  per-character vote) settles on the reading most frames agree on. OCR stops
  for a track once it is confident, which saves compute.
* **Plate-format rules.** Correct each position using the known layout,
  drop province words and stray characters, and flag reads that fit no format.
* **Deskew.** Straighten tilted plates before enhancement.
* **Evidence and audit.** Every event stores the timestamp, camera, the
  full frame with the box drawn, the plate crop, all individual reads and
  the alerts. Old records are purged according to `retention_days`.
* **Fuzzy watchlist.** A one-character OCR error still raises a watchlist
  alert, marked `fuzzy` for the operator.

**See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for the full checklist:
data collection, OCR fine-tuning, camera placement, hardware sizing, pilot
testing and privacy.

## Tests

```bash
pip install numpy opencv-python pytest
python -m pytest -q
```

The tests cover the enhancement stages, OCR output parsing and multi-line
ordering, the metrics (including the F1 value in Table 1), the verification
logic, deskew, format correction, multi-frame voting, storage, retention,
alerts and webhooks, the camera worker on a video file, and the REST API.
The models are replaced by stubs, so YOLO and PaddleOCR are not needed.
The API tests need `fastapi` and `httpx`.
