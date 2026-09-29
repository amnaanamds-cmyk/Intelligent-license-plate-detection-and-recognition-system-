# Taking the system into real-world use

The code in this repository covers the full software path, from camera to
event database, alerts and API. What it cannot provide is **your data,
your cameras and your site**. Real-world accuracy depends mostly on those.
Work through these steps in order.

---

## Step 1 - Train the detector on your own images

The paper's model was trained on 2,200 images. For a deployment, add
images **from the actual cameras you will use** (same angle, height, lens,
day and night). A detector trained on phone photos loses accuracy on
CCTV footage.

| Need | Recommendation |
|---|---|
| Images | 2,000+ total. Include at least 300-500 from each deployment camera |
| Conditions | day, night (IR), rain, glare, dirty plates, motorcycles, trucks, multi-line plates |
| Negatives | ~10% frames with **no** plate (empty road), to reduce false alarms |
| Labels | one class `license_plate`, tight boxes, label **every** visible plate |

```bash
python scripts/split_dataset.py --src raw_plates --dst datasets/plates
python scripts/train.py --data configs/data.yaml --model yolo11s.pt --epochs 150 --device 0
python scripts/evaluate.py detect --weights weights/plate_yolo11.pt --split test
```

`yolo11s` is a good accuracy/speed balance. Use `yolo11n` for weak CPUs or
edge devices, and `yolo11m` if you have a GPU to spare.

## Step 2 - Make the OCR read your plates

1. **Set the plate formats.** In `configs/system.yaml` → `plate_format.templates`,
   list the registration layouts used where the system will run
   (`L` = letter, `D` = digit). The shipped templates are only examples.
   Correct templates fix most O/0, I/1, B/8 and S/5 mistakes automatically.
2. **Set the ignore words.** List the printed words that are not part of
   the number (province names, "GOVT", dealer text) in `ignore_words`.
3. **Measure.** Build a CSV `image,plate` of 300+ test images with the true
   plate string and run:
   ```bash
   python scripts/evaluate.py recognize --weights weights/plate_yolo11.pt \
       --labels datasets/plates/test_plates.csv --ablation --failures outputs/failures.json
   ```
   Look through `failures.json` to see what goes wrong.
4. **Fine-tune the recogniser** if exact-match accuracy is still below about
   95%. The stock PaddleOCR model has never seen your plate fonts.
   ```bash
   python scripts/export_ocr_dataset.py --weights weights/plate_yolo11.pt \
       --labels datasets/plates/all_plates.csv --out datasets/ocr_rec
   git clone https://github.com/PaddlePaddle/PaddleOCR && cd PaddleOCR
   # choose the PP-OCR recognition config matching your PaddleOCR version (configs/rec/...)
   python tools/train.py -c configs/rec/<PP-OCR version>/<rec config>.yml \
       -o Global.pretrained_model=<downloaded pretrained rec model> \
          Global.character_dict_path=../datasets/ocr_rec/plate_dict.txt \
          Train.dataset.data_dir=../datasets/ocr_rec \
          Train.dataset.label_file_list=[../datasets/ocr_rec/rec_gt_train.txt] \
          Eval.dataset.data_dir=../datasets/ocr_rec \
          Eval.dataset.label_file_list=[../datasets/ocr_rec/rec_gt_val.txt]
   python tools/export_model.py -c <same config> \
       -o Global.pretrained_model=output/<run>/best_accuracy Global.save_inference_dir=../weights/plate_rec
   ```
   Then set `ocr.rec_model_dir: weights/plate_rec` (and `ocr.rec_model_name`
   to the model name used in the config) in `configs/system.yaml`.
   A few thousand plate crops is usually enough for a clear improvement.

## Step 3 - Install the cameras correctly

Camera placement matters more than any model. Rules of thumb for ANPR cameras:

| Factor | Target |
|---|---|
| Plate size in image | ≥ 100-130 px wide (characters ≥ ~15 px tall). `detector.min_plate_width` skips plates that are still too small |
| Horizontal angle to plate | ≤ 30° |
| Vertical angle | ≤ 30° (mount 1.5-3 m high looking down the lane) |
| Shutter speed | 1/1000 s or faster for moving traffic (prevents motion blur) |
| Night | IR illuminator + camera with IR-cut filter, or a dedicated ANPR camera |
| Lens | varifocal zoomed onto the lane, not a wide-angle overview |
| Stream | RTSP, H.264/H.265, 1080p, 15-25 fps is plenty |
| Coverage | one camera per lane for toll/gates. Overview cameras miss plates |

Put the stream URL in `cameras:` in `configs/system.yaml`. `stride: 2`
processes every second frame, which is plenty at 25 fps.

## Step 4 - Choose hardware

Rough throughput (1080p stream, `yolo11s`, one plate at a time):

| Hardware | Cameras (approx.) |
|---|---|
| 4-core CPU, no GPU | 1-2 at reduced FPS (use `yolo11n`, `stride: 3`, ONNX/OpenVINO) |
| Intel i5/i7 with OpenVINO export | 2-4 |
| NVIDIA GTX 1660 / RTX 3060 | 6-12 |
| Jetson Orin Nano / NX (TensorRT) | 2-4 per device, edge install |

Export for speed:
```bash
python scripts/export.py --weights weights/plate_yolo11.pt --format engine --half   # NVIDIA
python scripts/export.py --weights weights/plate_yolo11.pt --format openvino        # Intel
```
Measure on your own hardware: `GET /cameras` shows the live FPS of each camera.

## Step 5 - Deploy the service

```bash
cp <trained weights> weights/plate_yolo11.pt
# edit configs/system.yaml: cameras, templates, api_key, webhook_url, retention_days
docker compose up -d --build
curl http://localhost:8000/health
```
or without Docker: `python scripts/serve.py --config configs/system.yaml`.

* **API docs**: `http://<server>:8000/docs`
* **Offline sites**: the first start downloads the PaddleOCR models. Start
  once with internet access (they are cached in the `lpr-models` volume),
  or copy the model folders and set `ocr.det_model_dir` / `ocr.rec_model_dir`.
* **Accounts**: open the web app right after the first start and create
  the admin account. Until then the system is in open setup mode. Give every
  person their own account with the lowest role they need (viewer /
  operator / admin). The audit log records logins, searches, exports and
  changes.
* **Security**: keep the service on the internal network or behind a reverse
  proxy with HTTPS (Caddy, nginx, or `scripts/serve.py --ssl-certfile/--ssl-keyfile`).
  Use `api.api_key` only for machine integrations, and never expose camera
  RTSP ports to the internet. HTTPS also stops session tokens from being
  sent in clear text over Wi-Fi.
* **Integration**: toll, parking and barrier systems either poll
  `GET /events?since=...`, or receive pushes through `alerts.webhook_url`
  (alerts), or call `POST /recognize` with their own snapshots.

## Step 6 - Run a pilot before going live

Run 2-4 weeks in **shadow mode**: the system records, but people still make
the decisions. Every day, compare a sample of events with the snapshots:

* **Capture rate** = vehicles with an event ÷ vehicles that passed
* **Read accuracy** = events with the correct plate ÷ events
* **False alerts** per day (watchlist fuzzy matches, verification mismatches)

Adjust `detector.conf`, `tracking.min_reads`, `plate_format` and
`alerts.watchlist_max_distance` until the numbers meet the site's needs.
Only then connect automatic actions (barriers, fines, billing), and keep a
human review step for anything with legal consequences.

## Step 7 - Legal and privacy

Plate numbers and vehicle images are personal data in most jurisdictions.
Before operating:
* get authorisation from the site owner or authority, and put up signage
  where the law requires it;
* set `storage.retention_days` to the shortest period you need (events and
  snapshots are purged automatically);
* restrict API access (`api_key`) and keep an audit trail of who searches;
* treat watchlist and verification alerts as **leads for human review**,
  never as automatic proof.

## Maintenance checklist

- [ ] Check `GET /cameras` daily: `connected`, `fps`, `error`
- [ ] Clean camera lenses and IR lights monthly
- [ ] Re-label misread snapshots and retrain every few months
- [ ] Back up `data/lpr.db` and keep the watchlist up to date
