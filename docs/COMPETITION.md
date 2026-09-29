# Competition guide - PlateVision LPR

How to prepare, demo and pitch the system in a commercial or innovation
competition.

---

## 1. Checklist before the event

| # | Task | How |
|---|---|---|
| 1 | Trained detector in `weights/plate_yolo11.pt` | `notebooks/train_on_colab.ipynb` (free GPU) or `scripts/train.py` |
| 2 | Your own measured results | `scripts/evaluate.py detect` and `evaluate.py recognize --ablation`. Put **these** numbers on your slides |
| 3 | Real plate formats in `configs/system.yaml` → `plate_format.templates` | the local series (e.g. `LLLDDDD`) |
| 4 | Start the server **once with internet** | downloads the PaddleOCR models (cached afterwards, so the demo works offline) |
| 5 | Admin account created | open the web app → first-run setup, or `python scripts/manage_users.py add admin --role admin` |
| 6 | Demo data | 10-20 test photos of local cars on the phone, 1 short traffic video in `samples/` |
| 7 | Watchlist entry for the demo car | Watchlist tab → add its plate with reason "Reported stolen" |
| 8 | Laptop and phone on the **same Wi-Fi** | a phone hotspot works well and avoids venue Wi-Fi that blocks devices from reaching each other |
| 9 | Rehearse the 5-minute demo below twice | time it |

## 2. Start it

```bash
# Windows: double-click start.bat        Linux/Mac: ./start.sh
```
The console prints the addresses, for example:
```
  PlateVision LPR is starting. Open the web app at:
    http://localhost:8000
    http://192.168.43.17:8000      <- type this on the phone
```

Optional live-camera demo: add a camera in the **Cameras** tab, with source
`samples/traffic.mp4` or a phone running an IP-camera app
(`http://<phone-ip>:8080/video`).

## 3. Five-minute demo script

| Time | Show | Say |
|---|---|---|
| 0:00 | Title slide | "Manual plate checking at gates, tolls and checkpoints is slow and error-prone. Plate cloning goes undetected." |
| 0:30 | **Phone → Scan → Take photo** of a demo car / printed plate | "Any phone becomes a plate reader. No app install is needed: it's a web app." |
| 1:00 | Result: plate, confidence, detected crop vs **enhanced for OCR** | "YOLOv11 finds the plate, our CLAHE-bilateral-normalise-unsharp pipeline cleans it, PaddleOCR reads it, and format rules fix O/0 and B/8 mix-ups." |
| 1:45 | Scan the **watchlisted** car → red alert, phone vibrates | "Stolen or wanted vehicles alert instantly, even when the OCR gets one character wrong (fuzzy match)." |
| 2:30 | **Laptop → Live dashboard**: counters, hourly chart, the alert just appeared | "The control room sees every read from every camera and phone in real time." |
| 3:00 | Cameras tab: the video stream running, fps | "Existing CCTV cameras connect over RTSP. One read per vehicle, voted over many frames, so a single blurred frame can't produce a wrong result." |
| 3:30 | Search a plate → event → evidence photo → **Export CSV** | "Every read is evidence: time, camera, photo, plate crop, all stored. Old records are deleted automatically for privacy." |
| 4:00 | Admin: users with roles + **audit log** | "Operators, viewers and admins. Every search is logged, so the system is accountable." |
| 4:30 | Architecture slide + your measured numbers | "mAP@50 of X%, recognition accuracy of Y%, Z ms per plate on a laptop CPU." |

**If something fails on stage:** the Scan tab also works with gallery photos.
Keep screenshots of every step as a backup.

## 4. What sets it apart (pitch points)

- **Pre-OCR enhancement** (the research contribution): measurable accuracy
  gain on low-light and tilted plates. Show the `--ablation` numbers.
- **Vehicle-plate verification** against the registration record, to detect
  cloned or swapped plates, which plain plate readers cannot do.
- **Multi-frame voting and format-aware correction**: an engineering answer
  to real-world blur and OCR confusions.
- **Works with what customers already have**: existing CCTV (RTSP), any
  phone browser, a normal PC. No proprietary ANPR cameras required.
- **Open integration**: REST API, webhooks and CSV for toll, parking, gate
  and police systems.
- **Offline-capable and on-premises**: data never has to leave the site.
- **Privacy by design**: roles, audit log, automatic retention purge.

## 5. Business model ideas

| Customer | Offer |
|---|---|
| Housing societies, campuses, factories | Gate access log + visitor records, monthly licence per gate |
| Parking operators | Entry/exit timing → automatic billing via the API |
| Toll plazas, city traffic | Watchlist alerts, cloned-plate detection, analytics |
| Police checkpoints | Phone-based scanning with instant stolen-vehicle alerts |

Pricing options include a one-time installation fee plus a per-camera
monthly licence, or software-only for integrators.

## 6. Be ready for these judge questions

- **"How accurate is it?"** Quote your own test-set numbers (detection
  mAP@50, exact-plate accuracy), and say which conditions are hardest.
- **"What about night / rain?"** IR cameras + the enhancement pipeline. Show
  low-light examples from your test set.
- **"Privacy?"** Retention purge, roles, audit log, on-premises storage;
  alerts are reviewed by a human before any action is taken.
- **"How does it scale?"** One thread per camera, a shared OCR engine,
  ONNX/TensorRT export. Mention the number of cameras per GPU you measured.
- **"What's next?"** A native offline mobile app (TFLite), OCR fine-tuned on
  local plates, integration with the vehicle registration database for
  verification.
