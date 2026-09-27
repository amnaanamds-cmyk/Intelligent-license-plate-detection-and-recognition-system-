"""PyQt5 desktop application (Section 5.5).

Workflow: load models, then open an image. The system finds and crops each
plate, runs the enhancement stages and shows the recognised plate string.
If a vehicle detector and a registry file are loaded, each plate also gets
a vehicle-plate verification flag (Section 8).

    python app/gui.py --weights weights/plate_yolo11.pt
    python app/gui.py --weights weights/plate_yolo11.pt \
        --vehicle-weights yolo11n.pt --registry configs/registry_example.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpr.pipeline import LicensePlatePipeline, PlateReading, draw_readings  # noqa: E402

STAGE_TITLES = {
    "input": "Cropped plate", "clahe": "CLAHE", "bilateral": "Bilateral filter",
    "normalize": "Normalisation", "unsharp": "Unsharp mask", "output": "OCR input",
}


def to_pixmap(img: np.ndarray, max_w: int | None = None, max_h: int | None = None) -> QPixmap:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pm = QPixmap.fromImage(qimg)
    if max_w or max_h:
        pm = pm.scaled(max_w or pm.width(), max_h or pm.height(),
                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm


class Worker(QObject):
    """Loads models and runs the pipeline off the UI thread."""

    loaded = pyqtSignal(str)
    finished = pyqtSignal(object, object)  # image, readings
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.pipeline: LicensePlatePipeline | None = None

    def load(self, opts: dict):
        try:
            self.pipeline = LicensePlatePipeline.from_weights(**opts)
            msg = "Models loaded"
            if self.pipeline.verification_enabled:
                msg += f" (verification on, {len(self.pipeline.registry)} registered plates)"
            self.loaded.emit(msg)
        except Exception as e:  # shown to the user rather than crashing the GUI
            self.failed.emit(f"Failed to load models: {e}")

    def run(self, image: np.ndarray, use_enhancement: bool):
        if self.pipeline is None:
            self.failed.emit("Load the models first.")
            return
        try:
            self.pipeline.use_enhancement = use_enhancement
            self.finished.emit(image, self.pipeline.process(image))
        except Exception as e:
            self.failed.emit(f"Processing failed: {e}")


class MainWindow(QMainWindow):
    request_load = pyqtSignal(dict)
    request_run = pyqtSignal(object, bool)

    def __init__(self, defaults: argparse.Namespace):
        super().__init__()
        self.setWindowTitle("Intelligent License Plate Recognition - YOLOv11 + PaddleOCR")
        self.resize(1300, 820)
        self.image: np.ndarray | None = None
        self.readings: list[PlateReading] = []

        self.thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.request_load.connect(self.worker.load)
        self.request_run.connect(self.worker.run)
        self.worker.loaded.connect(self._on_loaded)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()

        self._build_ui(defaults)
        if defaults.weights:
            self._load_models()

    # ------------------------------------------------------------------ UI --
    def _path_row(self, text: str, file_filter: str) -> tuple[QWidget, QLineEdit]:
        edit = QLineEdit(text or "")
        btn = QPushButton("…")
        btn.setFixedWidth(30)
        btn.clicked.connect(lambda: self._browse(edit, file_filter))
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit)
        lay.addWidget(btn)
        return w, edit

    def _build_ui(self, d):
        models = QGroupBox("Models")
        form = QFormLayout(models)
        row, self.weights_edit = self._path_row(d.weights, "Weights (*.pt)")
        form.addRow("Plate detector", row)
        row, self.vehicle_edit = self._path_row(d.vehicle_weights, "Weights (*.pt)")
        form.addRow("Vehicle detector (opt.)", row)
        row, self.registry_edit = self._path_row(d.registry, "Registry (*.json *.csv)")
        form.addRow("Registry (opt.)", row)
        self.load_btn = QPushButton("Load models")
        self.load_btn.clicked.connect(self._load_models)
        form.addRow(self.load_btn)

        self.open_btn = QPushButton("Upload image…")
        self.open_btn.clicked.connect(self._open_image)
        self.run_btn = QPushButton("Detect && recognise")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.enhance_chk = QCheckBox("Enhancement pipeline")
        self.enhance_chk.setChecked(True)
        self.save_btn = QPushButton("Save result…")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)

        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 28px; font-weight: bold; padding: 8px;")
        self.verify_label = QLabel("")
        self.verify_label.setWordWrap(True)
        self.verify_label.setAlignment(Qt.AlignCenter)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Plate", "OCR conf.", "Det. conf.", "Vehicle", "Verification"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_row_selected)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(models)
        ll.addWidget(self.open_btn)
        ll.addWidget(self.enhance_chk)
        ll.addWidget(self.run_btn)
        ll.addWidget(self.save_btn)
        ll.addWidget(QLabel("Recognised plate:"))
        ll.addWidget(self.result_label)
        ll.addWidget(self.verify_label)
        ll.addWidget(self.table, 1)

        self.image_label = QLabel("Upload an image to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setStyleSheet("background: #222; color: #aaa;")

        self.stages_box = QWidget()
        self.stages_layout = QHBoxLayout(self.stages_box)
        stages_scroll = QScrollArea()
        stages_scroll.setWidgetResizable(True)
        stages_scroll.setWidget(self.stages_box)
        stages_scroll.setFixedHeight(170)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.image_label, 1)
        rl.addWidget(QLabel("Enhancement stages (selected plate):"))
        rl.addWidget(stages_scroll)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 3)
        self.setCentralWidget(split)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------- actions --
    def _browse(self, edit: QLineEdit, file_filter: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", file_filter)
        if path:
            edit.setText(path)

    def _load_models(self):
        weights = self.weights_edit.text().strip()
        if not weights:
            QMessageBox.warning(self, "Missing weights", "Select the plate detector weights.")
            return
        opts = {"plate_weights": weights}
        if self.vehicle_edit.text().strip() and self.registry_edit.text().strip():
            opts["vehicle_weights"] = self.vehicle_edit.text().strip()
            opts["registry_path"] = self.registry_edit.text().strip()
        self._busy(True, "Loading models (first run downloads PaddleOCR weights)…")
        self.request_load.emit(opts)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Cannot read {path}")
            return
        self.image = img
        self.readings = []
        self._show_main(img)
        self._clear_results()
        self.run_btn.setEnabled(self.worker.pipeline is not None)
        self.statusBar().showMessage(Path(path).name)

    def _run(self):
        if self.image is None:
            return
        self._busy(True, "Processing…")
        self.request_run.emit(self.image, self.enhance_chk.isChecked())

    def _save(self):
        if self.image is None or not self.readings:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "result.jpg", "Images (*.jpg *.png)")
        if path:
            cv2.imwrite(path, draw_readings(self.image, self.readings))

    # ------------------------------------------------------------- results --
    def _busy(self, busy: bool, msg: str = ""):
        for b in (self.load_btn, self.open_btn, self.run_btn):
            b.setEnabled(not busy)
        if not busy:
            self.run_btn.setEnabled(self.image is not None and self.worker.pipeline is not None)
        if msg:
            self.statusBar().showMessage(msg)

    def _on_loaded(self, msg: str):
        self._busy(False, msg)

    def _on_failed(self, msg: str):
        self._busy(False, msg)
        QMessageBox.critical(self, "Error", msg)

    def _on_finished(self, image, readings: list[PlateReading]):
        self._busy(False, f"{len(readings)} plate(s) found")
        self.readings = readings
        self.save_btn.setEnabled(bool(readings))
        self._show_main(draw_readings(image, readings))
        self.table.setRowCount(len(readings))
        for i, r in enumerate(readings):
            v = r.verification
            cells = [r.plate_text or "(unreadable)", f"{r.ocr.confidence:.2f}",
                     f"{r.detection.confidence:.2f}",
                     r.vehicle.label if r.vehicle else "—",
                     v.status.value.replace("_", " ") if v else "—"]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if v is not None and v.status.value == "mismatch":
                    item.setBackground(Qt.red)
                    item.setForeground(Qt.white)
                self.table.setItem(i, j, item)
        if readings:
            self.table.selectRow(0)
        else:
            self._clear_results("No plate detected")

    def _on_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.readings):
            return
        r = self.readings[rows[0].row()]
        self.result_label.setText(r.plate_text or "(unreadable)")
        v = r.verification
        if v is None:
            self.verify_label.setText("")
            self.verify_label.setStyleSheet("")
        else:
            colour = {"match": "#1b8a3a", "mismatch": "#c62828"}.get(v.status.value, "#8a6d1b")
            self.verify_label.setText(v.summary())
            self.verify_label.setStyleSheet(
                f"color: white; background: {colour}; padding: 6px; border-radius: 4px;")
        self._show_stages(r.stages)

    def _clear_results(self, text: str = "—"):
        self.table.setRowCount(0)
        self.result_label.setText(text)
        self.verify_label.setText("")
        self.verify_label.setStyleSheet("")
        self._show_stages({})

    def _show_main(self, img: np.ndarray):
        self.image_label.setPixmap(
            to_pixmap(img, self.image_label.width(), self.image_label.height()))

    def _show_stages(self, stages: dict[str, np.ndarray]):
        while self.stages_layout.count():
            w = self.stages_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        for key, title in STAGE_TITLES.items():
            if key not in stages:
                continue
            box = QWidget()
            lay = QVBoxLayout(box)
            img = QLabel()
            img.setPixmap(to_pixmap(stages[key], 220, 110))
            lay.addWidget(img, alignment=Qt.AlignCenter)
            lay.addWidget(QLabel(title), alignment=Qt.AlignCenter)
            self.stages_layout.addWidget(box)
        self.stages_layout.addStretch(1)

    def closeEvent(self, event):
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/plate_yolo11.pt"
                    if Path("weights/plate_yolo11.pt").exists() else None)
    ap.add_argument("--vehicle-weights", default=None)
    ap.add_argument("--registry", default=None)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
