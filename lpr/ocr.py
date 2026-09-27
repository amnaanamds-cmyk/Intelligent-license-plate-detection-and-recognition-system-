"""PaddleOCR text recognition for plate crops (Section 5.4).

PaddleOCR finds one or more text lines in the enhanced crop and reads each
one. For multi-line plates the lines are sorted top-to-bottom (and
left-to-right within a row) and joined. Both the PaddleOCR 2.x ``ocr()`` API
and the 3.x ``predict()`` API are supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TextLine:
    text: str
    confidence: float
    box: np.ndarray  # (4, 2) polygon

    @property
    def center_y(self) -> float:
        return float(np.mean(self.box[:, 1]))

    @property
    def min_x(self) -> float:
        return float(np.min(self.box[:, 0]))

    @property
    def height(self) -> float:
        return float(np.ptp(self.box[:, 1])) or 1.0


@dataclass
class OCRResult:
    text: str
    confidence: float
    lines: list[TextLine] = field(default_factory=list)


_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def clean_plate_text(text: str) -> str:
    """Uppercase and remove everything except A-Z and 0-9.

    Separators such as '-', '.', spaces and the odd stray symbol read from
    bolts or plate frames are removed so strings can be compared directly.
    """
    return _NON_ALNUM.sub("", text.upper())


def order_lines(lines: list[TextLine]) -> list[list[TextLine]]:
    """Group text lines into rows and sort them top-to-bottom, left-to-right.

    Two boxes are in the same row when their vertical centres are less than
    half the smaller box height apart.
    """
    rows: list[list[TextLine]] = []
    for line in sorted(lines, key=lambda l: l.center_y):
        if rows:
            ref = rows[-1][0]
            if abs(line.center_y - ref.center_y) < 0.5 * min(line.height, ref.height):
                rows[-1].append(line)
                continue
        rows.append([line])
    return [sorted(r, key=lambda l: l.min_x) for r in rows]


def merge_lines(lines: list[TextLine], row_separator: str = " ") -> OCRResult:
    if not lines:
        return OCRResult(text="", confidence=0.0, lines=[])
    rows = order_lines(lines)
    ordered = [l for row in rows for l in row]
    text = row_separator.join("".join(l.text for l in row) for row in rows)
    # Confidence weighted by character count, so a stray 1-char box
    # does not affect the score much.
    weights = np.array([max(1, len(l.text)) for l in ordered], dtype=float)
    confs = np.array([l.confidence for l in ordered], dtype=float)
    conf = float((weights * confs).sum() / weights.sum())
    return OCRResult(text=text, confidence=conf, lines=ordered)


def parse_paddle_output(raw) -> list[TextLine]:
    """Convert raw PaddleOCR output (2.x or 3.x) into ``TextLine`` objects."""
    lines: list[TextLine] = []
    if not raw:
        return lines
    for page in raw:
        if page is None:
            continue
        # PaddleOCR 3.x: dict-like result with rec_texts / rec_scores / rec_polys
        if hasattr(page, "keys") or isinstance(page, dict):
            data = page.get("res", page) if hasattr(page, "get") else page
            texts = data.get("rec_texts", []) or []
            scores = data.get("rec_scores", []) or []
            polys = data.get("rec_polys", None)
            if polys is None:
                polys = data.get("dt_polys", []) or []
            for t, s, p in zip(texts, scores, polys):
                if t:
                    lines.append(TextLine(str(t), float(s),
                                          np.asarray(p, dtype=float).reshape(-1, 2)))
            continue
        # PaddleOCR 2.x: [[box, (text, score)], ...]
        for item in page:
            if not item or len(item) < 2:
                continue
            box, (t, s) = item[0], item[1]
            if t:
                lines.append(TextLine(str(t), float(s),
                                      np.asarray(box, dtype=float).reshape(-1, 2)))
    return lines


class PlateOCR:
    def __init__(self, lang: str = "en", use_gpu: bool = False,
                 min_line_confidence: float = 0.3, det_model_dir: str | None = None,
                 rec_model_dir: str | None = None, rec_model_name: str | None = None):
        """``det_model_dir`` / ``rec_model_dir`` point to local model folders,
        for offline sites or a recogniser fine-tuned on local plates. Leave
        them unset to download PaddleOCR's default models on first use."""
        import paddleocr  # imported here so tests do not need it

        self.min_line_confidence = min_line_confidence
        version = getattr(paddleocr, "__version__", "2")
        self._v3 = int(str(version).split(".")[0]) >= 3
        if self._v3:
            extra = {k: v for k, v in {
                "text_detection_model_dir": det_model_dir,
                "text_recognition_model_dir": rec_model_dir,
                "text_recognition_model_name": rec_model_name,
            }.items() if v}
            self.engine = paddleocr.PaddleOCR(
                **({} if rec_model_name else {"lang": lang}),
                **extra,
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device="gpu" if use_gpu else "cpu",
            )
        else:
            extra = {k: v for k, v in {"det_model_dir": det_model_dir,
                                       "rec_model_dir": rec_model_dir}.items() if v}
            self.engine = paddleocr.PaddleOCR(
                lang=lang, use_angle_cls=True, use_gpu=use_gpu, show_log=False, **extra,
            )

    def recognize(self, image: np.ndarray) -> OCRResult:
        raw = self.engine.predict(image) if self._v3 else self.engine.ocr(image, cls=True)
        lines = [l for l in parse_paddle_output(raw)
                 if l.confidence >= self.min_line_confidence]
        return merge_lines(lines)
