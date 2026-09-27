"""Evaluation metrics (Section 6).

The detection metrics (precision, recall, mAP@50) come from Ultralytics'
``model.val()``. This module adds the F1-score and the recognition metrics
that Ultralytics does not report: exact plate accuracy and character error
rate (CER).
"""

from __future__ import annotations

from dataclasses import dataclass

from .ocr import clean_plate_text


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_error_rate(prediction: str, reference: str) -> float:
    """Edit distance divided by reference length, after cleaning both strings."""
    p, r = clean_plate_text(prediction), clean_plate_text(reference)
    if not r:
        return 0.0 if not p else 1.0
    return levenshtein(p, r) / len(r)


@dataclass
class RecognitionReport:
    n: int
    exact_match_accuracy: float
    mean_cer: float
    character_accuracy: float

    def as_dict(self) -> dict:
        return {
            "samples": self.n,
            "exact_match_accuracy": round(self.exact_match_accuracy, 4),
            "mean_cer": round(self.mean_cer, 4),
            "character_accuracy": round(self.character_accuracy, 4),
        }


def recognition_report(predictions: list[str], references: list[str]) -> RecognitionReport:
    if len(predictions) != len(references):
        raise ValueError("predictions and references differ in length")
    n = len(references)
    if n == 0:
        return RecognitionReport(0, 0.0, 0.0, 0.0)
    exact = sum(clean_plate_text(p) == clean_plate_text(r)
                for p, r in zip(predictions, references))
    total_edits = sum(levenshtein(clean_plate_text(p), clean_plate_text(r))
                      for p, r in zip(predictions, references))
    total_chars = sum(len(clean_plate_text(r)) for r in references)
    cers = [character_error_rate(p, r) for p, r in zip(predictions, references)]
    char_acc = max(0.0, 1 - total_edits / total_chars) if total_chars else 0.0
    return RecognitionReport(n, exact / n, sum(cers) / n, char_acc)
