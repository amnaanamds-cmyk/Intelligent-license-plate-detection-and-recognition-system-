"""Intelligent License Plate Detection and Recognition (YOLOv11 + PaddleOCR)."""

from .enhancement import EnhancementConfig, PlateEnhancer
from .metrics import character_error_rate, f1_score, recognition_report
from .ocr import OCRResult, clean_plate_text
from .verification import (
    AttributeComparator,
    VehicleRegistry,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "EnhancementConfig",
    "PlateEnhancer",
    "OCRResult",
    "clean_plate_text",
    "f1_score",
    "character_error_rate",
    "recognition_report",
    "AttributeComparator",
    "VehicleRegistry",
    "VerificationResult",
    "VerificationStatus",
]
