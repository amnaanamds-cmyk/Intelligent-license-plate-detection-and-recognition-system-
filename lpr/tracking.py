"""Combining the reads of one plate over many video frames.

A car stays in view for dozens of frames. Reading a single frame is fragile
(motion blur, glare, partial occlusion), but most frames of the same car
agree. Each plate is tracked across frames (ByteTrack through Ultralytics),
the OCR reads for each track are collected, and one event is emitted per
vehicle using a confidence-weighted vote.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackRead:
    text: str
    confidence: float
    valid: bool
    frame_index: int


@dataclass
class PlateTrack:
    track_id: int
    first_frame: int
    last_frame: int
    reads: list[TrackRead] = field(default_factory=list)
    best_read: object | None = None      # the PlateReading with the highest score
    best_frame: np.ndarray | None = None  # full frame of the best read (evidence)
    best_score: float = -1.0
    emitted: bool = False

    def weights(self) -> dict[str, float]:
        w: dict[str, float] = defaultdict(float)
        for r in self.reads:
            if r.text:
                # Reads that fit a known plate format count more.
                w[r.text] += r.confidence * (1.5 if r.valid else 1.0)
        return w

    def consensus(self) -> tuple[str, float, float]:
        """Return (text, confidence, agreement).

        ``agreement`` is the share of the total vote weight held by the
        winning string. The winner is refined by a per-character vote over
        all reads of the same length, which can fix a character that no
        single frame read correctly.
        """
        w = self.weights()
        if not w:
            return "", 0.0, 0.0
        total = sum(w.values())
        winner = max(w, key=w.get)
        same_len = [r for r in self.reads if len(r.text) == len(winner)]
        if len(same_len) >= 3:
            chars = []
            for i in range(len(winner)):
                votes: Counter = Counter()
                for r in same_len:
                    votes[r.text[i]] += r.confidence * (1.5 if r.valid else 1.0)
                chars.append(votes.most_common(1)[0][0])
            voted = "".join(chars)
            if voted != winner:
                # Keep the per-character result only when some frame supports
                # it or it fits a format, so no new string is invented from noise.
                if voted in w or any(r.valid and r.text == voted for r in self.reads):
                    winner = voted
        agreement = w.get(winner, 0.0) / total
        confs = [r.confidence for r in self.reads if r.text == winner] or [0.0]
        return winner, float(np.mean(confs)) * agreement, agreement


class TrackVoter:
    def __init__(self, min_reads: int = 3, agreement: float = 0.6,
                 max_reads: int = 12, lost_frames: int = 30):
        self.min_reads = min_reads
        self.agreement = agreement
        self.max_reads = max_reads
        self.lost_frames = lost_frames
        self.tracks: dict[int, PlateTrack] = {}

    def needs_ocr(self, track_id: int) -> bool:
        t = self.tracks.get(track_id)
        return t is None or (not t.emitted and len(t.reads) < self.max_reads)

    def seen(self, track_id: int, frame_index: int) -> PlateTrack:
        t = self.tracks.get(track_id)
        if t is None:
            t = self.tracks[track_id] = PlateTrack(track_id, frame_index, frame_index)
        t.last_frame = frame_index
        return t

    def add(self, track_id: int, frame_index: int, reading, frame: np.ndarray,
            valid: bool) -> None:
        t = self.seen(track_id, frame_index)
        text = reading.plate_text
        t.reads.append(TrackRead(text, reading.ocr.confidence, valid, frame_index))
        score = reading.ocr.confidence * reading.detection.confidence + (0.5 if valid else 0)
        if text and score > t.best_score:
            t.best_score, t.best_read, t.best_frame = score, reading, frame.copy()

    def _ready(self, t: PlateTrack) -> bool:
        if t.emitted or not t.reads:
            return False
        text, _, agreement = t.consensus()
        support = sum(1 for r in t.reads if r.text == text)
        return bool(text) and support >= self.min_reads and agreement >= self.agreement

    def collect(self, frame_index: int) -> list[PlateTrack]:
        """Return the tracks to emit now: those confident enough (early emit,
        for low latency at gates) and those that just left the view. Old
        tracks are dropped from memory."""
        out = []
        for tid in list(self.tracks):
            t = self.tracks[tid]
            lost = frame_index - t.last_frame > self.lost_frames
            if self._ready(t) or (lost and not t.emitted and t.consensus()[0]):
                t.emitted = True
                out.append(t)
            if lost:
                del self.tracks[tid]
        return out

    def flush(self) -> list[PlateTrack]:
        """Emit every remaining track that has a read (end of video)."""
        out = [t for t in self.tracks.values() if not t.emitted and t.consensus()[0]]
        for t in out:
            t.emitted = True
        self.tracks.clear()
        return out
