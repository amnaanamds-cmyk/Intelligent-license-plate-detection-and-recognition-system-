from types import SimpleNamespace

import numpy as np

from lpr.tracking import PlateTrack, TrackRead, TrackVoter

FRAME = np.zeros((10, 10, 3), np.uint8)


def reading(text, conf=0.9):
    return SimpleNamespace(plate_text=text, ocr=SimpleNamespace(confidence=conf),
                           detection=SimpleNamespace(confidence=0.9))


def test_consensus_majority():
    t = PlateTrack(1, 0, 0, [TrackRead("LEB1234", .9, True, 0), TrackRead("LEB1284", .8, True, 1),
                             TrackRead("LEB1234", .85, True, 2)])
    text, conf, agreement = t.consensus()
    assert text == "LEB1234" and agreement > 0.6 and 0 < conf <= 1


def test_character_vote_repairs_when_supported():
    reads = [TrackRead(s, .9, True, i) for i, s in
             enumerate(["LEB1284", "LFB1234", "LEB1234", "LEB7234"])]
    assert PlateTrack(1, 0, 0, reads).consensus()[0] == "LEB1234"


def test_early_emit_after_agreeing_reads_and_stop_ocr():
    v = TrackVoter(min_reads=3, agreement=0.6, max_reads=10, lost_frames=5)
    for i in range(2):
        v.add(7, i, reading("ABC123"), FRAME, True)
        assert v.collect(i) == []
    v.add(7, 2, reading("ABC123"), FRAME, True)
    [t] = v.collect(2)
    assert t.consensus()[0] == "ABC123" and not v.needs_ocr(7)
    assert v.collect(3) == []  # emitted only once


def test_emit_when_track_lost_and_cleanup():
    v = TrackVoter(min_reads=5, lost_frames=3)
    v.add(1, 0, reading("XYZ999", 0.5), FRAME, False)
    assert v.collect(2) == []
    [t] = v.collect(10)
    assert t.track_id == 1 and 1 not in v.tracks


def test_flush_and_best_frame():
    v = TrackVoter(min_reads=99)
    v.add(1, 0, reading("AAA111", 0.4), FRAME, True)
    v.add(1, 1, reading("AAA111", 0.95), FRAME + 1, True)
    [t] = v.flush()
    assert t.best_read.ocr.confidence == 0.95 and t.best_frame.max() == 1
    assert v.flush() == []
