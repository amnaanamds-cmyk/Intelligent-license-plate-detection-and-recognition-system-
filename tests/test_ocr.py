import numpy as np

from lpr.ocr import TextLine, clean_plate_text, merge_lines, parse_paddle_output


def box(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)


def test_clean_plate_text():
    assert clean_plate_text(" abc-12.34 ") == "ABC1234"
    assert clean_plate_text("") == ""


def test_merge_multiline_top_to_bottom():
    lines = [TextLine("1234", 0.9, box(10, 50, 90, 80)),
             TextLine("LEB", 0.8, box(20, 5, 80, 35))]
    r = merge_lines(lines)
    assert r.text == "LEB 1234"
    assert clean_plate_text(r.text) == "LEB1234"


def test_merge_same_row_left_to_right():
    lines = [TextLine("5678", 0.9, box(100, 10, 180, 40)),
             TextLine("ABC", 0.9, box(10, 12, 80, 42))]
    assert merge_lines(lines).text == "ABC5678"


def test_confidence_weighted_by_length():
    lines = [TextLine("ABCDEF", 1.0, box(0, 0, 100, 30)),
             TextLine("x", 0.0, box(110, 0, 120, 30))]
    assert abs(merge_lines(lines).confidence - 6 / 7) < 1e-9


def test_empty():
    r = merge_lines([])
    assert r.text == "" and r.confidence == 0.0


def test_parse_paddle_v2_format():
    raw = [[[box(0, 0, 50, 20).tolist(), ("ABC", 0.95)],
            [box(0, 30, 50, 50).tolist(), ("123", 0.9)]]]
    lines = parse_paddle_output(raw)
    assert [l.text for l in lines] == ["ABC", "123"]
    assert parse_paddle_output([None]) == []


def test_parse_paddle_v3_format():
    raw = [{"rec_texts": ["PES", "5678"], "rec_scores": [0.9, 0.8],
            "rec_polys": [box(0, 0, 50, 20), box(0, 30, 50, 50)]}]
    lines = parse_paddle_output(raw)
    assert merge_lines(lines).text == "PES 5678"
