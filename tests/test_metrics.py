import pytest

from lpr.metrics import character_error_rate, f1_score, levenshtein, recognition_report


def test_f1_matches_paper_table():
    # Table 1: P = 98.3%, R = 99.3%  ->  F1 = 98.8%
    assert round(f1_score(0.983, 0.993) * 100, 1) == 98.8
    assert f1_score(0, 0) == 0.0


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_cer_ignores_formatting():
    assert character_error_rate("abc-123", "ABC123") == 0.0
    assert character_error_rate("ABC128", "ABC123") == pytest.approx(1 / 6)


def test_recognition_report():
    rep = recognition_report(["ABC123", "XYZ9", ""], ["ABC123", "XYZ99", "LMN1"])
    assert rep.n == 3
    assert rep.exact_match_accuracy == pytest.approx(1 / 3)
    assert rep.character_accuracy == pytest.approx(1 - 5 / 15)
    with pytest.raises(ValueError):
        recognition_report(["a"], [])
