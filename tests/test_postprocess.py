import pytest

from lpr.postprocess import PlateFormatter

TEMPLATES = ["LLLDDDD", "LLLDDD", "LLDDDD"]


def fmt(**kw):
    return PlateFormatter(TEMPLATES, ["PUNJAB", "SINDH", "ISLAMABAD"], **kw)


def test_exact_match_no_changes():
    r = fmt().format("LEB 1234")
    assert (r.text, r.valid, r.template, r.substitutions) == ("LEB1234", True, "LLLDDDD", 0)


def test_confusable_characters_fixed_by_position():
    # 0 in a letter slot -> O, and O/S/B in digit slots -> 0/5/8
    r = fmt().format("LE8 12S4")
    assert (r.text, r.valid, r.substitutions) == ("LEB1254", True, 2)
    # more corrections than allowed -> kept as read, flagged invalid, never cut short
    r = fmt().format("L0B 1O5B")
    assert not r.valid and r.text == "L0B1O5B"
    r = fmt(max_substitutions=4).format("L0B 1O5B")
    assert r.text == "LOB1058" and r.valid


def test_province_words_dropped():
    r = fmt().format("PUNJAB LEB 1234")
    assert r.text == "LEB1234" and r.valid
    assert fmt().format(["SINDH", "KHI", "4567"]).text == "KHI4567"


def test_stray_edge_character_trimmed():
    r = fmt().format("ILEB1234")  # a bolt read as "I"
    assert r.text == "LEB1234" and r.valid and r.trimmed == 1


def test_invalid_kept_but_flagged():
    r = fmt().format("12")
    assert not r.valid and r.text == "12"


def test_no_templates_means_clean_only():
    r = PlateFormatter().format("ab-12")
    assert r.text == "AB12" and r.valid


def test_bad_template_rejected():
    with pytest.raises(ValueError):
        PlateFormatter(["LLX"])
