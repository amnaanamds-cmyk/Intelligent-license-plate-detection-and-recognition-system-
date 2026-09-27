"""Plate-format validation and correction of commonly confused characters.

OCR engines regularly mix up characters that look alike, such as O/0, I/1,
S/5 and B/8. When the plate layout is known (for example three letters then
four digits), each position can only hold a letter or a digit, so most of
these mistakes can be fixed. Templates use ``L`` for a letter and ``D`` for
a digit.

Printed words that are not part of the registration number (province
names, "GOVT", ...) are removed line by line before matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ocr import clean_plate_text

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "T": "7", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G",
             "7": "T", "8": "B"}


@dataclass
class FormattedPlate:
    text: str
    valid: bool
    template: str | None = None
    substitutions: int = 0
    trimmed: int = 0


def _fit(candidate: str, template: str) -> tuple[str, int] | None:
    """Make ``candidate`` fit ``template``. Returns (text, number of
    substitutions), or None when a character cannot be converted."""
    out, subs = [], 0
    for ch, kind in zip(candidate, template):
        if kind == "L":
            if ch.isalpha():
                out.append(ch)
            elif ch in TO_LETTER:
                out.append(TO_LETTER[ch]); subs += 1
            else:
                return None
        elif kind == "D":
            if ch.isdigit():
                out.append(ch)
            elif ch in TO_DIGIT:
                out.append(TO_DIGIT[ch]); subs += 1
            else:
                return None
        else:  # any character
            out.append(ch)
    return "".join(out), subs


class PlateFormatter:
    def __init__(self, templates: list[str] | None = None,
                 ignore_words: list[str] | None = None,
                 max_substitutions: int = 2, max_trim: int = 2):
        self.templates = [t.upper() for t in (templates or [])]
        bad = [t for t in self.templates if set(t) - {"L", "D", "A"}]
        if bad:
            raise ValueError(f"templates may only contain L, D, A: {bad}")
        self.ignore_words = {clean_plate_text(w) for w in (ignore_words or []) if w}
        self.max_substitutions = max_substitutions
        self.max_trim = max_trim

    def _drop_ignored(self, lines: list[str]) -> list[str]:
        kept = []
        for line in lines:
            tokens = [clean_plate_text(t) for t in line.split()]
            tokens = [t for t in tokens if t and t not in self.ignore_words]
            if tokens and clean_plate_text("".join(tokens)) not in self.ignore_words:
                kept.append("".join(tokens))
        return kept

    def format(self, text_or_lines: str | list[str]) -> FormattedPlate:
        lines = ([text_or_lines] if isinstance(text_or_lines, str)
                 else list(text_or_lines))
        text = "".join(self._drop_ignored(lines))
        if not self.templates:
            return FormattedPlate(text, valid=bool(text))
        best: FormattedPlate | None = None
        # Also try with up to max_trim characters removed from either end,
        # since stray marks (bolts, frame edges) are sometimes read as letters.
        for trim in range(0, self.max_trim + 1):
            for left in range(trim + 1):
                right = trim - left
                cand = text[left:len(text) - right]
                for tpl in self.templates:
                    if len(cand) != len(tpl):
                        continue
                    fit = _fit(cand, tpl)
                    # Each trimmed character uses two corrections from the
                    # budget, so a correctly sized read is never cut down just
                    # to avoid substitutions.
                    if fit is None or fit[1] + 2 * trim > self.max_substitutions:
                        continue
                    res = FormattedPlate(fit[0], True, tpl, fit[1], trim)
                    # Fewer edits wins. On a tie the longer template wins.
                    key = (res.substitutions + 2 * res.trimmed, -len(tpl))
                    if best is None or key < (best.substitutions + 2 * best.trimmed,
                                              -len(best.template)):
                        best = res
            if best is not None:
                break
        return best or FormattedPlate(text, valid=False)

    @classmethod
    def from_config(cls, cfg: dict) -> "PlateFormatter":
        return cls(cfg.get("templates"), cfg.get("ignore_words"),
                   cfg.get("max_substitutions", 2))
