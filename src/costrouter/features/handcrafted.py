"""Cheap, interpretable text features computed from the user's text only."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[A-Za-z']+")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
_MATH_OP = re.compile(r"[+\-*/=^×÷]")
_UNIT = re.compile(
    r"\b(?:km|kg|g|lb|lbs|miles?|meters?|feet|foot|inch(?:es)?|hours?|minutes?|seconds?|days?|"
    r"weeks?|months?|years?|dollars?|cents?|percent|liters?|gallons?|mph)\b",
    re.IGNORECASE,
)
_OPTION_LINE = re.compile(r"^\s*\(?[A-D][.)]\s", re.MULTILINE)
_HARD_WORDS = re.compile(
    r"\b(?:prove|probability|ratio|average|rate|percent(?:age)?|proportion|integral|derivative|"
    r"equation|interest|discount|remainder|fraction|twice|half|each|per|respectively|except)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:not|except|least|never|false|incorrect)\b", re.IGNORECASE)

FEATURE_NAMES = [
    "log_chars",
    "log_words",
    "n_sentences",
    "avg_word_len",
    "long_word_share",
    "n_numbers",
    "n_distinct_numbers",
    "has_decimal_or_fraction",
    "has_percent_or_currency",
    "n_math_ops",
    "n_units",
    "n_options",
    "n_hard_words",
    "has_negation",
    "starts_how_many_or_much",
    "n_question_marks",
    "flesch_reading_ease",
]


def _flesch(text: str) -> float:
    try:
        import textstat

        return float(textstat.flesch_reading_ease(text))
    except Exception:  # textstat is optional; a neutral value keeps the column usable
        return 60.0


def featurize(text: str) -> list[float]:
    words = _WORD.findall(text)
    numbers = _NUMBER.findall(text)
    n_words = max(len(words), 1)
    lower = text.lower().lstrip()
    return [
        float(np.log1p(len(text))),
        float(np.log1p(len(words))),
        float(max(len(_SENTENCE_END.findall(text)), 1)),
        sum(len(w) for w in words) / n_words,
        sum(len(w) >= 9 for w in words) / n_words,
        float(len(numbers)),
        float(len(set(numbers))),
        float(bool(re.search(r"\d[.]\d|\d/\d", text))),
        float(bool(re.search(r"[%$€£]|percent", text, re.IGNORECASE))),
        float(len(_MATH_OP.findall(text))),
        float(len(_UNIT.findall(text))),
        float(len(_OPTION_LINE.findall(text))),
        float(len(_HARD_WORDS.findall(text))),
        float(bool(_NEGATION.search(text))),
        float(lower.startswith(("how many", "how much"))),
        float(text.count("?")),
        _flesch(text),
    ]


def handcrafted_matrix(texts: Sequence[str]) -> np.ndarray:
    return np.asarray([featurize(t) for t in texts], dtype=np.float32)
