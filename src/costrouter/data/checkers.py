"""Deterministic answer extraction and grading against the gold key."""

from __future__ import annotations

import re

from costrouter.data.schema import Dataset

# "Answer: 42", "**Answer:** $1,234.50", "answer: -3"
_ANSWER_NUMBER = re.compile(r"answer\s*[:：]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)\s*\}")

# "Answer: C", "Answer: (c)", "**Answer: B**"
_ANSWER_LETTER = re.compile(r"answer\s*[:：]?\s*\**\s*\(?([ABCD])\)?(?![A-Za-z])", re.IGNORECASE)
# Fallbacks on the last line only: "(C)", "C.", "C)", or a bare "C" ...
_LAST_LINE_LETTER = re.compile(r"^\W*\(?([ABCD])\)?[.)]?\W*$")
# ... or the chosen option restated: "D. Hepatitis C", "(B) adhesion plaques"
_LAST_LINE_OPTION = re.compile(r"^\W*\(?([ABCD])[.)]\s+\S")


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    """The model's final numeric answer, or None if it gave none."""
    for pattern in (_ANSWER_NUMBER, _BOXED):
        matches = pattern.findall(text)
        if matches:
            return _to_float(matches[-1])
    numbers = _ANY_NUMBER.findall(text)
    return _to_float(numbers[-1]) if numbers else None


def extract_choice(text: str) -> str | None:
    """The model's final letter choice (A-D), or None if it gave none."""
    matches = _ANSWER_LETTER.findall(text)
    if matches:
        return matches[-1].upper()
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if lines:
        last = lines[-1].strip()
        m = _LAST_LINE_LETTER.match(last) or _LAST_LINE_OPTION.match(last)
        if m:
            return m.group(1)
    return None


def grade(dataset: Dataset, text: str, gold: str) -> tuple[str | None, bool]:
    """Return (parsed answer, is_correct)."""
    if dataset == "gsm8k":
        pred = extract_number(text)
        gold_value = _to_float(gold)
        parsed = None if pred is None else _format_number(pred)
        correct = pred is not None and gold_value is not None and abs(pred - gold_value) < 1e-6
        return parsed, correct
    choice = extract_choice(text)
    return choice, choice is not None and choice == gold.strip().upper()


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
