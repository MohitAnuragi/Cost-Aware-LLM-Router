import pytest

from costrouter.data.checkers import extract_choice, extract_number, grade


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She has 3 + 4 = 7 apples.\nAnswer: 7", 7.0),
        ("Answer: 1,234", 1234.0),
        ("**Answer:** $18.50", 18.5),
        ("Answer: -3", -3.0),
        ("The total is 72.\nAnswer: 72.", 72.0),
        ("answer: 5\nWait, recompute.\nAnswer: 6", 6.0),
        ("So the result is \\boxed{42}", 42.0),
        ("First 10, then 20, so 30 in total", 30.0),
        ("I cannot solve this.", None),
    ],
)
def test_extract_number(text, expected):
    assert extract_number(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Paris is the capital.\nAnswer: C", "C"),
        ("Answer: (b)", "B"),
        ("**Answer: D**", "D"),
        ("Answer: A.", "A"),
        ("Option B seems right, but\nAnswer: C", "C"),
        ("The best choice is\n(B)", "B"),
        ("D", "D"),
        # Real allam-2-7b outputs: the chosen option restated, with no "Answer:" line
        ("D. Hepatitis C ", "D"),
        ('C. it would lead to a "tragedy of the commons." ', "C"),
        ("(B) adhesion plaques", "B"),
        ("Answer: Because the force doubles, it is C", None),
        ("A. is wrong because of X.\nB. is wrong too.\nSo I am unsure", None),
        ("I am not sure about any of these.", None),
    ],
)
def test_extract_choice(text, expected):
    assert extract_choice(text) == expected


def test_grade_gsm8k():
    assert grade("gsm8k", "Answer: 1,000", "1000") == ("1000", True)
    assert grade("gsm8k", "Answer: 999", "1000") == ("999", False)
    assert grade("gsm8k", "no idea", "1000") == (None, False)
    assert grade("gsm8k", "Answer: 2.5", "2.5") == ("2.5", True)


def test_grade_mmlu():
    assert grade("mmlu", "Answer: b", "B") == ("B", True)
    assert grade("mmlu", "Answer: A", "B") == ("A", False)
    assert grade("mmlu", "none of them", "B") == (None, False)
