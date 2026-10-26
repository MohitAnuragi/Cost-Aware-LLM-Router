"""Versioned prompt templates. Bump TEMPLATE_VERSION whenever a template changes,
so cached responses from the old wording are never mixed with new ones."""

from __future__ import annotations

TEMPLATE_VERSION = "v1"

LETTERS = "ABCD"

GSM8K_TEMPLATE = """Solve the following math problem step by step. Keep the reasoning concise.
On the last line, write "Answer: " followed by only the final number.

Problem: {question}"""

MMLU_TEMPLATE = """Answer the following multiple choice question.

{question}

{choices}

Think briefly, then on the last line write "Answer: " followed by only the letter (A, B, C, or D)."""


def render_gsm8k(question: str) -> str:
    return GSM8K_TEMPLATE.format(question=question.strip())


def render_mmlu(question: str, choices: list[str]) -> str:
    lines = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices, strict=True))
    return MMLU_TEMPLATE.format(question=question.strip(), choices=lines)


def user_text(prompt: str) -> str:
    """What a real user would have typed: the prompt minus our fixed instructions.

    The router sees only this, both in training and when serving live traffic, so it never
    learns from the template wording (which would differ in production)."""
    head, sep, rest = prompt.partition("Problem: ")
    if sep and head.startswith("Solve the following math problem"):
        return rest.strip()
    lines = prompt.strip().splitlines()
    if lines and lines[0].startswith("Answer the following multiple choice question"):
        return "\n".join(lines[1:-1]).strip()
    return prompt.strip()
