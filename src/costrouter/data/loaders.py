"""Load GSM8K and MMLU from the Hugging Face Hub into the unified Query schema."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from costrouter.config import DatasetsConfig
from costrouter.data.prompts import LETTERS, render_gsm8k, render_mmlu
from costrouter.data.schema import Query


def gsm8k_gold(answer: str) -> str:
    """GSM8K solutions end with '#### <number>'."""
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_row_to_query(row: Mapping[str, Any], idx: int, split: str) -> Query:
    return Query(
        id=f"gsm8k-{split}-{idx}",
        dataset="gsm8k",
        subject=None,
        question=row["question"],
        prompt=render_gsm8k(row["question"]),
        gold=gsm8k_gold(row["answer"]),
    )


def mmlu_row_to_query(row: Mapping[str, Any], idx: int) -> Query:
    return Query(
        id=f"mmlu-test-{idx}",
        dataset="mmlu",
        subject=row["subject"],
        question=row["question"],
        prompt=render_mmlu(row["question"], list(row["choices"])),
        gold=LETTERS[int(row["answer"])],
    )


def load_gsm8k(split: str) -> list[Query]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [gsm8k_row_to_query(row, i, split) for i, row in enumerate(ds)]


def load_mmlu() -> list[Query]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    return [mmlu_row_to_query(row, i) for i, row in enumerate(ds)]


def sample_per_subject(queries: Iterable[Query], per_subject: int, rng: random.Random) -> list[Query]:
    by_subject: dict[str | None, list[Query]] = defaultdict(list)
    for q in queries:
        by_subject[q.subject].append(q)
    picked: list[Query] = []
    for subject in sorted(by_subject, key=str):
        group = by_subject[subject]
        picked.extend(rng.sample(group, min(per_subject, len(group))))
    return picked


def build_queries(cfg: DatasetsConfig) -> list[Query]:
    """Sample the working query set described in configs/datasets.yaml."""
    rng = random.Random(cfg.seed)
    gsm_test = load_gsm8k("test")
    if cfg.gsm8k_test != "all":
        gsm_test = rng.sample(gsm_test, min(cfg.gsm8k_test, len(gsm_test)))
    gsm_train = rng.sample(load_gsm8k("train"), cfg.gsm8k_train) if cfg.gsm8k_train else []
    mmlu = sample_per_subject(load_mmlu(), cfg.mmlu_per_subject, rng)
    return gsm_test + gsm_train + mmlu
