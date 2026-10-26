"""The unified query record shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Dataset = Literal["gsm8k", "mmlu"]


@dataclass(frozen=True)
class Query:
    id: str
    dataset: Dataset
    subject: str | None
    question: str  # raw question text; router features are computed from this only
    prompt: str  # question rendered with the versioned prompt template
    gold: str  # "42" for GSM8K, "C" for MMLU


def save_queries(queries: list[Query], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(q) for q in queries]).to_parquet(path, index=False)


def load_queries(path: Path) -> list[Query]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `router data prepare` first")
    df = pd.read_parquet(path)
    df["subject"] = df["subject"].astype(object).where(df["subject"].notna(), None)
    return [Query(**row) for row in df.to_dict(orient="records")]
