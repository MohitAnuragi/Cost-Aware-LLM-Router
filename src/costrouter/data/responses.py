"""Turn cached responses into graded tables: one row per call, and one label row per query."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from costrouter.config import ModelConfig, Profile
from costrouter.data.checkers import grade
from costrouter.eval.cost import cost_usd
from costrouter.data.prompts import TEMPLATE_VERSION
from costrouter.data.schema import Query
from costrouter.llm.cache import ResponseCache

RESPONSE_COLUMNS = [
    "query_id", "dataset", "subject", "role", "model", "call_type", "sample_idx", "text",
    "parsed", "correct", "gold", "input_tokens", "output_tokens", "latency_ms", "cost_usd",
    "stop_reason",
]


def graded_records(
    cache: ResponseCache,
    queries: list[Query],
    models: list[str],
    prices: Mapping[str, ModelConfig] | None = None,
) -> pd.DataFrame:
    """Every cached call by these models on these queries, graded against the gold key.

    Grading and pricing happen here, at read time: a checker fix re-grades everything and a
    price change re-prices everything, with no new API calls. `prices` maps model name to
    the config whose list price to apply; without it the price stored at call time is kept."""
    records = cache.records(models, TEMPLATE_VERSION)
    columns = [c for c in RESPONSE_COLUMNS if c != "role"]
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records).drop(columns=["role"])
    qdf = pd.DataFrame(
        [{"query_id": q.id, "dataset": q.dataset, "subject": q.subject, "gold": q.gold} for q in queries]
    )
    df = df.merge(qdf, on="query_id", how="inner")
    graded = [grade(d, t, g) for d, t, g in zip(df["dataset"], df["text"], df["gold"], strict=True)]
    df["parsed"] = [p for p, _ in graded]
    df["correct"] = [c for _, c in graded]
    for name, cfg in (prices or {}).items():
        rows = df["model"] == name
        df.loc[rows, "cost_usd"] = [
            cost_usd(cfg, int(i), int(o))
            for i, o in zip(df.loc[rows, "input_tokens"], df.loc[rows, "output_tokens"], strict=True)
        ]
    return df[columns]


def response_table(cache: ResponseCache, queries: list[Query], profile: Profile) -> pd.DataFrame:
    """One graded row per call by the profile's two models, tagged small / large."""
    role_by_model = {profile.small.name: "small", profile.large.name: "large"}
    prices = {profile.small.name: profile.small, profile.large.name: profile.large}
    df = graded_records(cache, queries, list(role_by_model), prices)
    df.insert(3, "role", df["model"].map(role_by_model))
    return df[RESPONSE_COLUMNS].sort_values(["query_id", "role", "call_type", "sample_idx"])


def label_table(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per query that has both models' main answer: the router's training labels."""
    answers = responses[responses["call_type"] == "answer"]
    # Pivot on query_id alone: GSM8K's subject is None, and pivot_table drops null keys.
    meta = answers[["query_id", "dataset", "subject"]].drop_duplicates("query_id")
    wide = answers.pivot_table(
        index="query_id",
        columns="role",
        values=["correct", "cost_usd", "latency_ms", "input_tokens", "output_tokens"],
        aggfunc="first",
    )
    wide.columns = [f"{role}_{metric}" for metric, role in wide.columns]
    wide = meta.merge(wide.reset_index(), on="query_id")
    for col in ("small_correct", "large_correct"):
        if col not in wide:
            wide[col] = pd.NA
    wide = wide.dropna(subset=["small_correct", "large_correct"])
    for col in ("small_correct", "large_correct"):
        wide[col] = wide[col].astype(bool)
    return wide
