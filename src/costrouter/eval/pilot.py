"""Pilot report: is the accuracy gap big enough to route on, and what will the full run cost?"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from costrouter.config import ROLES, Profile

MIN_GAP_PTS = 15.0


def summarize(responses: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and role: accuracy, parse failures, tokens, latency and cost of main answers."""
    answers = responses[responses["call_type"] == "answer"].copy()
    answers["parse_failed"] = answers["parsed"].isna()
    return (
        answers.groupby(["dataset", "role"])
        .agg(
            n=("query_id", "count"),
            accuracy=("correct", "mean"),
            parse_fail_rate=("parse_failed", "mean"),
            mean_in_tokens=("input_tokens", "mean"),
            mean_out_tokens=("output_tokens", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_cost_usd=("cost_usd", "mean"),
        )
        .reset_index()
    )


def accuracy_gaps(labels: pd.DataFrame) -> dict[str, float]:
    """Large minus small accuracy in points, per dataset and overall."""
    gaps = {
        str(ds): 100 * (g["large_correct"].mean() - g["small_correct"].mean())
        for ds, g in labels.groupby("dataset")
    }
    gaps["overall"] = 100 * (labels["large_correct"].mean() - labels["small_correct"].mean())
    return gaps


def outcome_counts(labels: pd.DataFrame) -> dict[str, int]:
    s, lg = labels["small_correct"], labels["large_correct"]
    return {
        "both_correct (easy)": int((s & lg).sum()),
        "only_large (hard)": int((~s & lg).sum()),
        "only_small": int((s & ~lg).sum()),
        "both_wrong": int((~s & ~lg).sum()),
    }


def project_full_run(
    summary: pd.DataFrame,
    profile: Profile,
    full_counts: Mapping[str, int],
    self_consistency: int,
) -> pd.DataFrame:
    """Scale pilot means to the full query set, per role, under each model's rate limits."""
    rows = []
    for role in ROLES:
        cfg = profile.role(role)
        calls_per_query = 1 + (self_consistency if role == "small" else 0)
        requests = tokens = cost = 0.0
        for _, row in summary[summary["role"] == role].iterrows():
            n = full_counts.get(row["dataset"], 0) * calls_per_query
            requests += n
            tokens += n * (row["mean_in_tokens"] + row["mean_out_tokens"])
            cost += n * row["mean_cost_usd"]
        minutes = max(
            requests / cfg.rpm if cfg.rpm else 0.0,
            tokens / cfg.tpm if cfg.tpm else 0.0,
        )
        days = max(
            requests / cfg.rpd if cfg.rpd else 0.0,
            tokens / cfg.tpd if cfg.tpd else 0.0,
        )
        rows.append(
            {
                "role": role,
                "model": cfg.name,
                "requests": int(requests),
                "tokens": int(tokens),
                "hours_at_rate_limit": minutes / 60,
                "days_at_daily_limit": days if (cfg.rpd or cfg.tpd) else None,
                "list_cost_usd": cost,
            }
        )
    return pd.DataFrame(rows)


def build_pilot_report(
    responses: pd.DataFrame,
    labels: pd.DataFrame,
    profile: Profile,
    full_counts: Mapping[str, int],
    self_consistency: int = 2,
) -> dict[str, Any]:
    summary = summarize(responses)
    gaps = accuracy_gaps(labels)
    return {
        "summary": summary,
        "gaps_pts": gaps,
        "outcomes": outcome_counts(labels),
        "projection": project_full_run(summary, profile, full_counts, self_consistency),
        "gate_passed": bool(gaps["overall"] >= MIN_GAP_PTS),
        "min_gap_pts": MIN_GAP_PTS,
    }
