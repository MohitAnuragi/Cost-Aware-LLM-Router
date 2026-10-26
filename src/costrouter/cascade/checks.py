"""Self-check signals for the small model's answer, computed from cached calls only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CheckSignals:
    """Per query, aligned with the label table rows."""

    passed: np.ndarray  # bool: the small model's answer survives the check
    agreement: np.ndarray  # share of extra samples that agree with the main answer
    cost: np.ndarray  # USD spent on the extra samples
    latency: np.ndarray  # ms; samples run in parallel, so the slowest one counts

    def take(self, idx: np.ndarray) -> CheckSignals:
        return CheckSignals(
            self.passed[idx], self.agreement[idx], self.cost[idx], self.latency[idx]
        )


def self_consistency(responses: pd.DataFrame, query_ids: list[str]) -> CheckSignals:
    """Parse check + self-consistency: pass when the main answer parsed and at least 2 of the
    3 answers (main + 2 samples) agree with it. Queries without samples always pass at no cost,
    so callers should restrict to queries that have them (see has_samples)."""
    small = responses[responses["role"] == "small"]
    main = small[small["call_type"] == "answer"].set_index("query_id")
    samples = small[small["call_type"] == "sc"]
    grouped = {qid: g for qid, g in samples.groupby("query_id")}

    passed, agreement, cost, latency = [], [], [], []
    for qid in query_ids:
        answer = main.at[qid, "parsed"] if qid in main.index else None
        g = grouped.get(qid)
        if g is None or g.empty:
            passed.append(answer is not None)
            agreement.append(1.0)
            cost.append(0.0)
            latency.append(0.0)
            continue
        agree = float((g["parsed"] == answer).mean()) if answer is not None else 0.0
        passed.append(answer is not None and agree >= 0.5)
        agreement.append(agree)
        cost.append(float(g["cost_usd"].sum()))
        latency.append(float(g["latency_ms"].max()))
    return CheckSignals(
        np.array(passed, bool), np.array(agreement), np.array(cost), np.array(latency)
    )


def has_samples(responses: pd.DataFrame, query_ids: list[str], k: int) -> np.ndarray:
    sc = responses[(responses["role"] == "small") & (responses["call_type"] == "sc")]
    counts = sc.groupby("query_id").size()
    return np.array([counts.get(q, 0) >= k for q in query_ids], bool)
