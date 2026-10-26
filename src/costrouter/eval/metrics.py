"""Metrics for a set of routing decisions, replayed against cached answers of both models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Outcomes:
    """Cached results of both models on the same queries (one entry per query)."""

    small_ok: np.ndarray  # bool
    large_ok: np.ndarray  # bool
    small_cost: np.ndarray  # USD
    large_cost: np.ndarray
    small_latency: np.ndarray  # ms
    large_latency: np.ndarray

    @classmethod
    def from_labels(cls, labels: pd.DataFrame) -> Outcomes:
        return cls(
            labels["small_correct"].to_numpy(bool),
            labels["large_correct"].to_numpy(bool),
            labels["small_cost_usd"].to_numpy(float),
            labels["large_cost_usd"].to_numpy(float),
            labels["small_latency_ms"].to_numpy(float),
            labels["large_latency_ms"].to_numpy(float),
        )

    def __len__(self) -> int:
        return len(self.small_ok)

    def take(self, idx: np.ndarray) -> Outcomes:
        return Outcomes(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))

    @property
    def hard(self) -> np.ndarray:
        """Small wrong, large right: the queries that must go to the large model."""
        return ~self.small_ok & self.large_ok


def summarize(
    o: Outcomes,
    served_small: np.ndarray,
    cost: np.ndarray,
    latency: np.ndarray,
) -> dict[str, float]:
    """served_small: which model's answer the user finally got. cost / latency: what each
    query actually cost under the policy (a cascade pays both models on escalation)."""
    correct = np.where(served_small, o.small_ok, o.large_ok)
    acc_s, acc_l = o.small_ok.mean(), o.large_ok.mean()
    hard, easy = o.hard, o.small_ok
    gap = acc_l - acc_s
    return {
        "accuracy": float(correct.mean()),
        "delta_vs_large_pts": float(100 * (correct.mean() - acc_l)),
        "cost_usd": float(cost.sum()),
        "cost_saved": float(1 - cost.sum() / o.large_cost.sum()),
        "pct_large": float(1 - served_small.mean()),
        "hard_misroute": float((served_small & hard).sum() / max(hard.sum(), 1)),
        "over_escalation": float((~served_small & easy).sum() / max(easy.sum(), 1)),
        "pgr": float((correct.mean() - acc_s) / gap) if gap > 0 else float("nan"),
        "latency_p50_ms": float(np.percentile(latency, 50)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "latency_mean_ms": float(latency.mean()),
    }


def router_decisions(o: Outcomes, to_small: np.ndarray, router_ms: float = 0.0) -> dict[str, float]:
    """A plain router: each query costs exactly one call to the chosen model."""
    cost = np.where(to_small, o.small_cost, o.large_cost)
    latency = np.where(to_small, o.small_latency, o.large_latency) + router_ms
    return summarize(o, to_small, cost, latency)


def baselines(o: Outcomes) -> dict[str, dict[str, float]]:
    n = len(o)
    oracle = o.small_ok | ~o.large_ok  # small when it is right, or when both are wrong
    return {
        "always_small": router_decisions(o, np.ones(n, bool)),
        "always_large": router_decisions(o, np.zeros(n, bool)),
        "oracle": router_decisions(o, oracle),
    }


def classifier_diagnostics(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y = y.astype(int)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = sum(
        abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
        for b in range(bins)
        if (idx == b).any()
    )
    auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")
    return {"roc_auc": float(auc), "brier": float(brier_score_loss(y, p)), "ece": float(ece)}
