"""Three-zone hybrid policy: clearly hard -> large; clearly easy -> small, unchecked; the
uncertain middle band -> small plus a self-check, escalating to large when the check fails."""

from __future__ import annotations

import numpy as np
import pandas as pd

from costrouter.cascade.checks import CheckSignals
from costrouter.eval.metrics import Outcomes, summarize


def hybrid(
    o: Outcomes,
    checks: CheckSignals,
    p: np.ndarray,
    tau_low: float,
    tau_high: float,
    router_ms: float = 0.0,
) -> dict[str, float]:
    """tau_low = tau_high is the plain router; tau_low = -inf, tau_high = +inf is the pure
    cascade (always small first, always checked)."""
    direct_large = p < tau_low
    direct_small = p >= tau_high
    middle = ~direct_large & ~direct_small
    escalated = middle & ~checks.passed
    served_small = direct_small | (middle & checks.passed)

    cost = np.where(direct_large, o.large_cost, o.small_cost)
    cost = cost + np.where(middle, checks.cost, 0.0) + np.where(escalated, o.large_cost, 0.0)
    latency = np.where(direct_large, o.large_latency, o.small_latency) + router_ms
    latency = latency + np.where(middle, checks.latency, 0.0)
    latency = latency + np.where(escalated, o.large_latency, 0.0)

    out = summarize(o, served_small, cost, latency)
    out.update(
        {
            "tau_low": float(tau_low),
            "tau_high": float(tau_high),
            "pct_checked": float(middle.mean()),
            "pct_escalated": float(escalated.mean()),
        }
    )
    return out


def hybrid_grid(
    o: Outcomes, checks: CheckSignals, p: np.ndarray, n: int = 21, router_ms: float = 0.0
) -> pd.DataFrame:
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n)))
    grid = np.concatenate([[-np.inf], qs, [np.inf]])
    rows = [hybrid(o, checks, p, lo, hi, router_ms) for i, lo in enumerate(grid) for hi in grid[i:]]
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Policies no other policy beats on both cost saved and accuracy."""
    ordered = df.sort_values(["cost_saved", "accuracy"], ascending=[False, False])
    best, keep = -np.inf, []
    for idx, row in ordered.iterrows():
        if row["accuracy"] > best + 1e-12:
            keep.append(idx)
            best = row["accuracy"]
    return df.loc[keep].sort_values("cost_saved", ignore_index=True)


def check_quality(o: Outcomes, checks: CheckSignals) -> dict[str, float]:
    """How well 'check failed' detects 'small model was wrong'."""
    failed, wrong = ~checks.passed, ~o.small_ok
    tp = (failed & wrong).sum()
    return {
        "precision": float(tp / max(failed.sum(), 1)),
        "recall": float(tp / max(wrong.sum(), 1)),
        "fail_rate": float(failed.mean()),
    }


def pick_hybrid(grid: pd.DataFrame, acc_large: float, max_drop_pts: float = 2.0) -> pd.Series:
    ok = grid[grid["accuracy"] >= acc_large - max_drop_pts / 100 - 1e-12]
    if ok.empty:
        ok = grid
    return ok.sort_values(["cost_saved", "accuracy"], ascending=False).iloc[0]
