"""Threshold sweeps and the RouteLLM curve metrics (APGR, CPT), all offline from the cache."""

from __future__ import annotations

import numpy as np
import pandas as pd

from costrouter.eval.metrics import Outcomes, router_decisions


def thresholds(p: np.ndarray, n: int = 101) -> np.ndarray:
    """Quantiles of p (so each point is 'x% of queries go to large') plus +inf = all large."""
    return np.append(np.unique(np.quantile(p, np.linspace(0, 1, n))), np.inf)


def threshold_sweep(
    p: np.ndarray, o: Outcomes, taus: np.ndarray | None = None, router_ms: float = 0.0
) -> pd.DataFrame:
    taus = thresholds(p) if taus is None else taus
    rows = [{"tau": float(t), **router_decisions(o, p >= t, router_ms)} for t in taus]
    return pd.DataFrame(rows).sort_values("pct_large", ignore_index=True)


def random_line(o: Outcomes, n: int = 101) -> pd.DataFrame:
    """Expected result of routing a random x% of queries to large: a straight line."""
    rows = []
    for x in np.linspace(0, 1, n):
        acc = (1 - x) * o.small_ok.mean() + x * o.large_ok.mean()
        cost = (1 - x) * o.small_cost.sum() + x * o.large_cost.sum()
        rows.append(
            {
                "pct_large": x,
                "accuracy": acc,
                "cost_saved": 1 - cost / o.large_cost.sum(),
                "hard_misroute": 1 - x,
            }
        )
    return pd.DataFrame(rows)


def _best_accuracy_within(sweep: pd.DataFrame, budgets: np.ndarray) -> np.ndarray:
    """Upper envelope: the best accuracy reachable with at most x% of calls to large."""
    pct, acc = sweep["pct_large"].to_numpy(), sweep["accuracy"].to_numpy()
    return np.array(
        [acc[pct <= b + 1e-9].max() if (pct <= b + 1e-9).any() else np.nan for b in budgets]
    )


def pgr_curve(sweep: pd.DataFrame, o: Outcomes, n: int = 101) -> pd.DataFrame:
    budgets = np.linspace(0, 1, n)
    acc = _best_accuracy_within(sweep, budgets)
    gap = o.large_ok.mean() - o.small_ok.mean()
    pgr = (acc - o.small_ok.mean()) / gap if gap > 0 else np.full_like(acc, np.nan)
    return pd.DataFrame({"pct_large": budgets, "accuracy": acc, "pgr": pgr})


def apgr(sweep: pd.DataFrame, o: Outcomes) -> float:
    """Area under the PGR curve over 0-100% large calls (random = 0.5, as in RouteLLM)."""
    curve = pgr_curve(sweep, o).dropna()
    x, y = curve["pct_large"].to_numpy(), np.clip(curve["pgr"].to_numpy(), None, 1.0)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def cpt(sweep: pd.DataFrame, o: Outcomes, target: float) -> float:
    """Smallest share of calls to large that reaches `target` PGR (e.g. 0.5 for CPT(50%))."""
    curve = pgr_curve(sweep, o).dropna()
    hit = curve[curve["pgr"] >= target - 1e-9]
    return float(hit["pct_large"].min()) if len(hit) else float("nan")


def pick_operating_point(sweep: pd.DataFrame, o: Outcomes, max_drop_pts: float = 2.0) -> float:
    """The tau with the largest cost saving whose accuracy is within max_drop_pts of
    always-large. Chosen on validation, then frozen for the single test-set evaluation."""
    ok = sweep[sweep["accuracy"] >= o.large_ok.mean() - max_drop_pts / 100 - 1e-12]
    if ok.empty:
        return float("inf")
    return float(ok.sort_values(["cost_saved", "accuracy"], ascending=False).iloc[0]["tau"])


def tau_for_budget(p: np.ndarray, pct_large: float) -> float:
    """Threshold that sends about pct_large of these queries to the large model."""
    return float(np.quantile(p, pct_large)) if pct_large < 1 else float("inf")
