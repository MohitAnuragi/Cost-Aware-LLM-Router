"""Uncertainty for the headline numbers: bootstrap intervals and McNemar's test."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    stat: Callable[[np.ndarray], dict[str, float]],
    n: int,
    keys: list[str],
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals: `stat` gets resampled row indices and returns a metric dict."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        values = stat(rng.integers(0, n, n))
        for k in keys:
            draws[k].append(values[k])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
        for k, v in draws.items()
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired correctness: do A and B differ in accuracy?"""
    from scipy.stats import binomtest

    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_value": float(p)}
