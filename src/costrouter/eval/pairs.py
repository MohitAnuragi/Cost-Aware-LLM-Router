"""Compare every candidate small/large pair on the queries all models have answered.

The deciding numbers are the accuracy gap (is there anything to route on?) and the oracle's
cost saving (the most a perfect router could save at no accuracy loss)."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd


def pair_matrix(answers: pd.DataFrame) -> pd.DataFrame:
    """`answers`: graded rows for several models, as returned by graded_records()."""
    main = answers[answers["call_type"] == "answer"]
    correct = main.pivot_table(index="query_id", columns="model", values="correct", aggfunc="first")
    cost = main.pivot_table(index="query_id", columns="model", values="cost_usd", aggfunc="first")
    shared = correct.dropna().index.intersection(cost.dropna().index)
    correct, cost = correct.loc[shared].astype(bool), cost.loc[shared]

    rows = []
    for small, large in permutations(correct.columns, 2):
        if cost[small].mean() >= cost[large].mean():
            continue  # the small model must be the cheaper one
        s, lg = correct[small].to_numpy(), correct[large].to_numpy()
        cs, cl = cost[small].to_numpy(), cost[large].to_numpy()
        to_small = s | ~lg  # oracle: small when it is right, or when both are wrong
        rows.append(
            {
                "small": small,
                "large": large,
                "n": len(shared),
                "small_acc": s.mean(),
                "large_acc": lg.mean(),
                "gap_pts": 100 * (lg.mean() - s.mean()),
                "hard_share": (~s & lg).mean(),
                "cost_ratio": cl.mean() / cs.mean(),
                "oracle_acc": np.where(to_small, s, lg).mean(),
                "oracle_cost_saved": 1 - np.where(to_small, cs, cl).sum() / cl.sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("oracle_cost_saved", ascending=False, ignore_index=True)
