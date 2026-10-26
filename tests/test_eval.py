import numpy as np
import pandas as pd
import pytest

from costrouter.cascade.checks import CheckSignals, self_consistency
from costrouter.cascade.policy import check_quality, hybrid, hybrid_grid, pareto_front
from costrouter.eval.metrics import Outcomes, baselines, classifier_diagnostics, router_decisions
from costrouter.eval.stats import bootstrap_ci, mcnemar
from costrouter.eval.sweep import apgr, cpt, pick_operating_point, random_line, threshold_sweep


@pytest.fixture
def outcomes() -> Outcomes:
    rng = np.random.default_rng(0)
    n = 400
    difficulty = rng.random(n)
    small_ok = difficulty < 0.55
    large_ok = difficulty < 0.9
    return Outcomes(
        small_ok,
        large_ok,
        np.full(n, 0.0001),
        np.full(n, 0.001),
        np.full(n, 500.0),
        np.full(n, 1500.0),
    )


def test_baselines(outcomes):
    b = baselines(outcomes)
    assert b["always_large"]["pgr"] == pytest.approx(1.0)
    assert b["always_small"]["pgr"] == pytest.approx(0.0)
    assert b["always_large"]["cost_saved"] == pytest.approx(0.0)
    assert b["always_small"]["cost_saved"] == pytest.approx(0.9)
    assert b["oracle"]["hard_misroute"] == 0.0
    assert b["oracle"]["accuracy"] >= b["always_large"]["accuracy"]


def test_sweep_is_monotonic_and_spans_both_ends(outcomes):
    p = np.random.default_rng(1).random(len(outcomes))
    sw = threshold_sweep(p, outcomes)
    assert sw["pct_large"].is_monotonic_increasing
    assert sw["pct_large"].iloc[0] == pytest.approx(0.0, abs=0.01)
    assert sw["pct_large"].iloc[-1] == pytest.approx(1.0)


def test_random_router_apgr_is_about_half(outcomes):
    random_sweep = random_line(outcomes)
    assert apgr(random_sweep, outcomes) == pytest.approx(0.5, abs=0.02)


def test_perfect_router_beats_random(outcomes):
    perfect_p = outcomes.small_ok.astype(float) + 0.01 * np.random.default_rng(2).random(
        len(outcomes)
    )
    sw = threshold_sweep(perfect_p, outcomes)
    # 45% of queries are beyond the small model, so even a perfect router needs ~45% of calls
    # to large to recover the full gap: PGR climbs to 1 by x = 0.45, so APGR is about 0.775.
    assert apgr(sw, outcomes) > 0.7
    assert cpt(sw, outcomes, 0.8) <= 0.45


def test_operating_point_respects_accuracy_budget(outcomes):
    p = outcomes.small_ok.astype(float) * 0.6 + 0.4 * np.random.default_rng(3).random(len(outcomes))
    sw = threshold_sweep(p, outcomes)
    tau = pick_operating_point(sw, outcomes, max_drop_pts=2.0)
    res = router_decisions(outcomes, p >= tau)
    assert res["delta_vs_large_pts"] >= -2.0 - 1e-9
    assert res["cost_saved"] > 0


def test_cascade_extremes_match_simple_policies(outcomes):
    n = len(outcomes)
    p = np.random.default_rng(4).random(n)
    checks = CheckSignals(np.ones(n, bool), np.ones(n), np.zeros(n), np.zeros(n))
    tau = 0.5
    assert hybrid(outcomes, checks, p, tau, tau)["accuracy"] == pytest.approx(
        router_decisions(outcomes, p >= tau)["accuracy"]
    )
    # A check that always passes and costs nothing: pure cascade == always small.
    only = hybrid(outcomes, checks, p, -np.inf, np.inf)
    assert only["accuracy"] == pytest.approx(outcomes.small_ok.mean())
    # A perfect check: pure cascade reaches oracle-like accuracy and pays for both on escalation.
    perfect = CheckSignals(outcomes.small_ok, np.ones(n), np.zeros(n), np.zeros(n))
    res = hybrid(outcomes, perfect, p, -np.inf, np.inf)
    assert res["accuracy"] == pytest.approx(
        np.where(outcomes.small_ok, True, outcomes.large_ok).mean()
    )
    assert res["hard_misroute"] == 0.0
    assert check_quality(outcomes, perfect)["precision"] == pytest.approx(1.0)


def test_pareto_front_is_non_dominated(outcomes):
    n = len(outcomes)
    p = np.random.default_rng(5).random(n)
    checks = CheckSignals(outcomes.small_ok, np.ones(n), np.full(n, 0.00002), np.zeros(n))
    front = pareto_front(hybrid_grid(outcomes, checks, p, n=11))
    assert front["cost_saved"].is_monotonic_increasing
    assert front["accuracy"].is_monotonic_decreasing


def test_self_consistency_signal():
    rows = []
    for qid, main, samples in (
        ("q1", "4", ["4", "4"]),
        ("q2", "4", ["5", "6"]),
        ("q3", None, ["1", "1"]),
    ):
        rows.append({"query_id": qid, "role": "small", "call_type": "answer", "parsed": main,
                     "cost_usd": 0.0, "latency_ms": 100.0})  # fmt: skip
        for s in samples:
            rows.append({"query_id": qid, "role": "small", "call_type": "sc", "parsed": s,
                         "cost_usd": 0.001, "latency_ms": 200.0})  # fmt: skip
    sig = self_consistency(pd.DataFrame(rows), ["q1", "q2", "q3"])
    assert sig.passed.tolist() == [True, False, False]
    assert sig.cost.tolist() == pytest.approx([0.002, 0.002, 0.002])


def test_classifier_diagnostics():
    y = np.array([0, 0, 1, 1])
    d = classifier_diagnostics(np.array([0.1, 0.2, 0.8, 0.9]), y)
    assert d["roc_auc"] == 1.0 and d["brier"] < 0.05


def test_stats():
    a = np.array([True] * 30 + [False] * 10)
    b = np.array([True] * 20 + [False] * 20)
    assert mcnemar(a, b)["p_value"] < 0.01
    ci = bootstrap_ci(lambda ix: {"m": a[ix].mean()}, len(a), ["m"], n_boot=200)
    assert ci["m"][0] < a.mean() < ci["m"][1]
