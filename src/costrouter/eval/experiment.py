"""The whole offline experiment, replayed from cached answers (no API calls):

features -> out-of-fold predictions on train + validation (compare routers, pick thresholds)
-> refit on train + validation -> one frozen evaluation on the test split -> self-check
cascade -> robustness checks -> figures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from costrouter.cascade.checks import has_samples, self_consistency
from costrouter.cascade.policy import check_quality, hybrid, hybrid_grid, pareto_front, pick_hybrid
from costrouter.config import ModelConfig
from costrouter.data.prompts import user_text
from costrouter.data.responses import label_table
from costrouter.data.schema import Query
from costrouter.data.splits import SPLITS
from costrouter.eval import plots
from costrouter.eval.cost import cost_usd
from costrouter.eval.metrics import Outcomes, baselines, classifier_diagnostics, router_decisions
from costrouter.eval.stats import bootstrap_ci, mcnemar
from costrouter.eval.sweep import (
    apgr,
    cpt,
    pgr_curve,
    pick_operating_point,
    random_line,
    tau_for_budget,
    threshold_sweep,
)
from costrouter.features import Features, build_features
from costrouter.routers.base import Router
from costrouter.routers.registry import all_routers

LEARNED = plots.LEARNED
CI_KEYS = ["accuracy", "delta_vs_large_pts", "cost_saved", "hard_misroute", "pct_large"]
MIN_PER_SPLIT = 20


class NotEnoughDataError(RuntimeError):
    pass


@dataclass
class ExperimentData:
    labels: pd.DataFrame
    responses: pd.DataFrame
    texts: list[str]
    features: Features
    outcomes: Outcomes
    idx: dict[str, np.ndarray]


def load_data(
    queries: list[Query],
    splits: dict[str, list[str]],
    responses: pd.DataFrame,
    embed_dir: Path | None,
) -> ExperimentData:
    labels = label_table(responses)
    split_of = {qid: s for s in SPLITS for qid in splits[s]}
    labels["split"] = labels["query_id"].map(split_of)
    labels = labels.dropna(subset=["split"]).reset_index(drop=True)
    prompt_of = {q.id: q.prompt for q in queries}
    texts = [user_text(prompt_of[qid]) for qid in labels["query_id"]]
    idx = {s: np.flatnonzero(labels["split"].to_numpy() == s) for s in SPLITS}
    short = {s: len(i) for s, i in idx.items() if len(i) < MIN_PER_SPLIT}
    if short:
        raise NotEnoughDataError(
            f"need at least {MIN_PER_SPLIT} labelled queries per split, have {short}; "
            "run `collect` again after the daily free-tier reset"
        )
    return ExperimentData(
        labels=labels,
        responses=responses,
        texts=texts,
        features=build_features(texts, cache_dir=embed_dir),
        outcomes=Outcomes.from_labels(labels),
        idx=idx,
    )


def _repriced(labels: pd.DataFrame, o: Outcomes, small: ModelConfig) -> Outcomes:
    cost = np.array(
        [
            cost_usd(small, int(i), int(t))
            for i, t in zip(
                labels["small_input_tokens"], labels["small_output_tokens"], strict=True
            )
        ]
    )
    return Outcomes(o.small_ok, o.large_ok, cost, o.large_cost, o.small_latency, o.large_latency)


def _router_latency_ms(router: Router, texts: list[str], n: int = 40) -> dict[str, float]:
    """Wall-clock cost of one routing decision: features (embedding included) + predict.

    Measured the way the gateway runs it: one query at a time on one CPU thread (for a single
    short text, one thread is faster and steadier than many). Other load on the machine
    inflates these numbers, so the CPU load at measurement time is recorded alongside."""
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        build_features(texts[:1])  # warm-up
        samples = []
        for text in texts[:n]:
            t0 = time.perf_counter()
            router.predict_proba(build_features([text]))
            samples.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(threads)
    return {
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "cpu_load_pct": _cpu_load_pct(),
    }


def _cpu_load_pct() -> float | None:
    """Whole-machine CPU load right now (Windows: WMI; elsewhere: 1-minute load average)."""
    import os
    import subprocess

    try:
        if os.name == "nt":
            cmd = "(Get-CimInstance Win32_Processor).LoadPercentage"
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout.split()
            return float(np.mean([float(v) for v in out])) if out else None
        return 100 * os.getloadavg()[0] / (os.cpu_count() or 1)
    except Exception:
        return None


def _make(name: str, seed: int) -> Router:
    return next(r for r in all_routers(seed) if r.name == name)


def _out_of_fold(data: ExperimentData, rows: np.ndarray, name: str, seed: int) -> np.ndarray:
    """5-fold out-of-fold P(small correct) for `rows`: every row is scored by a router that
    never saw it, so thresholds and router choice can be tuned on all non-test data."""
    y = data.outcomes.small_ok[rows]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros(len(rows))
    for fit_i, score_i in folds.split(rows, y):
        router = _make(name, seed).fit(data.features.take(rows[fit_i]), y[fit_i])
        out[score_i] = router.predict_proba(data.features.take(rows[score_i]))
    return out


def run_experiment(
    data: ExperimentData,
    out_dir: Path,
    models_dir: Path,
    *,
    seeds: tuple[int, ...] = (42, 1, 2),
    max_drop_pts: float = 2.0,
    conservative_small: ModelConfig | None = None,
    sc_samples: int = 2,
) -> dict[str, Any]:
    """Development set = train + validation. Routers are compared, and thresholds chosen, on
    5-fold out-of-fold predictions over it; the final routers are refit on all of it; the test
    split is scored once at the frozen thresholds."""
    o, F, idx = data.outcomes, data.features, data.idx
    dev, te = np.concatenate([idx["train"], idx["val"]]), idx["test"]
    od, ot = o.take(dev), o.take(te)
    y = o.small_ok
    fig_dir = out_dir / "figures"
    names = [r.name for r in all_routers(seeds[0])]

    # 1. Out-of-fold predictions on the development set; final routers refit on all of it.
    oof = {n: _out_of_fold(data, dev, n, seeds[0]) for n in names}
    routers = {n: _make(n, seeds[0]).fit(F.take(dev), y[dev]) for n in names}
    for name, r in routers.items():
        r.save(models_dir / f"{name}.joblib")
    p_test = {n: r.predict_proba(F.take(te)) for n, r in routers.items()}

    # 2. Per router: choose tau on the out-of-fold sweep, report the test sweep.
    per_router: dict[str, dict[str, Any]] = {}
    sweeps_test: dict[str, pd.DataFrame] = {}
    for name in names:
        sw_dev, sw_test = threshold_sweep(oof[name], od), threshold_sweep(p_test[name], ot)
        tau = pick_operating_point(sw_dev, od, max_drop_pts)
        sweeps_test[name] = sw_test
        per_router[name] = {
            "cv_diagnostics": classifier_diagnostics(oof[name], y[dev]),
            "cv_apgr": apgr(sw_dev, od),
            "test_apgr": apgr(sw_test, ot),
            "test_cpt50": cpt(sw_test, ot, 0.5),
            "test_cpt80": cpt(sw_test, ot, 0.8),
            "tau": tau,
            "cv_at_tau": router_decisions(od, oof[name] >= tau),
            "test_at_tau": router_decisions(ot, p_test[name] >= tau),
            "best_params": getattr(routers[name], "best_params_", {}),
        }
    best = max(LEARNED, key=lambda n: per_router[n]["cv_apgr"])
    best_router, tau = routers[best], per_router[best]["tau"]

    # 3. Frozen test evaluation of the best router (chosen on out-of-fold APGR).
    latency = _router_latency_ms(best_router, [data.texts[i] for i in te])
    to_small = p_test[best] >= tau
    headline = router_decisions(ot, to_small, latency["p50"])
    ci = bootstrap_ci(
        lambda ix: router_decisions(ot.take(ix), to_small[ix], latency["p50"]), len(te), CI_KEYS
    )
    served_correct = np.where(to_small, ot.small_ok, ot.large_ok)
    significance = mcnemar(served_correct, ot.large_ok)
    base = baselines(ot)

    # The whole trade-off, not one point: thresholds chosen on dev for several accuracy
    # budgets, each scored once on test.
    budgets = {}
    dev_sweep_best = threshold_sweep(oof[best], od)
    for drop in (1.0, 2.0, 5.0, 10.0):
        t = pick_operating_point(dev_sweep_best, od, drop)
        budgets[f"{drop:g}"] = {"tau": t, **router_decisions(ot, p_test[best] >= t)}

    # 4. Seed stability: refit with other seeds, report test APGR mean and spread.
    seed_apgr: dict[str, list[float]] = {n: [per_router[n]["test_apgr"]] for n in LEARNED}
    for seed in seeds[1:]:
        for n in LEARNED:
            refit = _make(n, seed).fit(F.take(dev), y[dev])
            seed_apgr[n].append(apgr(threshold_sweep(refit.predict_proba(F.take(te)), ot), ot))
    for n in LEARNED:
        per_router[n]["test_apgr_mean"] = float(np.mean(seed_apgr[n]))
        per_router[n]["test_apgr_std"] = float(np.std(seed_apgr[n]))

    # 5. Threshold presets for the API, from out-of-fold predictions.
    presets = {
        "quality": tau,
        "balanced": tau_for_budget(oof[best], 0.5),
        "budget_30": tau_for_budget(oof[best], 0.3),
        "budget_70": tau_for_budget(oof[best], 0.7),
    }

    # 6. Self-check cascade on queries that have the extra small-model samples.
    ids = data.labels["query_id"].tolist()
    checks = self_consistency(data.responses, ids)
    has = has_samples(data.responses, ids, sc_samples)
    cascade: dict[str, Any] = {"available": False}
    hybrid_front = None
    dev_mask, te_c = has[dev], te[has[te]]
    if dev_mask.sum() >= MIN_PER_SPLIT and len(te_c) >= MIN_PER_SPLIT:
        dev_c = dev[dev_mask]
        od_c, ot_c = o.take(dev_c), o.take(te_c)
        pd_c, pt = oof[best][dev_mask], best_router.predict_proba(F.take(te_c))
        grid_dev = hybrid_grid(od_c, checks.take(dev_c), pd_c, router_ms=latency["p50"])
        chosen = pick_hybrid(grid_dev, od_c.large_ok.mean(), max_drop_pts)
        ct = checks.take(te_c)
        hybrid_front = pareto_front(hybrid_grid(ot_c, ct, pt, router_ms=latency["p50"]))
        cascade = {
            "available": True,
            "n_dev": int(len(dev_c)),
            "n_test": int(len(te_c)),
            "covers_full_test": bool(len(te_c) == len(te)),
            "tau_low": float(chosen["tau_low"]),
            "tau_high": float(chosen["tau_high"]),
            "hybrid": hybrid(ot_c, ct, pt, chosen["tau_low"], chosen["tau_high"], latency["p50"]),
            "router_only": hybrid(ot_c, ct, pt, tau, tau, latency["p50"]),
            "cascade_only": hybrid(ot_c, ct, pt, -np.inf, np.inf, latency["p50"]),
            "always_large": router_decisions(ot_c, np.zeros(len(te_c), bool)),
            "check_quality": check_quality(ot_c, ct),
        }
        plots.cost_vs_accuracy(
            {"router_only": threshold_sweep(pt, ot_c)},
            random_line(ot_c),
            baselines(ot_c),
            cascade["hybrid"],
            "hybrid",
            fig_dir / "fig7_cascade.png",
            hybrid_front,
            subtitle=f"{len(te_c)} test queries with self-consistency samples",
        )

    # 7. Robustness: conservative price for the small model, per-dataset, generalisation.
    sensitivity = {}
    if conservative_small is not None:
        oc = _repriced(data.labels, o, conservative_small).take(te)
        sensitivity = {
            "small_price": (
                f"${conservative_small.usd_in_per_m} / ${conservative_small.usd_out_per_m}"
            ),
            "cost_saved": router_decisions(oc, to_small)["cost_saved"],
            "oracle_cost_saved": baselines(oc)["oracle"]["cost_saved"],
        }

    ds_test = data.labels["dataset"].to_numpy()[te]
    per_dataset, ds_sweeps, ds_base = {}, {}, {}
    for ds in sorted(set(ds_test)):
        m = ds_test == ds
        o_ds = ot.take(np.flatnonzero(m))
        ds_sweeps[ds] = threshold_sweep(p_test[best][m], o_ds)
        ds_base[ds] = baselines(o_ds)
        per_dataset[ds] = {
            "n": int(m.sum()),
            "at_tau": router_decisions(o_ds, to_small[m]),
            "always_small_acc": ds_base[ds]["always_small"]["accuracy"],
            "always_large_acc": ds_base[ds]["always_large"]["accuracy"],
            "apgr": apgr(ds_sweeps[ds], o_ds),
        }

    generalisation = _cross_dataset(data, best, seeds[0])

    # 8. Error analysis: hard queries the router most confidently sent to the small model.
    hard_to_small = np.flatnonzero(ot.hard & to_small)
    worst = hard_to_small[np.argsort(-p_test[best][hard_to_small])][:10]
    errors = [
        {
            "query_id": data.labels["query_id"].iat[te[i]],
            "dataset": data.labels["dataset"].iat[te[i]],
            "subject": data.labels["subject"].iat[te[i]],
            "p_small": float(p_test[best][i]),
            "text": data.texts[te[i]][:220].replace("\n", " "),
        }
        for i in worst
    ]

    # 9. Figures and the per-query table the dashboard reads.
    all_p = best_router.predict_proba(F)
    labels = data.labels.assign(p_small=all_p, routed_large=all_p < tau)
    mmlu = labels[labels["dataset"] == "mmlu"]
    plots.cost_vs_accuracy(
        sweeps_test,
        random_line(ot),
        base,
        headline,
        best,
        fig_dir / "fig1_cost_vs_accuracy.png",
        hybrid_front if cascade.get("covers_full_test") else None,
        subtitle=f"{len(te)} held-out queries · operating point chosen on out-of-fold predictions",
    )
    plots.pgr_curves(
        {n: pgr_curve(sweeps_test[n], ot) for n in names},
        {n: per_router[n]["test_apgr"] for n in names},
        fig_dir / "fig2_pgr.png",
    )
    plots.threshold_tradeoff(sweeps_test[best], tau, best, fig_dir / "fig3_threshold.png")
    plots.per_dataset(ds_sweeps, ds_base, best, fig_dir / "fig4a_per_dataset.png")
    if len(mmlu):
        plots.subject_routing(
            mmlu.groupby("subject")["routed_large"].mean(), fig_dir / "fig4b_subjects.png"
        )
    plots.reliability({n: (oof[n], y[dev]) for n in LEARNED}, fig_dir / "fig5_calibration.png")
    lat = {
        "Always large": ot.large_latency,
        f"Router ({best})": np.where(to_small, ot.small_latency, ot.large_latency) + latency["p50"],
    }
    plots.latency_boxes(lat, fig_dir / "fig6_latency.png")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sw in sweeps_test.items():
        (out_dir / "sweeps").mkdir(exist_ok=True)
        sw.to_parquet(out_dir / "sweeps" / f"{name}.parquet", index=False)
    cols = [
        "query_id",
        "dataset",
        "subject",
        "split",
        "small_correct",
        "large_correct",
        "small_cost_usd",
        "large_cost_usd",
        "small_latency_ms",
        "large_latency_ms",
    ]
    table = labels[cols].assign(text=data.texts, check_passed=checks.passed, has_samples=has)
    for n in routers:
        table[f"p_{n}"] = routers[n].predict_proba(F)
    table.to_parquet(out_dir / "predictions.parquet", index=False)

    return {
        "n": {s: int(len(i)) for s, i in idx.items()},
        "best_router": best,
        "tau": tau,
        "presets": presets,
        "router_latency_ms": latency,
        "headline": headline,
        "headline_ci95": ci,
        "mcnemar_vs_large": significance,
        "baselines_test": base,
        "routers": per_router,
        "cascade": cascade,
        "price_sensitivity": sensitivity,
        "per_dataset": per_dataset,
        "accuracy_budgets_test": budgets,
        "generalisation": generalisation,
        "worst_misroutes": errors,
        "max_drop_pts": max_drop_pts,
    }


def _cross_dataset(data: ExperimentData, best: str, seed: int) -> dict[str, Any]:
    """Train the best router type on one dataset's train split, test on the other's test."""
    ds = data.labels["dataset"].to_numpy()
    out: dict[str, Any] = {}
    for src, dst in (("gsm8k", "mmlu"), ("mmlu", "gsm8k")):
        tr = data.idx["train"][ds[data.idx["train"]] == src]
        te = data.idx["test"][ds[data.idx["test"]] == dst]
        y = data.outcomes.small_ok[tr]
        if len(tr) < MIN_PER_SPLIT or len(te) < MIN_PER_SPLIT or y.all() or not y.any():
            continue
        router = next(r for r in all_routers(seed) if r.name == best)
        router.fit(data.features.take(tr), y)
        o_te = data.outcomes.take(te)
        out[f"{src}_to_{dst}"] = {
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "apgr": apgr(threshold_sweep(router.predict_proba(data.features.take(te)), o_te), o_te),
        }
    return out
