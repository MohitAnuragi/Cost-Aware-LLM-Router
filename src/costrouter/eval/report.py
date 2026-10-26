"""Render the experiment's results as reports/RESULTS.md."""

from __future__ import annotations

from typing import Any

from costrouter.eval.plots import LABELS


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(ci: tuple[float, float], scale: float = 100, unit: str = "%") -> str:
    return f"[{scale * ci[0]:.1f}{unit}, {scale * ci[1]:.1f}{unit}]"


def render(results: dict[str, Any], meta: dict[str, Any]) -> str:
    r, best = results, results["best_router"]
    h, ci, base = r["headline"], r["headline_ci95"], r["baselines_test"]
    n = r["n"]
    lines = [
        "# Results",
        "",
        f"Generated {meta['generated']} from cached answers · profile `{meta['profile']}` · "
        f"small `{meta['small']}` vs large `{meta['large']}` · "
        f"prices as of {meta['prices_as_of']}.",
        "",
        f"Labelled queries: {n['train']} train / {n['val']} validation / {n['test']} test "
        f"(of {meta['n_queries']} in the full query set). Routers were compared and thresholds "
        "chosen on 5-fold out-of-fold predictions over train + validation; the final routers "
        "were refit on train + validation; the test split was scored once.",
        "",
        "## Headline",
        "",
        f"**{LABELS.get(best, best)} saved {_pct(h['cost_saved'])} of LLM cost "
        f"(95% CI {_ci(ci['cost_saved'])}) with a {h['delta_vs_large_pts']:+.1f}-point accuracy "
        f"change vs always using the large model (95% CI "
        f"{_ci(ci['delta_vs_large_pts'], 1, ' pts')}), adding "
        f"{r['router_latency_ms']['p50']:.0f} ms (p50) of routing latency.**",
        "",
        f"- Accuracy {_pct(h['accuracy'])} vs "
        f"{_pct(base['always_large']['accuracy'])} always-large "
        f"and {_pct(base['always_small']['accuracy'])} always-small.",
        f"- {_pct(h['pct_large'])} of queries went to the large model.",
        f"- Hard-query mis-route rate {_pct(h['hard_misroute'])} (95% CI "
        f"{_ci(ci['hard_misroute'])}): share of queries only the large model solves that were "
        "sent to the small one.",
        f"- McNemar test vs always-large: p = {r['mcnemar_vs_large']['p_value']:.3f} "
        f"({r['mcnemar_vs_large']['a_only']} queries only the router got right, "
        f"{r['mcnemar_vs_large']['b_only']} only always-large got right).",
        f"- Routing overhead (embedding + classifier, one CPU thread): p50 "
        f"{r['router_latency_ms']['p50']:.1f} ms, p95 {r['router_latency_ms']['p95']:.1f} ms"
        + (
            f", measured while the machine's CPU was {r['router_latency_ms']['cpu_load_pct']:.0f}% "
            "busy with other work."
            if r["router_latency_ms"].get("cpu_load_pct") is not None
            else "."
        ),
        f"- Oracle router (upper bound): {_pct(base['oracle']['cost_saved'])} saved at "
        f"{_pct(base['oracle']['accuracy'])} accuracy.",
        "",
        "![Cost vs accuracy](figures/fig1_cost_vs_accuracy.png)",
        "",
        "## All routers (test split)",
        "",
        "| Router | Accuracy at τ | Δ vs large | Cost saved | % to large | Hard mis-route | "
        "APGR (mean ± sd, 3 seeds) | CPT(50%) | CPT(80%) | CV ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, d in r["routers"].items():
        t = d["test_at_tau"]
        apgr = (
            f"{d['test_apgr_mean']:.3f} ± {d['test_apgr_std']:.3f}"
            if "test_apgr_mean" in d
            else f"{d['test_apgr']:.3f}"
        )
        lines.append(
            f"| {LABELS.get(name, name)}{' **(best)**' if name == best else ''} | "
            f"{_pct(t['accuracy'])} | {t['delta_vs_large_pts']:+.1f} | {_pct(t['cost_saved'])} | "
            f"{_pct(t['pct_large'])} | {_pct(t['hard_misroute'])} | {apgr} | "
            f"{_pct(d['test_cpt50'])} | {_pct(d['test_cpt80'])} | "
            f"{d['cv_diagnostics']['roc_auc']:.3f} |"
        )
    lines += [
        f"| Always small | {_pct(base['always_small']['accuracy'])} | "
        f"{base['always_small']['delta_vs_large_pts']:+.1f} | "
        f"{_pct(base['always_small']['cost_saved'])} | 0% | 100% | 0 | | | |",
        f"| Always large | {_pct(base['always_large']['accuracy'])} | 0.0 | 0% | 100% | 0% "
        "| 1 | | | |",
        f"| Oracle | {_pct(base['oracle']['accuracy'])} | "
        f"{base['oracle']['delta_vs_large_pts']:+.1f} | {_pct(base['oracle']['cost_saved'])} | "
        f"{_pct(base['oracle']['pct_large'])} | 0% | | | | |",
        "",
        "APGR = area under the performance-gap-recovered curve (random = 0.5). CPT(x%) = share "
        "of calls to the large model needed to recover x% of the accuracy gap. The best router "
        "is the learned router with the highest out-of-fold APGR.",
        "",
        "### The full trade-off (best router, test split)",
        "",
        "Each row's threshold was chosen on out-of-fold predictions for that accuracy budget, "
        "then scored once on test.",
        "",
        "| Allowed drop | τ | Accuracy | Δ vs large | Cost saved | % to large | Hard mis-route |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {k} pts | {v['tau']:.3f} | {_pct(v['accuracy'])} | {v['delta_vs_large_pts']:+.1f} "
            f"| {_pct(v['cost_saved'])} | {_pct(v['pct_large'])} | {_pct(v['hard_misroute'])} |"
            for k, v in r.get("accuracy_budgets_test", {}).items()
        ],
        "",
        "![PGR curves](figures/fig2_pgr.png)",
        "",
        "![Threshold trade-off](figures/fig3_threshold.png)",
        "",
    ]

    c = r["cascade"]
    lines += ["## Self-check cascade", ""]
    if c.get("available"):
        q = c["check_quality"]
        lines += [
            f"Evaluated on the {c['n_test']} test queries that have 2 extra small-model samples. "
            f"Check = the main answer parses and at least 2 of 3 answers agree.",
            "",
            "| Policy | Accuracy | Δ vs large | Cost saved | % answered by large | "
            "Hard mis-route | % checked | p95 latency |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key, label in (
            ("router_only", "Router only"),
            ("hybrid", f"Hybrid (τ_low {c['tau_low']:.2f}, τ_high {c['tau_high']:.2f})"),
            ("cascade_only", "Cascade only (always small first)"),
            ("always_large", "Always large"),
        ):
            d = c[key]
            lines.append(
                f"| {label} | {_pct(d['accuracy'])} | {d['delta_vs_large_pts']:+.1f} | "
                f"{_pct(d['cost_saved'])} | {_pct(d['pct_large'])} | {_pct(d['hard_misroute'])} | "
                f"{_pct(d.get('pct_checked', 0))} | {d['latency_p95_ms'] / 1000:.1f} s |"
            )
        lines += [
            "",
            f"Check quality: precision {_pct(q['precision'])}, recall {_pct(q['recall'])} for "
            f"'check failed' as a detector of 'small model was wrong'; it fails on "
            f"{_pct(q['fail_rate'])} of queries.",
            "",
            "![Cascade](figures/fig7_cascade.png)",
            "",
        ]
    else:
        lines += ["Not enough queries with self-consistency samples yet.", ""]

    lines += ["## Robustness", ""]
    s = r["price_sensitivity"]
    if s:
        lines += [
            f"- **Small-model price stand-in.** `{meta['small']}` has no published price. At the "
            f"more conservative {s['small_price']} per 1M tokens, the same frozen router saves "
            f"{_pct(s['cost_saved'])} (oracle {_pct(s['oracle_cost_saved'])}).",
        ]
    for ds, d in r["per_dataset"].items():
        a = d["at_tau"]
        lines.append(
            f"- **{ds.upper()}** ({d['n']} test queries): small {_pct(d['always_small_acc'])}, "
            f"large {_pct(d['always_large_acc'])}; at τ accuracy {_pct(a['accuracy'])}, "
            f"{_pct(a['cost_saved'])} saved, {_pct(a['pct_large'])} to large, APGR {d['apgr']:.3f}."
        )
    for key, g in r["generalisation"].items():
        src, dst = key.split("_to_")
        lines.append(
            f"- **Generalisation {src.upper()} → {dst.upper()}**: trained on {g['n_train']} "
            f"{src.upper()} queries only, APGR {g['apgr']:.3f} on {g['n_test']} {dst.upper()} "
            "test queries (random = 0.5)."
        )
    lines += [
        "",
        "![Per dataset](figures/fig4a_per_dataset.png)",
        "",
        "![Subjects](figures/fig4b_subjects.png)",
        "",
        "![Calibration](figures/fig5_calibration.png)",
        "",
        "![Latency](figures/fig6_latency.png)",
        "",
        "## Where the router fails",
        "",
        "Hard queries (only the large model was right) that the router most confidently sent to "
        "the small model:",
        "",
        "| Query | p(small ok) | Text |",
        "|---|---|---|",
    ]
    for e in r["worst_misroutes"]:
        text = e["text"].replace("|", "\\|")
        lines.append(f"| {e['query_id']} | {e['p_small']:.2f} | {text} |")
    lines += [
        "",
        "## API threshold presets (from out-of-fold predictions)",
        "",
        "| Preset | τ |",
        "|---|---|",
        *[f"| `router-{k}` | {v:.3f} |" for k, v in r["presets"].items()],
        "",
    ]
    return "\n".join(lines)
