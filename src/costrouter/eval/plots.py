"""Static README figures (matplotlib, PNG).

Colour rules: each learned router keeps one fixed categorical hue in every figure; baselines
are neutral grey; text is always ink, never a series colour; no dual-axis charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
GREY = "#8f8d87"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
LEARNED = ["logreg_handcrafted", "logreg_embedding", "knn_embedding", "gbm_both", "mlp_embedding"]
LABELS = {
    "random": "Random",
    "length": "Length heuristic",
    "logreg_handcrafted": "LogReg (handcrafted)",
    "logreg_embedding": "LogReg (embedding + handcrafted)",
    "knn_embedding": "kNN (embedding)",
    "gbm_both": "Gradient boosting",
    "mlp_embedding": "MLP (embedding)",
    "hybrid": "Hybrid cascade",
    "router_only": "Router only (same queries)",
}


def color(name: str) -> str:
    if name in LEARNED:
        return CATEGORICAL[LEARNED.index(name)]
    return CATEGORICAL[5] if name == "hybrid" else GREY


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 10,
            "lines.linewidth": 2,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def cost_vs_accuracy(
    sweeps: dict[str, pd.DataFrame],
    random_line: pd.DataFrame,
    baselines: dict[str, dict[str, float]],
    operating: dict[str, float],
    best: str,
    path: Path,
    hybrid_front: pd.DataFrame | None = None,
    subtitle: str = "",
) -> Path:
    """Figure 1, the headline: accuracy against cost as % of always-large (test split)."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(
        100 * (1 - random_line["cost_saved"]),
        100 * random_line["accuracy"],
        color=GREY,
        linestyle="--",
        linewidth=1.5,
        label=LABELS["random"],
    )
    for name, sw in sweeps.items():
        if name == "random":
            continue
        style = {"linestyle": ":", "linewidth": 1.5} if name == "length" else {}
        ax.plot(
            100 * (1 - sw["cost_saved"]),
            100 * sw["accuracy"],
            color=color(name),
            label=LABELS.get(name, name),
            alpha=1 if name == best else 0.75,
            linewidth=2.6 if name == best else style.get("linewidth", 2),
            linestyle=style.get("linestyle", "-"),
        )
    if hybrid_front is not None and len(hybrid_front):
        ax.plot(
            100 * (1 - hybrid_front["cost_saved"]),
            100 * hybrid_front["accuracy"],
            color=color("hybrid"),
            marker="o",
            markersize=4,
            linestyle="--",
            label=LABELS["hybrid"] + " (Pareto front)",
        )
    for key, label, marker in (
        ("always_small", "Always small", "s"),
        ("always_large", "Always large", "D"),
        ("oracle", "Oracle router", "*"),
    ):
        b = baselines[key]
        ax.scatter(
            100 * (1 - b["cost_saved"]),
            100 * b["accuracy"],
            marker=marker,
            s=110 if marker == "*" else 60,
            color=INK,
            zorder=5,
        )
        ax.annotate(
            label,
            (100 * (1 - b["cost_saved"]), 100 * b["accuracy"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8.5,
            color=INK_2,
        )
    ox, oy = 100 * (1 - operating["cost_saved"]), 100 * operating["accuracy"]
    ax.scatter(ox, oy, s=140, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=6)
    ax.annotate(
        f"Operating point · {LABELS.get(best, best)}\n"
        f"{100 * operating['cost_saved']:.0f}% cost saved, "
        f"{operating['delta_vs_large_pts']:+.1f} pts vs always-large",
        (ox, oy),
        textcoords="offset points",
        xytext=(12, -38),
        fontsize=8.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8},
    )
    ax.set_xlabel("Cost (% of always-large)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost vs accuracy across routing thresholds (test split)", pad=22)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.legend(loc="lower right")
    return _save(fig, path)


def pgr_curves(curves: dict[str, pd.DataFrame], apgrs: dict[str, float], path: Path) -> Path:
    """Figure 2: performance gap recovered vs share of calls to the large model."""
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([0, 100], [0, 1], color=GREY, linestyle="--", linewidth=1.5, label="Random (APGR 0.50)")
    for name, c in curves.items():
        if name == "random":
            continue
        ax.plot(
            100 * c["pct_large"],
            c["pgr"].clip(upper=1.05),
            color=color(name),
            linestyle=":" if name == "length" else "-",
            label=f"{LABELS.get(name, name)} (APGR {apgrs[name]:.2f})",
        )
    ax.set_xlabel("Calls sent to the large model (%)")
    ax.set_ylabel("Performance gap recovered (PGR)")
    ax.set_title("Share of the small-to-large accuracy gap recovered (test split)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def threshold_tradeoff(sweep: pd.DataFrame, tau_op: float, name: str, path: Path) -> Path:
    """Figure 3: cost saved and hard-query mis-routes vs threshold, as two stacked panels
    sharing the threshold axis (two measures, two scales: never one dual-axis chart)."""
    _style()
    sw = sweep[np.isfinite(sweep["tau"])].sort_values("tau")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    top.plot(sw["tau"], 100 * sw["cost_saved"], color=color(name))
    top.set_ylabel("Cost saved (%)")
    top.set_title(f"What the threshold trades off ({LABELS.get(name, name)}, test split)")
    bottom.plot(sw["tau"], 100 * sw["hard_misroute"], color=color(name))
    bottom.set_ylabel("Hard queries sent\nto small (%)")
    bottom.set_xlabel("Threshold τ (route to small when p ≥ τ)")
    for ax in (top, bottom):
        if np.isfinite(tau_op):
            ax.axvline(tau_op, color=INK_2, linestyle="--", linewidth=1)
    if np.isfinite(tau_op):
        top.annotate(
            "operating τ",
            (tau_op, top.get_ylim()[1]),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=8.5,
            color=INK_2,
        )
    return _save(fig, path)


def per_dataset(
    sweeps: dict[str, pd.DataFrame], baselines: dict[str, dict], name: str, path: Path
) -> Path:
    """Figure 4a: the best router's curve on each dataset separately (small multiples)."""
    _style()
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5 * len(sweeps), 4.2), sharey=False)
    for ax, (ds, sw) in zip(np.atleast_1d(axes), sweeps.items(), strict=True):
        ax.plot(100 * (1 - sw["cost_saved"]), 100 * sw["accuracy"], color=color(name))
        for key, marker in (("always_small", "s"), ("always_large", "D")):
            b = baselines[ds][key]
            ax.scatter(100 * (1 - b["cost_saved"]), 100 * b["accuracy"], marker=marker, color=INK)
        ax.set_title(ds.upper())
        ax.set_xlabel("Cost (% of always-large)")
        ax.set_ylabel("Accuracy (%)")
    fig.suptitle(
        f"{LABELS.get(name, name)} on each dataset (test split)",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    return _save(fig, path)


def subject_routing(share_large: pd.Series, path: Path, top_n: int = 15) -> Path:
    """Figure 4b: share of MMLU questions the router sends to large, by subject."""
    _style()
    s = share_large.sort_values()
    s = pd.concat([s.head(top_n), s.tail(top_n)]) if len(s) > 2 * top_n else s
    s = s[~s.index.duplicated()]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.24 * len(s) + 1)))
    ax.barh(
        [i.replace("_", " ") for i in s.index], 100 * s.to_numpy(), color=CATEGORICAL[0], height=0.7
    )
    ax.set_xlabel("Questions routed to the large model (%)")
    ax.set_title("Which MMLU subjects the router treats as hard (all splits)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, bins: int = 10
) -> Path:
    """Figure 5: calibration of p = P(small correct) on validation."""
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.2, label="Perfect calibration")
    edges = np.linspace(0, 1, bins + 1)
    for name, (p, y) in curves.items():
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 3]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 3]
        ax.plot(xs, ys, color=color(name), marker="o", markersize=4, label=LABELS.get(name, name))
    ax.set_xlabel("Predicted P(small model correct)")
    ax.set_ylabel("Observed share correct")
    ax.set_title("Router calibration (validation split)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def latency_boxes(latencies: dict[str, np.ndarray], path: Path) -> Path:
    """Figure 6: end-to-end latency per query under each policy (test split)."""
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names = list(latencies)
    ax.boxplot(
        [latencies[n] / 1000 for n in names],
        vert=False,
        tick_labels=names,
        widths=0.5,
        showfliers=False,
        medianprops={"color": CATEGORICAL[0], "linewidth": 2},
        boxprops={"color": INK_2},
        whiskerprops={"color": INK_2},
        capprops={"color": INK_2},
    )
    ax.set_xlabel("Latency per query (seconds)")
    ax.set_title("End-to-end latency by policy (test split, outliers hidden)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
