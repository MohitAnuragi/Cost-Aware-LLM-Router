"""Threshold explorer: move the slider, watch accuracy, cost and routing mix change.

Run: python -m streamlit run dashboard/app.py
Reads reports/<profile>/predictions.parquet and results.json written by `router evaluate`.
Everything is replayed from cached answers; the 'try a question' box makes no LLM call.
"""

from __future__ import annotations

import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from costrouter.config import load_models_config, project_root, reports_dir
from costrouter.eval.metrics import Outcomes, baselines, router_decisions
from costrouter.eval.plots import LABELS, LEARNED
from costrouter.eval.sweep import threshold_sweep

st.set_page_config(page_title="Cost-aware LLM router", layout="wide")

cfg = load_models_config()
profiles = [p for p in cfg.profiles if (reports_dir() / p / "predictions.parquet").exists()]
if not profiles:
    st.error("No evaluated profile found. Run `python -m costrouter evaluate` first.")
    st.stop()
default = os.environ.get("COSTROUTER_PROFILE", cfg.active_profile)

with st.sidebar:
    st.header("Setup")
    profile = st.selectbox(
        "Profile", profiles, index=profiles.index(default) if default in profiles else 0
    )
    results = json.loads((reports_dir() / profile / "results.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(reports_dir() / profile / "predictions.parquet")
    routers = [c.removeprefix("p_") for c in table.columns if c.startswith("p_")]
    best = results["best_router"]
    router = st.selectbox(
        "Router", routers, index=routers.index(best), format_func=lambda n: LABELS.get(n, n)
    )
    split = st.selectbox("Split", ["test", "val", "train"], index=0)
    st.caption(
        f"Small: `{results['meta']['small']}` · Large: `{results['meta']['large']}` · "
        f"prices as of {results['meta']['prices_as_of']}"
    )

rows = table[table["split"] == split].reset_index(drop=True)
o = Outcomes.from_labels(rows)
p = rows[f"p_{router}"].to_numpy()
op_share = results["routers"][router]["test_at_tau"]["pct_large"]

st.title("Cost-aware LLM router")
st.write(
    "Easy questions go to the cheap model, hard ones to the expensive one. "
    "Drag the slider to send more or fewer questions to the large model."
)
share = st.slider(
    "Share of questions sent to the large model",
    0.0,
    1.0,
    float(round(op_share, 2)),
    0.01,
    help="The router sends the questions with the lowest P(small model correct) to large.",
)
tau = float(np.quantile(p, share)) if share < 1 else float("inf")
to_small = p >= tau
m = router_decisions(o, to_small)
b = baselines(o)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{100 * m['accuracy']:.1f}%", f"{m['delta_vs_large_pts']:+.1f} pts vs large")
c2.metric("Cost saved", f"{100 * m['cost_saved']:.1f}%")
c3.metric("Sent to large", f"{100 * m['pct_large']:.1f}%")
c4.metric("Hard queries sent to small", f"{100 * m['hard_misroute']:.1f}%")
c5.metric("p95 latency", f"{m['latency_p95_ms'] / 1000:.1f} s")

sweep = threshold_sweep(p, o)
curve = pd.DataFrame(
    {
        "cost_pct": 100 * (1 - sweep["cost_saved"]),
        "accuracy_pct": 100 * sweep["accuracy"],
        "series": LABELS.get(router, router),
    }
)
random_pts = pd.DataFrame(
    {
        "cost_pct": [100 * (1 - b["always_small"]["cost_saved"]), 100.0],
        "accuracy_pct": [100 * b["always_small"]["accuracy"], 100 * b["always_large"]["accuracy"]],
        "series": "Random",
    }
)
here = pd.DataFrame(
    {"cost_pct": [100 * (1 - m["cost_saved"])], "accuracy_pct": [100 * m["accuracy"]]}
)
color = alt.Color(
    "series:N",
    scale=alt.Scale(domain=[LABELS.get(router, router), "Random"], range=["#2a78d6", "#8f8d87"]),
    legend=alt.Legend(title=None, orient="bottom"),
)
chart = alt.Chart(pd.concat([curve, random_pts])).mark_line(strokeWidth=2).encode(
    x=alt.X("cost_pct:Q", title="Cost (% of always-large)"),
    y=alt.Y("accuracy_pct:Q", title="Accuracy (%)", scale=alt.Scale(zero=False)),
    color=color,
    strokeDash=alt.condition(alt.datum.series == "Random", alt.value([5, 4]), alt.value([1, 0])),
    tooltip=[alt.Tooltip("cost_pct:Q", format=".1f"), alt.Tooltip("accuracy_pct:Q", format=".1f")],
) + alt.Chart(here).mark_point(size=160, filled=True, color="#0b0b0b").encode(
    x="cost_pct:Q", y="accuracy_pct:Q"
)
st.altair_chart(chart, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Routing mix by dataset")
    mix = rows.assign(to_large=~to_small).groupby("dataset")["to_large"].mean().mul(100).round(1)
    st.dataframe(mix.rename("% sent to large").to_frame(), use_container_width=True)
with right:
    st.subheader("Baselines on this split")
    st.dataframe(
        pd.DataFrame(
            {
                k.replace("_", " "): {
                    "accuracy %": round(100 * v["accuracy"], 1),
                    "cost saved %": round(100 * v["cost_saved"], 1),
                }
                for k, v in b.items()
            }
        ).T,
        use_container_width=True,
    )

st.subheader("Example decisions")
view = rows.assign(p_small=p, routed_to=np.where(to_small, "small", "large"))
view["outcome"] = np.select(
    [view["small_correct"] & view["large_correct"], ~view["small_correct"] & view["large_correct"],
     view["small_correct"] & ~view["large_correct"]],
    ["easy (both right)", "hard (only large right)", "only small right"],
    "both wrong",
)  # fmt: skip
st.dataframe(
    view[["query_id", "dataset", "p_small", "routed_to", "outcome", "text"]]
    .sort_values("p_small")
    .head(200),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Try a question (routing decision only, no LLM call)")
question = st.text_area("Question", "A shop sells pens at $3 each. How much do 4 pens cost?")
if st.button("Route it") and question.strip():
    from costrouter.features import build_features
    from costrouter.routers.base import Router

    model = Router.load(project_root() / "models" / profile / f"{router}.joblib")
    prob = float(model.predict_proba(build_features([question]))[0])
    target = results["meta"]["small"] if prob >= tau else results["meta"]["large"]
    st.write(f"P(small model correct) = **{prob:.2f}**, threshold τ = {tau:.2f} → **{target}**")

if router not in LEARNED:
    st.caption("This is a baseline router, shown for comparison.")
