"""Command-line entry point: `router data prepare | pilot | collect | export`."""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from costrouter.config import (
    ROLES,
    Profile,
    data_dir,
    load_datasets_config,
    load_models_config,
    project_root,
    reports_dir,
)
from costrouter.data.schema import Query, load_queries, save_queries
from costrouter.data.splits import load_splits

app = typer.Typer(help="Cost-aware LLM router.", no_args_is_help=True)
data_app = typer.Typer(help="Prepare the query set.", no_args_is_help=True)
app.add_typer(data_app, name="data")
console = Console()

ProfileOpt = Annotated[
    str | None, typer.Option("--profile", "-p", help="Model pair from configs/models.yaml.")
]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Use the fake 'dryrun' models; no API key needed.")
]
MaxCostOpt = Annotated[
    float | None, typer.Option("--max-cost", help="Stop new calls once list-price spend hits this.")
]


def _queries_path() -> Path:
    return data_dir() / "queries.parquet"


def _splits_path() -> Path:
    return data_dir() / "splits.json"


def _resolve_profile(profile: str | None, dry_run: bool) -> tuple[str, Profile]:
    name = "dryrun" if dry_run else (profile or load_models_config().active_profile)
    return name, load_models_config().profile(name)


def _cache_path(profile_name: str) -> Path:
    # Fake responses live in their own file so they can never mix with real ones.
    return data_dir() / ("cache-dryrun.sqlite" if profile_name == "dryrun" else "cache.sqlite")


def _run_collection(
    queries: list[Query],
    profile_name: str,
    profile: Profile,
    *,
    self_consistency: int,
    max_cost: float | None,
    sample_ids: set[str] | None = None,
) -> None:
    from costrouter.llm.base import MissingAPIKeyError
    from costrouter.llm.cache import ResponseCache
    from costrouter.llm.collect import collect, plan_jobs
    from costrouter.llm.factory import build_client

    try:
        clients = {role: build_client(profile.role(role), queries) for role in ROLES}
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    cache = ResponseCache(_cache_path(profile_name))
    try:
        jobs = plan_jobs(queries, self_consistency=self_consistency, sample_ids=sample_ids)
        stats = asyncio.run(collect(jobs, clients, cache, max_cost_usd=max_cost))
    finally:
        cache.close()
    console.print(
        f"planned {stats.planned} | cached {stats.cached} | new {stats.done} | "
        f"failed {stats.failed} | skipped (budget) {stats.skipped_budget} | "
        f"list cost of new calls ${stats.cost_usd:.4f}"
    )
    if stats.capped_roles:
        console.print(
            f"[yellow]daily free-tier cap reached for: {', '.join(sorted(stats.capped_roles))}; "
            f"{stats.skipped_daily_cap} calls left for the next run (rerun after the reset)."
        )
    for err in stats.errors[:5]:
        console.print(f"[red]error[/red] {err}")
    if len(stats.errors) > 5:
        console.print(f"... and {len(stats.errors) - 5} more errors (rerun to retry them)")


def _answered_by_any_model(profile_name: str, profile: Profile) -> set[str]:
    from costrouter.data.prompts import TEMPLATE_VERSION
    from costrouter.llm.cache import ResponseCache

    cache = ResponseCache(_cache_path(profile_name))
    try:
        records = cache.records([profile.small.name, profile.large.name], TEMPLATE_VERSION)
    finally:
        cache.close()
    return {r["query_id"] for r in records if r["call_type"] == "answer"}


def _load_tables(queries: list[Query], profile_name: str, profile: Profile) -> tuple[Any, Any]:
    from costrouter.data.responses import label_table, response_table
    from costrouter.llm.cache import ResponseCache

    cache = ResponseCache(_cache_path(profile_name))
    try:
        responses = response_table(cache, queries, profile)
    finally:
        cache.close()
    return responses, label_table(responses)


@app.callback()
def _main() -> None:
    load_dotenv(project_root() / ".env")
    # Windows without Developer Mode can't symlink; the HF cache still works, just noisily.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@data_app.command("prepare")
def data_prepare() -> None:
    """Download GSM8K + MMLU, sample the query set, and write the splits (incl. pilot)."""
    from costrouter.data.loaders import build_queries
    from costrouter.data.splits import make_splits, pick_pilot, save_splits

    cfg = load_datasets_config()
    queries = build_queries(cfg)
    splits: dict[str, list[str]] = make_splits(queries, cfg.split, cfg.seed)
    splits["pilot"] = pick_pilot(queries, splits["train"], cfg.pilot_per_dataset, cfg.seed)
    save_queries(queries, _queries_path())
    save_splits(splits, _splits_path())

    counts = Counter(q.dataset for q in queries)
    console.print(f"wrote {len(queries)} queries to {_queries_path()} ({dict(counts)})")
    console.print({name: len(ids) for name, ids in splits.items()})


@app.command()
def pilot(
    profile: ProfileOpt = None, dry_run: DryRunOpt = False, max_cost: MaxCostOpt = None
) -> None:
    """Run both models on the pilot subset and check the accuracy-gap gate."""
    from costrouter.eval.pilot import build_pilot_report

    name, prof = _resolve_profile(profile, dry_run)
    queries = load_queries(_queries_path())
    pilot_ids = set(load_splits(_splits_path())["pilot"])
    pilot_queries = [q for q in queries if q.id in pilot_ids]
    console.print(
        f"pilot: {len(pilot_queries)} queries, profile [bold]{name}[/bold] "
        f"({prof.small.name} vs {prof.large.name})"
    )

    _run_collection(pilot_queries, name, prof, self_consistency=0, max_cost=max_cost)
    responses, labels = _load_tables(pilot_queries, name, prof)
    if labels.empty:
        console.print("[red]No query has answers from both models yet; fix the errors and rerun.")
        raise typer.Exit(1)

    full_counts = Counter(q.dataset for q in queries)
    report = build_pilot_report(responses, labels, prof, full_counts)
    _print_pilot_report(report)
    out = reports_dir() / f"pilot_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # default=: NumPy scalars (np.int64, np.bool_) expose .item() to get the Python value.
    out.write_text(
        json.dumps(_jsonable(report), indent=2, default=lambda o: o.item()), encoding="utf-8"
    )
    console.print(f"saved {out}")


@app.command()
def pairs() -> None:
    """Score every small/large pair of the configured free models on the pilot (cache only)."""
    from costrouter.data.responses import graded_records
    from costrouter.eval.pairs import pair_matrix
    from costrouter.llm.cache import ResponseCache

    models_cfg = load_models_config()
    prices = {
        p.role(r).name: p.role(r)
        for n, p in models_cfg.profiles.items()
        if n != "dryrun"
        for r in ROLES
    }
    queries = load_queries(_queries_path())
    pilot_ids = set(load_splits(_splits_path())["pilot"])
    cache = ResponseCache(_cache_path("free"))
    try:
        pilot_queries = [q for q in queries if q.id in pilot_ids]
        answers = graded_records(cache, pilot_queries, sorted(prices), prices)
    finally:
        cache.close()
    matrix = pair_matrix(answers)

    t = Table(title="Candidate pairs on the pilot (sorted by oracle cost saved)")
    for col in (
        "small",
        "large",
        "n",
        "small acc",
        "large acc",
        "gap pts",
        "hard",
        "cost ratio",
        "oracle acc",
        "oracle saved",
    ):
        t.add_column(col, justify="right")
    for r in matrix.itertuples():
        t.add_row(
            r.small,
            r.large,
            str(r.n),
            f"{r.small_acc:.1%}",
            f"{r.large_acc:.1%}",
            f"{r.gap_pts:+.1f}",
            f"{r.hard_share:.1%}",
            f"{r.cost_ratio:.1f}x",
            f"{r.oracle_acc:.1%}",
            f"{r.oracle_cost_saved:.1%}",
        )
    console.print(t)
    out = reports_dir() / "pairs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_json(out, orient="records", indent=2)
    console.print(f"saved {out}")


@app.command("collect")
def collect_cmd(
    profile: ProfileOpt = None,
    dry_run: DryRunOpt = False,
    split: Annotated[str | None, typer.Option(help="Only this split (train/val/test).")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N queries.")] = None,
    self_consistency: Annotated[
        int, typer.Option(help="Extra small-model samples per query for the self-check.")
    ] = 0,
    max_cost: MaxCostOpt = None,
    samples_for_train: Annotated[
        bool, typer.Option(help="Also sample train queries (the cascade only evaluates val/test).")
    ] = False,
) -> None:
    """Run both models over the query set. Safe to interrupt and rerun: cached calls are skipped."""
    name, prof = _resolve_profile(profile, dry_run)
    queries = load_queries(_queries_path())
    splits = load_splits(_splits_path())
    if split:
        ids = set(splits[split])
        queries = [q for q in queries if q.id in ids]
    sample_ids = None if samples_for_train else set(splits["val"]) | set(splits["test"])
    # Seeded shuffle: a run cut short by a daily cap still leaves a balanced sample of every
    # split, dataset and subject, and the next run continues in the same order.
    random.Random(load_datasets_config().seed).shuffle(queries)
    # Then finish half-done queries first (one model answered, e.g. before a daily cap): they
    # become usable labels with a single call. The sort is stable, so the order is kept.
    started = _answered_by_any_model(name, prof)
    queries.sort(key=lambda q: q.id not in started)
    if limit:
        queries = queries[:limit]
    console.print(f"collect: {len(queries)} queries, profile [bold]{name}[/bold]")
    _run_collection(
        queries,
        name,
        prof,
        self_consistency=self_consistency,
        max_cost=max_cost,
        sample_ids=sample_ids,
    )


@app.command()
def export(profile: ProfileOpt = None, dry_run: DryRunOpt = False) -> None:
    """Grade every cached response; write responses_<profile> and labels_<profile> Parquet files."""
    name, prof = _resolve_profile(profile, dry_run)
    queries = load_queries(_queries_path())
    responses, labels = _load_tables(queries, name, prof)
    responses.to_parquet(data_dir() / f"responses_{name}.parquet", index=False)
    labels.to_parquet(data_dir() / f"labels_{name}.parquet", index=False)
    console.print(f"{len(responses)} responses, {len(labels)} labelled queries -> {data_dir()}")


@app.command()
def evaluate(
    profile: ProfileOpt = None,
    dry_run: DryRunOpt = False,
    max_drop: Annotated[
        float, typer.Option(help="Allowed accuracy drop vs always-large, in points.")
    ] = 2.0,
) -> None:
    """Train all routers, choose thresholds by cross-validation, evaluate once on test, and write
    reports/<profile>/RESULTS.md, results.json and figures. Uses cached answers only."""
    from datetime import date

    from costrouter.eval.experiment import NotEnoughDataError, load_data, run_experiment
    from costrouter.eval.report import render
    from costrouter.features.embeddings import DEFAULT_MODEL

    name, prof = _resolve_profile(profile, dry_run)
    models_cfg = load_models_config()
    queries = load_queries(_queries_path())
    splits = load_splits(_splits_path())
    responses, _ = _load_tables(queries, name, prof)
    try:
        data = load_data(queries, splits, responses, data_dir() / "embeddings")
    except NotEnoughDataError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    # Conservative re-pricing of the small model: the cheapest officially priced Groq model.
    conservative = None
    if name != "dryrun" and "free_oss" in models_cfg.profiles:
        conservative = models_cfg.profiles["free_oss"].small
        if conservative.name == prof.small.name:
            conservative = None
    out_dir = reports_dir() / name
    models_dir = project_root() / "models" / name
    console.print(f"evaluate: profile [bold]{name}[/bold], {len(data.labels)} labelled queries")
    results = run_experiment(
        data, out_dir, models_dir, max_drop_pts=max_drop, conservative_small=conservative
    )
    meta = {
        "generated": date.today().isoformat(),
        "profile": name,
        "small": prof.small.name,
        "large": prof.large.name,
        "prices_as_of": models_cfg.prices_as_of,
        "n_queries": len(queries),
    }
    dump = lambda obj: json.dumps(obj, indent=2, default=_json_default)  # noqa: E731
    (out_dir / "results.json").write_text(dump({"meta": meta, **results}), encoding="utf-8")
    (out_dir / "RESULTS.md").write_text(render(results, meta), encoding="utf-8")
    (models_dir / "metadata.json").write_text(
        dump(
            {
                **meta,
                "best_router": results["best_router"],
                "tau": results["tau"],
                "presets": results["presets"],
                "embedding_model": DEFAULT_MODEL,
                "router_input": "user text only (prompt template stripped)",
                "trained_on": f"{results['n']['train']} labelled train queries",
            }
        ),
        encoding="utf-8",
    )
    h = results["headline"]
    console.print(
        f"best router [bold]{results['best_router']}[/bold] at tau {results['tau']:.3f}: "
        f"{100 * h['cost_saved']:.1f}% cost saved, {h['delta_vs_large_pts']:+.1f} pts vs "
        f"always-large, {100 * h['pct_large']:.1f}% to large, hard mis-route "
        f"{100 * h['hard_misroute']:.1f}%"
    )
    console.print(f"wrote {out_dir / 'RESULTS.md'}")


@app.command()
def reproduce(profile: ProfileOpt = None, dry_run: DryRunOpt = False) -> None:
    """Rebuild every table and figure from the cached answers (no API calls)."""
    export(profile=profile, dry_run=dry_run)
    evaluate(profile=profile, dry_run=dry_run)


@app.command()
def serve(
    profile: ProfileOpt = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the OpenAI-compatible routing gateway (needs `evaluate` to have run first)."""
    import uvicorn

    if profile:
        os.environ["COSTROUTER_PROFILE"] = profile
    uvicorn.run("costrouter.serve.app:create_default_app", factory=True, host=host, port=port)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _print_pilot_report(report: dict[str, Any]) -> None:
    t = Table(title="Pilot accuracy and usage (main answers)")
    for col in (
        "dataset",
        "role",
        "n",
        "accuracy",
        "parse fail",
        "in tok",
        "out tok",
        "latency ms",
        "$/query",
    ):
        t.add_column(col, justify="right")
    for r in report["summary"].itertuples():
        t.add_row(
            r.dataset,
            r.role,
            str(r.n),
            f"{r.accuracy:.1%}",
            f"{r.parse_fail_rate:.1%}",
            f"{r.mean_in_tokens:.0f}",
            f"{r.mean_out_tokens:.0f}",
            f"{r.mean_latency_ms:.0f}",
            f"{r.mean_cost_usd:.6f}",
        )
    console.print(t)

    console.print(
        "accuracy gap, large minus small (points): "
        + ", ".join(f"{k} {v:+.1f}" for k, v in report["gaps_pts"].items())
    )
    console.print("outcomes: " + ", ".join(f"{k} {v}" for k, v in report["outcomes"].items()))

    p = Table(title="Projected full run (self-consistency = 2)")
    for col in (
        "role",
        "model",
        "requests",
        "tokens",
        "hours @ limit",
        "days @ daily cap",
        "list cost $",
    ):
        p.add_column(col, justify="right")
    for r in report["projection"].itertuples():
        days = "-" if pd.isna(r.days_at_daily_limit) else f"{r.days_at_daily_limit:.1f}"
        p.add_row(
            r.role,
            r.model,
            f"{r.requests:,}",
            f"{r.tokens:,}",
            f"{r.hours_at_rate_limit:.1f}",
            days,
            f"{r.list_cost_usd:.2f}",
        )
    console.print(p)

    if report["gate_passed"]:
        console.print(
            f"[green bold]GATE PASSED[/green bold]: overall gap >= {report['min_gap_pts']:.0f} "
            "points, so there is something to route on."
        )
    else:
        console.print(
            f"[yellow bold]GATE NOT PASSED[/yellow bold]: overall gap < "
            f"{report['min_gap_pts']:.0f} points. Try a weaker small model, a stronger large "
            "model, or a harder dataset before the full run."
        )


def _jsonable(report: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v.to_dict(orient="records") if hasattr(v, "to_dict") else v) for k, v in report.items()
    }


if __name__ == "__main__":
    app()
