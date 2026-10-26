from pathlib import Path

import pytest

from costrouter.config import ROLES, ModelConfig, load_models_config
from costrouter.data.responses import label_table, response_table
from costrouter.eval.cost import cost_usd
from costrouter.eval.pilot import build_pilot_report
from costrouter.llm.base import MissingAPIKeyError, request_temperature
from costrouter.llm.cache import ResponseCache
from costrouter.llm.collect import collect, job_key, plan_jobs
from costrouter.llm.factory import build_client


def test_cost_usd():
    cfg = ModelConfig(name="m", provider="fake", usd_in_per_m=2.0, usd_out_per_m=10.0)
    assert cost_usd(cfg, 1_000_000, 0) == pytest.approx(2.0)
    assert cost_usd(cfg, 300, 600) == pytest.approx(0.0066)


def test_request_temperature_respects_providers_without_sampling():
    no_sampling = ModelConfig(
        name="c", provider="openai_compat", usd_in_per_m=1, usd_out_per_m=1, temperature=None
    )
    groq = ModelConfig(name="g", provider="openai_compat", usd_in_per_m=1, usd_out_per_m=1)
    assert request_temperature(no_sampling, sample=True) is None
    assert request_temperature(groq, sample=False) == 0.0
    assert request_temperature(groq, sample=True) == 0.7


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    cfg = ModelConfig(
        name="g",
        provider="openai_compat",
        api_key_env="NO_SUCH_KEY",
        usd_in_per_m=1,
        usd_out_per_m=1,
    )
    with pytest.raises(MissingAPIKeyError, match="NO_SUCH_KEY"):
        build_client(cfg)


def test_extra_params_change_the_cache_key(queries):
    low = ModelConfig(
        name="m", provider="fake", usd_in_per_m=1, usd_out_per_m=1,
        extra_params={"reasoning_effort": "low"},
    )  # fmt: skip
    high = low.model_copy(update={"extra_params": {"reasoning_effort": "high"}})
    job = plan_jobs(queries[:1])[0]
    keys = {job_key(job, build_client(cfg, queries)) for cfg in (low, high)}
    assert len(keys) == 2


def test_shipped_profiles_load():
    cfg = load_models_config(Path(__file__).parents[1] / "configs" / "models.yaml")
    for name in cfg.profiles:
        cfg.profile(name)
    assert cfg.profile().large.tpd == 200_000


def test_plan_jobs_counts(queries):
    jobs = plan_jobs(queries, self_consistency=2)
    assert len(jobs) == len(queries) * 4
    assert sum(j.call_type == "sc" for j in jobs) == len(queries) * 2


async def test_collect_is_resumable_and_labels_are_consistent(tmp_path, queries, fake_profile):
    clients = {role: build_client(fake_profile.role(role), queries) for role in ROLES}
    cache = ResponseCache(tmp_path / "cache.sqlite")
    jobs = plan_jobs(queries, self_consistency=1)

    first = await collect(jobs, clients, cache, show_progress=False)
    assert first.done == len(jobs) and first.failed == 0
    second = await collect(jobs, clients, cache, show_progress=False)
    assert second.cached == len(jobs) and second.done == 0

    responses = response_table(cache, queries, fake_profile)
    labels = label_table(responses)
    assert len(responses) == len(jobs)
    assert len(labels) == len(queries)
    # The fake large model (skill 0.9) solves a superset of the small one (skill 0.5).
    assert not (labels["small_correct"] & ~labels["large_correct"]).any()
    assert labels["large_correct"].mean() > labels["small_correct"].mean()


async def test_budget_cap_stops_new_calls(tmp_path, queries, fake_profile):
    clients = {role: build_client(fake_profile.role(role), queries) for role in ROLES}
    cache = ResponseCache(tmp_path / "cache.sqlite")
    stats = await collect(plan_jobs(queries), clients, cache, max_cost_usd=0.0, show_progress=False)
    assert stats.done == 0 and stats.skipped_budget == len(queries) * 2


async def test_pilot_report(tmp_path, queries, fake_profile):
    clients = {role: build_client(fake_profile.role(role), queries) for role in ROLES}
    cache = ResponseCache(tmp_path / "cache.sqlite")
    await collect(plan_jobs(queries), clients, cache, show_progress=False)
    responses = response_table(cache, queries, fake_profile)
    report = build_pilot_report(
        responses, label_table(responses), fake_profile, {"gsm8k": 1000, "mmlu": 2000}
    )
    proj = report["projection"].set_index("role")
    assert proj.loc["small", "requests"] == 3000 * 3  # answer + 2 self-consistency samples
    assert proj.loc["large", "requests"] == 3000
    assert sum(report["outcomes"].values()) == len(queries)
    assert report["gaps_pts"]["overall"] > 0
