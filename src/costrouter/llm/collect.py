"""Run both models over a query set, resumably: cached calls are skipped, and an
interrupted run picks up where it stopped."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from costrouter.config import ROLES, Role
from costrouter.data.prompts import TEMPLATE_VERSION
from costrouter.data.schema import Query
from costrouter.llm.base import LLMClient, request_temperature
from costrouter.llm.cache import ResponseCache

CallType = Literal["answer", "sc"]  # "sc" = extra self-consistency sample


@dataclass(frozen=True)
class Job:
    query: Query
    role: Role
    call_type: CallType
    sample_idx: int


@dataclass
class CollectStats:
    planned: int = 0
    cached: int = 0
    done: int = 0
    failed: int = 0
    skipped_budget: int = 0
    skipped_daily_cap: int = 0
    capped_roles: set[str] = field(default_factory=set)
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)


def plan_jobs(
    queries: list[Query],
    *,
    self_consistency: int = 0,
    sample_ids: set[str] | None = None,
) -> list[Job]:
    """Per query, in the given order: both main answers, then the extra small-model samples,
    only for `sample_ids` when given (the cascade is evaluated on validation and test, so
    training queries don't need them). Callers put half-finished queries first."""
    jobs: list[Job] = []
    for q in queries:
        jobs.extend(Job(q, role, "answer", 0) for role in ROLES)
        if sample_ids is None or q.id in sample_ids:
            jobs.extend(Job(q, "small", "sc", i) for i in range(1, self_consistency + 1))
    return jobs


def job_key(job: Job, client: LLMClient) -> str:
    cfg = client.config
    return ResponseCache.make_key(
        model=cfg.name,
        template_version=TEMPLATE_VERSION,
        call_type=job.call_type,
        sample_idx=job.sample_idx,
        prompt=job.query.prompt,
        params={
            "max_tokens": cfg.max_tokens,
            "temperature": request_temperature(cfg, job.call_type == "sc"),
            "effort": cfg.effort,
            **cfg.extra_params,
        },
    )


def _is_fatal(exc: Exception) -> bool:
    """Auth and permission errors fail every call the same way; stop instead of retrying."""
    return getattr(exc, "status_code", None) in (401, 403)


def _is_daily_cap(exc: Exception) -> bool:
    """Groq answers 429 with 'requests per day (RPD)' or 'tokens per day (TPD)' once a model's
    daily quota is spent. Retrying is pointless until it resets, so that model stops for today."""
    text = str(exc).lower()
    return getattr(exc, "status_code", None) == 429 and (
        "per day" in text or "(rpd)" in text or "(tpd)" in text
    )


async def collect(
    jobs: list[Job],
    clients: Mapping[Role, LLMClient],
    cache: ResponseCache,
    *,
    max_cost_usd: float | None = None,
    show_progress: bool = True,
) -> CollectStats:
    stats = CollectStats(planned=len(jobs))
    pending: list[tuple[Job, str]] = []
    for job in jobs:
        key = job_key(job, clients[job.role])
        if cache.has(key):
            stats.cached += 1
        else:
            pending.append((job, key))

    semaphores = {role: asyncio.Semaphore(c.config.concurrency) for role, c in clients.items()}
    progress = Progress(
        TextColumn("[bold]collecting"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("${task.fields[cost]:.4f}"),
        TimeRemainingColumn(),
        disable=not show_progress,
    )

    def over_budget() -> bool:
        return max_cost_usd is not None and stats.cost_usd >= max_cost_usd

    async def run(job: Job, key: str, task_id: int) -> None:
        async with semaphores[job.role]:
            if over_budget():
                stats.skipped_budget += 1
            elif job.role in stats.capped_roles:
                stats.skipped_daily_cap += 1
            else:
                try:
                    resp = await clients[job.role].complete(
                        job.query.prompt, sample=job.call_type == "sc"
                    )
                except Exception as exc:
                    if _is_fatal(exc):
                        raise
                    if _is_daily_cap(exc):
                        stats.capped_roles.add(job.role)
                        stats.skipped_daily_cap += 1
                    else:
                        stats.failed += 1
                        stats.errors.append(f"{job.query.id} [{job.role}/{job.call_type}]: {exc!r}")
                else:
                    cache.put(
                        key,
                        query_id=job.query.id,
                        role=job.role,
                        call_type=job.call_type,
                        sample_idx=job.sample_idx,
                        template_version=TEMPLATE_VERSION,
                        response=resp,
                    )
                    stats.done += 1
                    stats.cost_usd += resp.cost_usd
        progress.update(task_id, advance=1, cost=stats.cost_usd)

    with progress:
        task_id = progress.add_task("collect", total=len(pending), cost=0.0)
        await asyncio.gather(*(run(job, key, task_id) for job, key in pending))
    return stats
