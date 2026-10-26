"""Live routing: one cheap prediction per request, then a call to the model it picks."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from costrouter.config import ROLES, Profile, Role
from costrouter.data.checkers import extract_choice, extract_number
from costrouter.eval.cost import cost_usd
from costrouter.features import Features, build_features
from costrouter.llm.base import LLMClient, LLMResponse
from costrouter.routers.base import Router

_OPTIONS = re.compile(r"^\s*\(?[A-D][.)]\s", re.MULTILINE)


@dataclass(frozen=True)
class Decision:
    p_small: float
    zone: str  # "small", "check" or "large"
    tau_low: float
    tau_high: float
    router_ms: float


@dataclass
class Usage:
    """Running totals for /metrics."""

    requests: dict[str, int] = field(default_factory=lambda: {"small": 0, "large": 0})
    escalations: int = 0
    cost_usd: float = 0.0
    always_large_estimate_usd: float = 0.0


def final_answer(text: str) -> str:
    """Normalised final answer for comparing self-consistency samples."""
    if _OPTIONS.search(text) or re.search(r"answer\s*[:：]?\s*\(?[A-D]\b", text, re.I):
        choice = extract_choice(text)
        if choice:
            return choice
    number = extract_number(text)
    if number is not None:
        return f"{number:g}"
    lines = [ln.strip().lower() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class RouterService:
    def __init__(
        self,
        profile: Profile,
        router: Router,
        clients: Mapping[Role, LLMClient],
        presets: dict[str, float],
        cascade: dict[str, float] | None = None,
        featurize: Callable[[list[str]], Features] = build_features,
        samples: int = 2,
    ):
        self.profile = profile
        self.router = router
        self.clients = clients
        self.presets = presets
        self.cascade = cascade  # {"tau_low": ..., "tau_high": ...} or None
        self.featurize = featurize
        self.samples = samples
        self.usage = Usage()

    @classmethod
    def from_artifacts(
        cls,
        models_dir: Path,
        results_path: Path,
        profile: Profile,
        clients: Mapping[Role, LLMClient],
    ) -> RouterService:
        meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        router = Router.load(models_dir / f"{meta['best_router']}.joblib")
        c = results.get("cascade", {})
        cascade = (
            {"tau_low": c["tau_low"], "tau_high": c["tau_high"]} if c.get("available") else None
        )
        return cls(profile, router, clients, meta["presets"], cascade)

    def thresholds(self, model_field: str | None) -> tuple[float, float, bool]:
        """'router-<preset>', 'router-<tau>', 'router-cascade' (default when available)."""
        spec = (model_field or "").removeprefix("router-")
        if spec in self.presets:
            tau = self.presets[spec]
            return tau, tau, False
        try:
            tau = float(spec)
            return tau, tau, False
        except ValueError:
            pass
        if self.cascade and spec in ("", "auto", "cascade"):
            return self.cascade["tau_low"], self.cascade["tau_high"], True
        tau = self.presets["quality"]
        return tau, tau, False

    def decide(self, text: str, model_field: str | None = None) -> Decision:
        t0 = time.perf_counter()
        p = float(self.router.predict_proba(self.featurize([text]))[0])
        router_ms = (time.perf_counter() - t0) * 1000
        lo, hi, _ = self.thresholds(model_field)
        zone = "large" if p < lo else "small" if p >= hi else "check"
        return Decision(p, zone, lo, hi, router_ms)

    async def _self_check(
        self, messages: list[dict[str, str]], first: LLMResponse
    ) -> tuple[bool, list[LLMResponse]]:
        extra = await asyncio.gather(
            *(self.clients["small"].chat(messages, sample=True) for _ in range(self.samples))
        )
        answers = [final_answer(r.text) for r in (first, *extra)]
        agree = sum(a == answers[0] for a in answers[1:]) / max(len(extra), 1)
        return bool(answers[0]) and agree >= 0.5, list(extra)

    async def chat(
        self, messages: list[dict[str, str]], model_field: str | None = None
    ) -> dict[str, Any]:
        text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        d = self.decide(text, model_field)
        calls: list[LLMResponse] = []
        escalated = False
        if d.zone == "large":
            final = await self.clients["large"].chat(messages)
            calls.append(final)
        else:
            final = await self.clients["small"].chat(messages)
            calls.append(final)
            if d.zone == "check":
                passed, extra = await self._self_check(messages, final)
                calls += extra
                if not passed:
                    escalated = True
                    final = await self.clients["large"].chat(messages)
                    calls.append(final)
        served: Role = "large" if final.model == self.profile.large.name else "small"
        self._account(calls, served, escalated)
        return {
            "response": final,
            "served_by": served,
            "decision": d,
            "escalated": escalated,
            "cost_usd": sum(c.cost_usd for c in calls),
            "latency_ms": d.router_ms + sum(c.latency_ms for c in calls if c is not None),
        }

    def _account(self, calls: list[LLMResponse], served: Role, escalated: bool) -> None:
        u = self.usage
        u.requests[served] += 1
        u.escalations += int(escalated)
        u.cost_usd += sum(c.cost_usd for c in calls)
        first = calls[0]
        # Counterfactual: the same exchange priced at the large model's rates (estimate).
        u.always_large_estimate_usd += cost_usd(
            self.profile.large, first.input_tokens, first.output_tokens
        )

    def metrics(self) -> dict[str, Any]:
        u = self.usage
        total = sum(u.requests.values())
        saved = 1 - u.cost_usd / u.always_large_estimate_usd if u.always_large_estimate_usd else 0.0
        return {
            "requests": total,
            "by_model": {self.profile.role(r).name: u.requests[r] for r in ROLES},
            "escalations": u.escalations,
            "cost_usd": round(u.cost_usd, 8),
            "always_large_estimate_usd": round(u.always_large_estimate_usd, 8),
            "estimated_cost_saved": round(float(np.clip(saved, -10, 1)), 4),
        }
