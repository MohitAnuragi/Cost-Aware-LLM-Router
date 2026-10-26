"""Offline stand-in for an LLM, so the whole pipeline runs and is tested with no API key.

Each query gets a fixed difficulty in [0, 1) from a hash of its prompt; a fake model answers
correctly when the difficulty is below its `skill`. A stronger model therefore solves a
superset of what a weaker one solves, like real model pairs mostly do."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable

from costrouter.config import ModelConfig
from costrouter.data.prompts import LETTERS
from costrouter.data.schema import Query
from costrouter.eval.cost import cost_usd
from costrouter.llm.base import LLMResponse, estimate_tokens


def difficulty(prompt: str) -> float:
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) / 0x1_0000_0000


class FakeClient:
    def __init__(self, config: ModelConfig, queries: Iterable[Query]):
        self.config = config
        self._key = {q.prompt: q for q in queries}
        self._calls: Counter[str] = Counter()

    async def complete(self, prompt: str, *, sample: bool = False) -> LLMResponse:
        cfg = self.config
        query = self._key.get(prompt)
        self._calls[prompt] += 1
        rng = random.Random(f"{cfg.name}|{prompt}|{self._calls[prompt]}")
        jitter = rng.uniform(-0.1, 0.1) if sample else 0.0
        correct = difficulty(prompt) + jitter < cfg.skill

        if query is None:
            answer = "42"
        elif query.dataset == "gsm8k":
            answer = query.gold if correct else str(int(float(query.gold)) + rng.randint(1, 9))
        else:
            answer = query.gold if correct else LETTERS[(LETTERS.index(query.gold) + 1) % 4]

        text = f"Let me work through this step by step.\nAnswer: {answer}"
        in_tok = estimate_tokens(prompt)
        out_tok = 150 if query is not None and query.dataset == "gsm8k" else 60
        latency = cfg.fake_latency_ms * rng.uniform(0.7, 1.4)
        return LLMResponse(
            model=cfg.name,
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
            cost_usd=cost_usd(cfg, in_tok, out_tok),
            stop_reason="stop",
        )

    async def chat(self, messages: list[dict[str, str]], *, sample: bool = False) -> LLMResponse:
        return await self.complete(messages[-1]["content"], sample=sample)
