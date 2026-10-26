"""The provider-neutral client interface every model call goes through."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from costrouter.config import ModelConfig


@dataclass(frozen=True)
class LLMResponse:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    stop_reason: str | None


class LLMClient(Protocol):
    config: ModelConfig

    async def complete(self, prompt: str, *, sample: bool = False) -> LLMResponse:
        """One single-turn completion. sample=True asks for a diverse sample
        (self-consistency) instead of the deterministic answer."""
        ...

    async def chat(self, messages: list[dict[str, str]], *, sample: bool = False) -> LLMResponse:
        """Same, for a whole conversation (used by the serving API)."""
        ...


class MissingAPIKeyError(RuntimeError):
    pass


def require_api_key(cfg: ModelConfig) -> str | None:
    if not cfg.api_key_env:
        return None
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise MissingAPIKeyError(
            f"{cfg.api_key_env} is not set. Copy .env.example to .env and add your key "
            f"(model {cfg.name}), or run with --dry-run to use fake models."
        )
    return key


def request_temperature(cfg: ModelConfig, sample: bool) -> float | None:
    """None when the provider rejects sampling parameters, so none are sent."""
    if cfg.temperature is None:
        return None
    return cfg.sample_temperature if sample else cfg.temperature


def estimate_tokens(text: str) -> int:
    """Rough pre-call estimate for rate limiting (about 4 characters per token)."""
    return max(1, len(text) // 4)
