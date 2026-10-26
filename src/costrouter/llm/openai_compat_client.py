"""Client for any OpenAI-compatible chat API (Groq, Together, OpenRouter, Ollama, ...)."""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI

from costrouter.config import ModelConfig
from costrouter.eval.cost import cost_usd
from costrouter.llm.base import LLMResponse, estimate_tokens, request_temperature, require_api_key
from costrouter.llm.ratelimit import RateLimiter


class OpenAICompatClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        # The SDK retries 429s and 5xx with backoff and honours retry-after headers.
        self._client = AsyncOpenAI(
            base_url=config.base_url, api_key=require_api_key(config), max_retries=8
        )
        self._limiter = RateLimiter(config.rpm, config.tpm)

    async def complete(self, prompt: str, *, sample: bool = False) -> LLMResponse:
        return await self.chat([{"role": "user", "content": prompt}], sample=sample)

    async def chat(self, messages: list[dict[str, str]], *, sample: bool = False) -> LLMResponse:
        cfg = self.config
        params: dict[str, Any] = dict(cfg.extra_params)
        temperature = request_temperature(cfg, sample)
        if temperature is not None:
            params["temperature"] = temperature

        text_in = "\n".join(m["content"] for m in messages)
        reservation = await self._limiter.acquire(estimate_tokens(text_in) + cfg.est_output_tokens)
        t0 = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=cfg.name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=cfg.max_tokens,
            **params,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        in_tok = resp.usage.prompt_tokens if resp.usage else estimate_tokens(text_in)
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        reservation[1] = in_tok + out_tok
        choice = resp.choices[0]
        return LLMResponse(
            model=cfg.name,
            text=choice.message.content or "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cost_usd=cost_usd(cfg, in_tok, out_tok),
            stop_reason=choice.finish_reason,
        )
