"""Cost model: list-price dollars from the token counts each API actually returns."""

from __future__ import annotations

from costrouter.config import ModelConfig


def cost_usd(cfg: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * cfg.usd_in_per_m + output_tokens * cfg.usd_out_per_m) / 1e6
