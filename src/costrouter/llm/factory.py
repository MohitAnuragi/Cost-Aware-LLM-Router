from __future__ import annotations

from collections.abc import Sequence

from costrouter.config import ModelConfig
from costrouter.data.schema import Query
from costrouter.llm.base import LLMClient


def build_client(cfg: ModelConfig, queries: Sequence[Query] = ()) -> LLMClient:
    """`queries` is only used by the fake provider, which needs the answer key."""
    if cfg.provider == "openai_compat":
        from costrouter.llm.openai_compat_client import OpenAICompatClient

        return OpenAICompatClient(cfg)
    from costrouter.llm.fake_client import FakeClient

    return FakeClient(cfg, queries)
