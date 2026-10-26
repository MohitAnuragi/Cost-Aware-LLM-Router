"""Feature bundles: everything a router may look at, computed from the user's text only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from costrouter.features.embeddings import DEFAULT_MODEL, cached_embed, embed
from costrouter.features.handcrafted import FEATURE_NAMES, handcrafted_matrix


@dataclass(frozen=True)
class Features:
    handcrafted: np.ndarray  # (n, len(FEATURE_NAMES))
    embedding: np.ndarray  # (n, d), L2-normalised

    def __len__(self) -> int:
        return len(self.handcrafted)

    def take(self, idx: np.ndarray) -> Features:
        return Features(self.handcrafted[idx], self.embedding[idx])


def build_features(
    texts: Sequence[str], cache_dir: Path | None = None, model_name: str = DEFAULT_MODEL
) -> Features:
    vectors = cached_embed(texts, cache_dir, model_name) if cache_dir else embed(texts, model_name)
    return Features(handcrafted_matrix(texts), vectors)


__all__ = ["FEATURE_NAMES", "Features", "build_features"]
