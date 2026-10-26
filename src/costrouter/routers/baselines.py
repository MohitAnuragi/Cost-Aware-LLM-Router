"""Reference routers every learned router is compared against."""

from __future__ import annotations

import numpy as np

from costrouter.features import Features
from costrouter.routers.base import Router


class RandomRouter(Router):
    """Ignores the query. Swept over thresholds it traces the straight line between
    always-small and always-large; the eval also computes that line exactly."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, features: Features, y: np.ndarray) -> RandomRouter:
        return self

    def predict_proba(self, features: Features) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(features))


class LengthRouter(Router):
    """'Long equals hard': shorter questions are assumed easier. p is the share of training
    questions at least as long, so it lies in [0, 1] and falls as length grows."""

    name = "length"

    def fit(self, features: Features, y: np.ndarray) -> LengthRouter:
        self._train_lengths = np.sort(features.handcrafted[:, 0])  # log_chars
        return self

    def predict_proba(self, features: Features) -> np.ndarray:
        lengths = features.handcrafted[:, 0]
        rank = np.searchsorted(self._train_lengths, lengths, side="left")
        return 1.0 - rank / max(len(self._train_lengths), 1)
