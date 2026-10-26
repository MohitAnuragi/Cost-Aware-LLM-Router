"""Every router maps features to p = P(small model answers correctly). One threshold tau turns
p into a decision, so swapping routers never changes the rest of the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np

from costrouter.features import Features


class Router(ABC):
    name: str

    @abstractmethod
    def fit(self, features: Features, y: np.ndarray) -> Router: ...

    @abstractmethod
    def predict_proba(self, features: Features) -> np.ndarray:
        """P(small model is correct), one value per row."""

    def route(self, features: Features, tau: float) -> np.ndarray:
        return np.where(self.predict_proba(features) >= tau, "small", "large")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Router:
        return joblib.load(path)
