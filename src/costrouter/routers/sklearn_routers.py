"""Learned routers: a scikit-learn estimator over one feature view, tuned by cross-validation
and then probability-calibrated so that p = 0.8 really means "right about 80% of the time"."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from costrouter.features import FEATURE_NAMES, Features
from costrouter.routers.base import Router

View = Literal["handcrafted", "embedding", "both"]
N_HC = len(FEATURE_NAMES)


def _matrix(features: Features, view: View) -> np.ndarray:
    if view == "handcrafted":
        return features.handcrafted
    if view == "embedding":
        return features.embedding
    return np.hstack([features.handcrafted, features.embedding])


class SklearnRouter(Router):
    def __init__(
        self,
        name: str,
        view: View,
        estimator: Any,
        param_grid: dict[str, list[Any]],
        seed: int = 42,
    ):
        self.name = name
        self.view = view
        self.estimator = estimator
        self.param_grid = param_grid
        self.seed = seed
        self.best_params_: dict[str, Any] = {}
        self.cv_auc_: float | None = None

    def fit(self, features: Features, y: np.ndarray) -> SklearnRouter:
        X, y = _matrix(features, self.view), np.asarray(y, dtype=int)
        n_splits = max(2, min(5, int(np.bincount(y).min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        search = GridSearchCV(clone(self.estimator), self.param_grid, scoring="roc_auc", cv=cv)
        search.fit(X, y)
        self.best_params_, self.cv_auc_ = search.best_params_, float(search.best_score_)
        # Isotonic calibration needs plenty of data; Platt (sigmoid) is safer for small sets.
        # ensemble=False: one model refit on all rows, calibrated on cross-validated scores,
        # so a live prediction runs one estimator instead of n_splits of them.
        method = "isotonic" if len(y) >= 1000 else "sigmoid"
        self.model_ = CalibratedClassifierCV(
            search.best_estimator_, method=method, cv=cv, ensemble=False
        )
        self.model_.fit(X, y)
        self.calibration_ = method
        return self

    def predict_proba(self, features: Features) -> np.ndarray:
        return self.model_.predict_proba(_matrix(features, self.view))[:, 1]


def default_routers(seed: int = 42) -> list[Router]:
    """R2-R6 from the implementation doc (R0 random and R1 length live in baselines)."""
    return [
        SklearnRouter(
            "logreg_handcrafted",
            "handcrafted",
            Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            seed,
        ),
        SklearnRouter(
            "logreg_embedding",
            "both",
            # Scale every column: L2-normalised embedding values are tiny (about 0.05), so left
            # unscaled their weights get regularised to zero and only the handcrafted ones count.
            Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            seed,
        ),
        SklearnRouter(
            "knn_embedding",
            "embedding",
            KNeighborsClassifier(metric="cosine", weights="distance"),
            {"n_neighbors": [10, 25, 50]},
            seed,
        ),
        SklearnRouter(
            "gbm_both",
            "both",
            Pipeline(
                [
                    (
                        "prep",
                        ColumnTransformer(
                            [("emb", PCA(n_components=32, random_state=seed), slice(N_HC, None))],
                            remainder="passthrough",
                        ),
                    ),
                    ("clf", HistGradientBoostingClassifier(random_state=seed)),
                ]
            ),
            {"clf__max_depth": [3, None], "clf__learning_rate": [0.05, 0.1]},
            seed,
        ),
        SklearnRouter(
            "mlp_embedding",
            "embedding",
            MLPClassifier(
                hidden_layer_sizes=(256,), early_stopping=True, max_iter=500, random_state=seed
            ),
            {"alpha": [1e-4, 1e-2, 1.0]},
            seed,
        ),
    ]
