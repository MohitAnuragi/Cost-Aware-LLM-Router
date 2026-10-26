import numpy as np
import pytest
from fastapi.testclient import TestClient

from costrouter.config import ROLES
from costrouter.data.prompts import render_gsm8k, render_mmlu, user_text
from costrouter.features import FEATURE_NAMES, Features
from costrouter.features.handcrafted import handcrafted_matrix
from costrouter.llm.factory import build_client
from costrouter.routers.base import Router
from costrouter.routers.registry import all_routers
from costrouter.serve.app import create_app
from costrouter.serve.service import RouterService, final_answer


def _features(n: int, seed: int = 0) -> tuple[Features, np.ndarray]:
    rng = np.random.default_rng(seed)
    hc = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    emb = rng.normal(size=(n, 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    y = (hc[:, 0] + emb[:, 0] + 0.3 * rng.normal(size=n)) > 0
    return Features(hc, emb), y


def test_user_text_strips_only_the_template():
    assert user_text(render_gsm8k("What is 2 + 2?")) == "What is 2 + 2?"
    mmlu = user_text(render_mmlu("Pick one.", ["a", "b", "c", "d"]))
    assert mmlu.startswith("Pick one.") and "D. d" in mmlu and "Think briefly" not in mmlu


def test_handcrafted_features_shape_and_signal():
    m = handcrafted_matrix(["What is 15% of 80?", "How many apples?\nA. 1\nB. 2\nC. 3\nD. 4"])
    assert m.shape == (2, len(FEATURE_NAMES))
    assert m[1, FEATURE_NAMES.index("n_options")] == 4
    assert m[0, FEATURE_NAMES.index("has_percent_or_currency")] == 1


@pytest.mark.parametrize("router", all_routers(seed=0), ids=lambda r: r.name)
def test_every_router_fits_predicts_and_roundtrips(router, tmp_path):
    feats, y = _features(160)
    router.fit(feats, y)
    p = router.predict_proba(feats)
    assert p.shape == (160,) and np.all((p >= 0) & (p <= 1))
    router.save(tmp_path / "r.joblib")
    assert np.allclose(Router.load(tmp_path / "r.joblib").predict_proba(feats), p)


def test_learned_router_beats_chance():
    feats, y = _features(400)
    test_feats, test_y = _features(200, seed=1)
    from sklearn.metrics import roc_auc_score

    router = next(r for r in all_routers(0) if r.name == "logreg_embedding")
    router.fit(feats, y)
    assert roc_auc_score(test_y, router.predict_proba(test_feats)) > 0.75


def test_final_answer_normalises():
    assert final_answer("so 12 apples.\nAnswer: 12") == final_answer("Answer: 12.0")
    assert final_answer("A. x\nB. y\nAnswer: B") == "B"


class _StubRouter(Router):
    name = "stub"

    def __init__(self, p: float):
        self.p = p

    def fit(self, features, y):
        return self

    def predict_proba(self, features):
        return np.full(len(features), self.p)


def _service(p: float, fake_profile, queries) -> RouterService:
    clients = {r: build_client(fake_profile.role(r), queries) for r in ROLES}
    return RouterService(
        fake_profile,
        _StubRouter(p),
        clients,
        presets={"quality": 0.5, "balanced": 0.4},
        cascade={"tau_low": 0.3, "tau_high": 0.7},
        featurize=lambda texts: Features(np.zeros((len(texts), 1)), np.zeros((len(texts), 1))),
    )


@pytest.mark.parametrize(
    ("p", "zone", "model"), [(0.9, "small", "fake-small"), (0.1, "large", "fake-large")]
)
def test_chat_completion_routes_by_probability(p, zone, model, fake_profile, queries):
    app = create_app(_service(p, fake_profile, queries))
    client = TestClient(app)
    body = {"model": "router-cascade", "messages": [{"role": "user", "content": queries[0].prompt}]}
    r = client.post("/v1/chat/completions", json=body).json()
    assert r["object"] == "chat.completion"
    assert r["x_router"]["zone"] == zone and r["model"] == model
    assert r["choices"][0]["message"]["content"]
    assert client.get("/metrics").json()["requests"] == 1


def test_middle_zone_runs_the_self_check(fake_profile, queries):
    service = _service(0.5, fake_profile, queries)
    client = TestClient(create_app(service))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "router-cascade",
            "messages": [{"role": "user", "content": queries[1].prompt}],
        },
    ).json()
    assert r["x_router"]["zone"] == "check"
    assert r["x_router"]["served_by"] in ("small", "large")


def test_route_endpoint_and_presets(fake_profile, queries):
    client = TestClient(create_app(_service(0.45, fake_profile, queries)))
    assert (
        client.post("/route", json={"text": "hi", "model": "router-quality"}).json()["zone"]
        == "large"
    )
    assert (
        client.post("/route", json={"text": "hi", "model": "router-balanced"}).json()["zone"]
        == "small"
    )
    assert (
        client.post("/route", json={"text": "hi", "model": "router-0.2"}).json()["zone"] == "small"
    )
    assert client.get("/health").json()["status"] == "ok"
