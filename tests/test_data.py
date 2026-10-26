import random

from costrouter.data.loaders import gsm8k_gold, sample_per_subject
from costrouter.data.schema import load_queries, save_queries
from costrouter.data.splits import SPLITS, make_splits, pick_pilot


def test_gsm8k_gold():
    assert gsm8k_gold("Step 1 <<2*3=6>>\n#### 1,200") == "1200"


def test_mmlu_row_rendering(queries):
    q = next(q for q in queries if q.dataset == "mmlu")
    assert q.gold == "A"
    assert "A. w" in q.prompt and "D. z" in q.prompt
    assert q.question == "Question 0?"


def test_splits_are_disjoint_and_complete(queries):
    splits = make_splits(queries, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    ids = [i for s in SPLITS for i in splits[s]]
    assert len(ids) == len(set(ids)) == len(queries)
    assert splits == make_splits(queries, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)


def test_pilot_comes_from_train(queries):
    splits = make_splits(queries, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    pilot = pick_pilot(queries, splits["train"], per_dataset=5, seed=1)
    assert len(pilot) == 10
    assert set(pilot) <= set(splits["train"])


def test_sample_per_subject_caps_each_subject(queries):
    mmlu = [q for q in queries if q.dataset == "mmlu"]
    picked = sample_per_subject(mmlu, 3, random.Random(0))
    assert len(picked) == 6
    assert {q.subject for q in picked} == {"astronomy", "virology"}


def test_queries_roundtrip(tmp_path, queries):
    path = tmp_path / "q.parquet"
    save_queries(queries, path)
    loaded = load_queries(path)
    assert loaded == queries
    assert loaded[0].subject is None
