"""Stratified train / val / test splits plus the pilot subset, saved as one JSON file."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from costrouter.data.schema import Query

SPLITS = ("train", "val", "test")


def make_splits(queries: list[Query], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Split within each (dataset, subject) group so every split has the same mix."""
    if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    rng = random.Random(seed)
    groups: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for q in queries:
        groups[(q.dataset, q.subject)].append(q.id)

    splits: dict[str, list[str]] = {s: [] for s in SPLITS}
    for key in sorted(groups, key=str):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n_train = round(len(ids) * ratios["train"])
        n_val = round(len(ids) * ratios["val"])
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])
    return splits


def pick_pilot(
    queries: list[Query], train_ids: list[str], per_dataset: int, seed: int
) -> list[str]:
    """Pilot queries come from the train split only, so they never leak into evaluation."""
    rng = random.Random(seed + 1)
    train = set(train_ids)
    pilot: list[str] = []
    for dataset in ("gsm8k", "mmlu"):
        pool = sorted(q.id for q in queries if q.dataset == dataset and q.id in train)
        pilot.extend(rng.sample(pool, min(per_dataset, len(pool))))
    return pilot


def save_splits(splits: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1), encoding="utf-8")


def load_splits(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `router data prepare` first")
    return json.loads(path.read_text(encoding="utf-8"))
