"""Sentence embeddings on CPU, cached to disk by text hash so each text is embedded once."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def embed(
    texts: Sequence[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64
) -> np.ndarray:
    """L2-normalised float32 embeddings, one row per text."""
    vectors = _load(model_name).encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def cached_embed(
    texts: Sequence[str], cache_dir: Path, model_name: str = DEFAULT_MODEL
) -> np.ndarray:
    """Like embed(), but reuses vectors stored in cache_dir/<model>.npz from earlier runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (model_name.replace("/", "__") + ".npz")
    store: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as f:
            store = dict(zip(f["keys"].tolist(), f["vectors"], strict=True))

    keys = [_text_key(t) for t in texts]
    missing = sorted({k: t for k, t in zip(keys, texts, strict=True) if k not in store}.items())
    if missing:
        new = embed([t for _, t in missing], model_name)
        store.update({k: v for (k, _), v in zip(missing, new, strict=True)})
        np.savez(path, keys=np.array(list(store)), vectors=np.stack(list(store.values())))
    return np.stack([store[k] for k in keys])
