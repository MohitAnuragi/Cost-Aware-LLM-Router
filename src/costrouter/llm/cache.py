"""SQLite response cache: no query is ever paid for twice, and every experiment replays
from here instead of calling an API."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from costrouter.llm.base import LLMResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    call_type TEXT NOT NULL,
    sample_idx INTEGER NOT NULL,
    template_version TEXT NOT NULL,
    text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    stop_reason TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_model ON responses (model, template_version);
"""


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # A generous lock timeout lets two collection runs share one cache file.
        self._conn = sqlite3.connect(path, timeout=60)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @staticmethod
    def make_key(
        *,
        model: str,
        template_version: str,
        call_type: str,
        sample_idx: int,
        prompt: str,
        params: dict[str, Any],
    ) -> str:
        payload = json.dumps(
            [model, template_version, call_type, sample_idx, prompt, params], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def has(self, key: str) -> bool:
        return (
            self._conn.execute("SELECT 1 FROM responses WHERE key = ?", (key,)).fetchone()
            is not None
        )

    def put(
        self,
        key: str,
        *,
        query_id: str,
        role: str,
        call_type: str,
        sample_idx: int,
        template_version: str,
        response: LLMResponse,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                query_id,
                role,
                response.model,
                call_type,
                sample_idx,
                template_version,
                response.text,
                response.input_tokens,
                response.output_tokens,
                response.latency_ms,
                response.cost_usd,
                response.stop_reason,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def records(self, models: list[str], template_version: str) -> list[dict[str, Any]]:
        placeholders = ",".join("?" * len(models))
        rows = self._conn.execute(
            f"SELECT * FROM responses WHERE model IN ({placeholders}) AND template_version = ?",
            (*models, template_version),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()
