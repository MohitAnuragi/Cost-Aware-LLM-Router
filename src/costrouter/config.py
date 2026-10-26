"""Typed loading of the YAML files in configs/ and the project's standard paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator

Role = Literal["small", "large"]
ROLES: tuple[Role, ...] = ("small", "large")


def project_root() -> Path:
    """Commands run from the repo root; COSTROUTER_ROOT overrides it (used by tests)."""
    return Path(os.environ.get("COSTROUTER_ROOT", Path.cwd()))


def config_dir() -> Path:
    return project_root() / "configs"


def data_dir() -> Path:
    return project_root() / "data"


def reports_dir() -> Path:
    return project_root() / "reports"


class ModelConfig(BaseModel):
    name: str
    provider: Literal["openai_compat", "fake"]
    base_url: str | None = None
    api_key_env: str | None = None
    usd_in_per_m: float
    usd_out_per_m: float
    max_tokens: int = 1024
    # None means the provider rejects sampling parameters, so none are sent.
    temperature: float | None = 0.0
    sample_temperature: float = 0.7
    effort: str | None = None
    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tpd: int | None = None
    # Provider-specific request fields, e.g. {"reasoning_effort": "low"} for gpt-oss on Groq.
    extra_params: dict[str, Any] = {}
    est_output_tokens: int = 400
    concurrency: int = 4
    # Fake provider only.
    skill: float = 0.7
    fake_latency_ms: float = 500.0


class Profile(BaseModel):
    small: ModelConfig
    large: ModelConfig

    @model_validator(mode="after")
    def _distinct_models(self) -> Profile:
        if self.small.name == self.large.name:
            raise ValueError("small and large must be different models")
        return self

    def role(self, role: Role) -> ModelConfig:
        return self.small if role == "small" else self.large


class ModelsConfig(BaseModel):
    prices_as_of: str
    active_profile: str
    profiles: dict[str, Profile]

    def profile(self, name: str | None = None) -> Profile:
        key = name or self.active_profile
        if key not in self.profiles:
            raise KeyError(f"unknown profile {key!r}; choose from {sorted(self.profiles)}")
        return self.profiles[key]


class DatasetsConfig(BaseModel):
    seed: int = 42
    gsm8k_test: int | Literal["all"] = "all"
    gsm8k_train: int = 0
    mmlu_per_subject: int = 30
    pilot_per_dataset: int = 100
    split: dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models_config(path: Path | None = None) -> ModelsConfig:
    return ModelsConfig.model_validate(_read_yaml(path or config_dir() / "models.yaml"))


def load_datasets_config(path: Path | None = None) -> DatasetsConfig:
    return DatasetsConfig.model_validate(_read_yaml(path or config_dir() / "datasets.yaml"))
