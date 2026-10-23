"""Typed configuration loaded from YAML.

Unknown keys are rejected so that a typo (e.g. `trust_remote_cod`) fails loudly instead of silently
falling back to a default.
"""

from __future__ import annotations

import dataclasses
import os
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ConfigError(ValueError):
    pass


def is_commit_sha(revision: str | None) -> bool:
    return bool(revision) and bool(_COMMIT_SHA.match(revision))


@dataclass
class ModelConfig:
    name: str
    adapter: str  # "coderank" | "modernbert" | "hf" (generic, for fine-tuned checkpoints)
    repo_id: str
    # Full 40-character commit SHA on the Hub. Branch names and tags are mutable and rejected.
    revision: str | None = None
    # Executing repository-provided Python is opt-in per model.
    trust_remote_code: bool = False
    # filename -> sha256 of every reviewed remote-code file. Required when trust_remote_code is true;
    # loading aborts if any downloaded .py file is missing from this map or its hash differs.
    remote_code_sha256: dict[str, str] = field(default_factory=dict)
    # Load from a local directory (an already-downloaded snapshot or a fine-tuned checkpoint).
    local_dir: str | None = None
    device: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    max_length: int | None = None  # None -> adapter's recommended value
    batch_size: int = 32
    # None -> the adapter's recommended prefix. Set to "" to disable explicitly.
    query_prefix: str | None = None
    document_prefix: str | None = None

    def validate(self) -> None:
        if self.local_dir is None and not is_commit_sha(self.revision):
            raise ConfigError(
                f"model '{self.name}': revision must be a full 40-char commit SHA, got {self.revision!r}"
            )
        if self.trust_remote_code and not self.remote_code_sha256:
            raise ConfigError(
                f"model '{self.name}': trust_remote_code requires remote_code_sha256 for every reviewed file"
            )
        if self.batch_size <= 0:
            raise ConfigError(f"model '{self.name}': batch_size must be positive")


@dataclass
class TaskConfig:
    name: str
    type: str  # "retrieval" | "localization"
    path: str
    # localization only: directory holding repository checkouts referenced by cases.
    repos_dir: str | None = None
    # Free-form note recorded in reports, e.g. known contamination of public benchmarks.
    contamination_note: str = ""
    max_queries: int | None = None
    max_corpus: int | None = None


@dataclass
class EvaluationConfig:
    models: list[str]  # paths to model YAML files
    tasks: list[TaskConfig]
    output_dir: str = "artifacts/eval"
    ks: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    retrieval_modes: list[str] = field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    primary_metric: str = "ndcg@10"
    bootstrap_samples: int = 1000
    seed: int = 13
    cache_dir: str = "artifacts/cache"


@dataclass
class TrainingConfig:
    model: str  # path to model YAML (the base checkpoint)
    train_files: list[str]
    output_dir: str = "artifacts/checkpoints/run"
    dev_task: TaskConfig | None = None
    # Sampling weight per training source, e.g. {"commit": 2.0, "docstring": 1.0}. Missing -> 1.0.
    source_weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 32
    # GradCache sub-batch size: lets `batch_size` (and therefore the number of in-batch negatives) exceed
    # what fits in memory. None disables it.
    mini_batch_size: int | None = None
    hard_negatives: int = 1
    temperature: float = 0.05
    symmetric_loss: bool = True
    matryoshka_dims: list[int] = field(default_factory=list)
    # Weight of the KL term that keeps the student's similarity distribution close to the frozen base model.
    distill_weight: float = 0.0
    distill_temperature: float = 0.05
    max_query_length: int = 256
    max_document_length: int = 512
    gradient_checkpointing: bool = False
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    max_grad_norm: float = 1.0
    eval_every: int = 500
    log_every: int = 25
    seed: int = 13
    # WiSE-FT style interpolation with the base weights applied to the final checkpoint (1.0 = no-op).
    interpolate_alpha: float = 1.0


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _coerce(non_none[0], value, where)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected mapping, got {type(value).__name__}")
        return from_dict(tp, value, where)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected list, got {type(value).__name__}")
        return [_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected mapping, got {type(value).__name__}")
        return {str(k): _coerce(args[1], v, f"{where}.{k}") for k, v in value.items()}
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in (int, float, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {type(value).__name__} ({value!r})")
    if tp is int and isinstance(value, bool):
        raise ConfigError(f"{where}: expected int, got bool")
    return value


def from_dict(cls: type[T], data: dict[str, Any], where: str | None = None) -> T:
    where = where or cls.__name__
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    cfg = from_dict(ModelConfig, load_yaml(path), where=str(path))
    if env_device := os.environ.get("CODEINTEL_DEVICE"):
        cfg.device = env_device
    cfg.validate()
    return cfg


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    return from_dict(EvaluationConfig, load_yaml(path), where=str(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return from_dict(TrainingConfig, load_yaml(path), where=str(path))
