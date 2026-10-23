"""Embedding model adapters.

Each candidate checkpoint gets its own adapter that pins the inference behaviour recommended by its
model card (pooling, normalization, prompts). Shared mechanics -- secure snapshot resolution,
remote-code hash verification, length-bucketed batching, device/dtype selection -- live here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from codeintel.config import ConfigError, ModelConfig

logger = logging.getLogger(__name__)

ADAPTER_SPEC_FILE = "codeintel_adapter.json"

# Only these files are ever fetched from the Hub. Pickle-based weights (*.bin, *.pt, *.ckpt) are never
# downloaded, so no code path can unpickle them.
SAFE_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.py",
    "*.model",
    "*.safetensors",
]
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")


@dataclass(frozen=True)
class AdapterSpec:
    """Inference contract for a checkpoint, taken from its model card / sentence-transformers config."""

    pooling: str  # cls | mean
    normalize: bool
    query_prefix: str
    document_prefix: str
    recommended_max_length: int
    requires_remote_code: bool = False
    # Regex over ONNX MatMul node names kept in fp32 by INT8 export (outlier-heavy layers).
    int8_fp32_nodes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class EmbeddingModel(ABC):
    """Minimal interface the retrieval/evaluation stack depends on."""

    name: str

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Stable identifier of everything that affects document embeddings (used as a cache key)."""

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if requested != "auto":
        if requested not in table:
            raise ConfigError(f"unknown dtype {requested!r}")
        return table[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # fp32 on CPU and MPS: half precision there is either slow (CPU) or numerically fragile (MPS).
    return torch.float32


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_snapshot(cfg: ModelConfig) -> Path:
    """Return a local directory for the model, downloading only safe file types at a pinned commit."""
    if cfg.local_dir:
        path = Path(cfg.local_dir).expanduser().resolve()
        if not path.is_dir():
            raise ConfigError(f"model '{cfg.name}': local_dir {path} does not exist")
        return path
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=cfg.repo_id, revision=cfg.revision, allow_patterns=SAFE_ALLOW_PATTERNS))
    return path


def verify_snapshot(cfg: ModelConfig, snapshot: Path) -> None:
    """Enforce weight-format and remote-code policies on a resolved snapshot directory."""
    pickles = [p for p in snapshot.rglob("*") if p.suffix in PICKLE_SUFFIXES]
    if pickles:
        raise ConfigError(
            f"model '{cfg.name}': refusing snapshot containing pickle-based files {[p.name for p in pickles]}"
        )
    if not list(snapshot.glob("*.safetensors")):
        raise ConfigError(f"model '{cfg.name}': no safetensors weights in {snapshot}")

    code_files = sorted(p for p in snapshot.rglob("*.py"))
    if not cfg.trust_remote_code:
        return
    pinned = cfg.remote_code_sha256
    seen = set()
    for p in code_files:
        rel = p.relative_to(snapshot).as_posix()
        seen.add(rel)
        expected = pinned.get(rel)
        if expected is None:
            raise ConfigError(f"model '{cfg.name}': remote code file {rel} has not been reviewed (no pinned hash)")
        actual = sha256_file(p)
        if actual != expected:
            raise ConfigError(
                f"model '{cfg.name}': remote code {rel} sha256 {actual} does not match reviewed {expected}"
            )
    missing = set(pinned) - seen
    if missing:
        raise ConfigError(f"model '{cfg.name}': reviewed remote code files missing from snapshot: {sorted(missing)}")

    # auto_map entries of the form "other/repo--module.Class" pull code from a different, unpinned repo.
    config_path = snapshot / "config.json"
    if config_path.exists():
        auto_map = json.loads(config_path.read_text()).get("auto_map", {})
        for target in auto_map.values():
            targets = target if isinstance(target, list) else [target]
            for t in targets:
                if t and "--" in t:
                    raise ConfigError(
                        f"model '{cfg.name}': auto_map references code in another repository ({t}); "
                        "vendor and pin that code instead"
                    )


def check_weights_loaded(model: torch.nn.Module, snapshot: Path) -> None:
    """Fail if any parameter of the instantiated model was not present in the checkpoint.

    Some remote-code loaders call `load_state_dict(strict=False)`, which would otherwise leave
    randomly-initialised layers in place without an error.
    """
    from safetensors import safe_open

    ckpt_keys: set[str] = set()
    for f in snapshot.glob("*.safetensors"):
        with safe_open(str(f), framework="pt") as handle:
            ckpt_keys.update(handle.keys())
    prefixes = {"model.", "bert.", f"{getattr(model, 'base_model_prefix', '') or 'model'}."}

    def candidates(name: str):
        yield name
        for p in prefixes:
            yield p + name
            if name.startswith(p):
                yield name[len(p) :]

    # `pooler.*` (BERT's tanh pooler) never affects last_hidden_state, which is what the adapters pool.
    missing = [
        name
        for name, _ in model.named_parameters()
        if not name.startswith("pooler.") and not any(c in ckpt_keys for c in candidates(name))
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters were not loaded from the checkpoint, e.g. {missing[:5]}")


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling mode {mode!r}")


class TransformerEmbedder(EmbeddingModel):
    """Bi-encoder around a Hugging Face encoder with a fixed `AdapterSpec`."""

    spec: AdapterSpec
    max_tokens_per_batch = 32768

    def __init__(self, cfg: ModelConfig, spec: AdapterSpec | None = None):
        from transformers import AutoModel, AutoTokenizer

        if spec is not None:
            self.spec = spec
        if self.spec.requires_remote_code and not cfg.trust_remote_code:
            raise ConfigError(
                f"model '{cfg.name}' requires remote code; set trust_remote_code: true and pin "
                "remote_code_sha256 after reviewing the code"
            )
        self.cfg = cfg
        self.name = cfg.name
        self.snapshot = resolve_snapshot(cfg)
        verify_snapshot(cfg, self.snapshot)
        self.check_snapshot_contract(self.snapshot)

        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        self.max_length = cfg.max_length or self.spec.recommended_max_length
        self.batch_size = cfg.batch_size
        self.query_prefix = self.spec.query_prefix if cfg.query_prefix is None else cfg.query_prefix
        self.document_prefix = self.spec.document_prefix if cfg.document_prefix is None else cfg.document_prefix

        self.model = AutoModel.from_pretrained(str(self.snapshot), trust_remote_code=cfg.trust_remote_code)
        # Passing the loaded config stops AutoTokenizer from re-resolving it (which would need remote code).
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), config=self.model.config, trust_remote_code=False)
        check_weights_loaded(self.model, self.snapshot)
        self.post_load()
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self._dim = int(getattr(self.model.config, "hidden_size", None) or self.model.config.n_embd)

    def post_load(self) -> None:
        """Compatibility fixes applied right after the weights are loaded (any adapter, incl. fine-tunes)."""
        if getattr(self.model.config, "model_type", "") == "nomic_bert":
            from codeintel.models.coderank import apply_transformers5_compat

            apply_transformers5_compat(self.model)

    def check_snapshot_contract(self, snapshot: Path) -> None:
        """Verify the snapshot's own sentence-transformers config agrees with this adapter's spec."""
        pooling_cfg = snapshot / "1_Pooling" / "config.json"
        if pooling_cfg.exists():
            data = json.loads(pooling_cfg.read_text())
            mode = "cls" if data.get("pooling_mode_cls_token") else "mean" if data.get("pooling_mode_mean_tokens") else "other"
            if mode != self.spec.pooling:
                raise ConfigError(f"model '{self.cfg.name}': snapshot pooling is {mode}, adapter expects {self.spec.pooling}")
        st_cfg = snapshot / "config_sentence_transformers.json"
        if st_cfg.exists():
            prompts = json.loads(st_cfg.read_text()).get("prompts", {}) or {}
            if prompts.get("query", "") != self.spec.query_prefix:
                raise ConfigError(
                    f"model '{self.cfg.name}': snapshot query prompt {prompts.get('query')!r} "
                    f"differs from adapter {self.spec.query_prefix!r}"
                )

    @property
    def dim(self) -> int:
        return self._dim

    _fingerprint: str | None = None

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            import transformers

            if self.cfg.local_dir:
                # Local checkpoints are mutable (e.g. a trainer's `best/`), so key on the weights themselves.
                weights = "".join(sha256_file(p) for p in sorted(self.snapshot.glob("*.safetensors")))
                ident = "local:" + hashlib.sha256(weights.encode()).hexdigest()[:16]
            else:
                ident = f"{self.cfg.repo_id}@{self.cfg.revision}"
            self._fingerprint = "|".join(
                [
                    ident,
                    f"torch={torch.__version__}",
                    f"transformers={transformers.__version__}",
                    f"pool={self.spec.pooling}",
                    f"norm={self.spec.normalize}",
                    f"maxlen={self.max_length}",
                    f"dtype={self.dtype}",
                    f"docprefix={hashlib.sha256(self.document_prefix.encode()).hexdigest()[:12]}",
                ]
            )
        return self._fingerprint

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Differentiable embedding of a tokenized batch (used for both inference and training)."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        emb = pool(out.last_hidden_state, attention_mask, self.spec.pooling)
        if self.spec.normalize:
            emb = F.normalize(emb.float(), p=2, dim=-1)
        return emb

    def tokenize(self, texts: Sequence[str], max_length: int | None = None) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.inference_mode()
    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out
        enc = self.tokenizer(
            [prefix + t for t in texts],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_token_type_ids=False,
        )
        ids = enc["input_ids"]
        lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        order = np.argsort(-lengths, kind="stable")
        i = 0
        while i < len(order):
            longest = int(lengths[order[i]])
            n = max(1, min(self.batch_size, self.max_tokens_per_batch // max(longest, 1)))
            idx = order[i : i + n]
            padded = self.tokenizer.pad(
                {"input_ids": [ids[j] for j in idx], "attention_mask": [enc["attention_mask"][j] for j in idx]},
                return_tensors="pt",
            )
            emb = self.embed(padded["input_ids"].to(self.device), padded["attention_mask"].to(self.device))
            out[idx] = emb.float().cpu().numpy()
            i += n
        return out

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.document_prefix)

    def save(self, output_dir: str | Path) -> None:
        """Save weights, tokenizer, and the adapter spec so the checkpoint reloads with adapter: hf."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir), safe_serialization=True)
        self.tokenizer.save_pretrained(str(output_dir))
        spec = self.spec.to_dict()
        spec["query_prefix"] = self.query_prefix
        spec["document_prefix"] = self.document_prefix
        spec["source"] = self.cfg.local_dir or f"{self.cfg.repo_id}@{self.cfg.revision}"
        (output_dir / ADAPTER_SPEC_FILE).write_text(json.dumps(spec, indent=2))


def load_adapter_spec(directory: str | Path) -> AdapterSpec:
    data = json.loads((Path(directory) / ADAPTER_SPEC_FILE).read_text())
    fields = AdapterSpec.__dataclass_fields__
    return AdapterSpec(**{k: v for k, v in data.items() if k in fields})


def load_model(cfg: ModelConfig) -> EmbeddingModel:
    from codeintel.models.coderank import CodeRankEmbedAdapter
    from codeintel.models.modernbert import GteModernBertAdapter

    if cfg.adapter == "coderank":
        return CodeRankEmbedAdapter(cfg)
    if cfg.adapter == "modernbert":
        return GteModernBertAdapter(cfg)
    if cfg.adapter == "onnx":
        from codeintel.models.export import OnnxEmbedder

        if not cfg.local_dir:
            raise ConfigError(f"model '{cfg.name}': adapter 'onnx' requires local_dir (an export directory)")
        return OnnxEmbedder(cfg)
    if cfg.adapter == "hf":
        if not cfg.local_dir:
            raise ConfigError(f"model '{cfg.name}': adapter 'hf' requires local_dir with {ADAPTER_SPEC_FILE}")
        return TransformerEmbedder(cfg, spec=load_adapter_spec(cfg.local_dir))
    raise ConfigError(f"model '{cfg.name}': unknown adapter {cfg.adapter!r}")
