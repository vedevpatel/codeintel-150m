from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_REPO = FIXTURES / "sample_repo"


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CODEINTEL_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set CODEINTEL_NETWORK_TESTS=1 to download and test real checkpoints")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


class HashingModel:
    """Deterministic bag-of-tokens embedder: lexical, but exercises the dense code paths without weights."""

    name = "hashing"
    dim = 64

    def __init__(self):
        self.calls = 0
        self.encoded = 0

    @property
    def fingerprint(self) -> str:
        return "hashing-v1"

    def _vec(self, text: str) -> np.ndarray:
        from codeintel.retrieval.bm25 import code_tokens

        v = np.zeros(self.dim, dtype=np.float32)
        for tok in code_tokens(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_queries(self, texts):
        return self.encode_documents(texts)

    def encode_documents(self, texts):
        self.calls += 1
        self.encoded += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


@pytest.fixture
def hashing_model():
    return HashingModel()


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory) -> Path:
    """A randomly initialised 2-layer BERT saved like a fine-tuned codeintel checkpoint (adapter: hf)."""
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    directory = tmp_path_factory.mktemp("tiny_bert")
    words = sorted(
        {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
        | set("abcdefghijklmnopqrstuvwxyz0123456789()[]{}:=.,_+-*/\"'#")
        | {"def", "return", "class", "self", "import", "function", "const", "parse", "json", "file", "retry", "http"}
    )
    (directory / "vocab.txt").write_text("\n".join(words) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(str(directory))
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=128, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    BertModel(config, add_pooling_layer=False).save_pretrained(str(directory), safe_serialization=True)
    spec = {
        "pooling": "mean", "normalize": True, "query_prefix": "query: ", "document_prefix": "",
        "recommended_max_length": 64, "requires_remote_code": False,
    }
    (directory / "codeintel_adapter.json").write_text(json.dumps(spec))
    return directory


@pytest.fixture
def tiny_model_cfg(tiny_bert_dir):
    from codeintel.config import ModelConfig

    return ModelConfig(name="tiny", adapter="hf", repo_id="local/tiny", local_dir=str(tiny_bert_dir), device="cpu", batch_size=4)


def git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True, env=env).stdout


@pytest.fixture
def git_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "math_utils.py").write_text(
        'def add(a, b):\n    """Add two numbers together."""\n    return a + b\n\n\n'
        'def scale(values, factor):\n    """Multiply every value by a constant factor."""\n    return [v * factor for v in values]\n'
    )
    (repo / "README.md").write_text("demo\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial import")
    (repo / "pkg" / "math_utils.py").write_text(
        'def add(a, b):\n    """Add two numbers together."""\n    return a + b\n\n\n'
        'def scale(values, factor):\n    """Multiply every value by a constant factor."""\n    if factor == 0:\n        return [0 for _ in values]\n    return [v * factor for v in values]\n'
    )
    git(repo, "commit", "-q", "-am", "Fix scale returning negative zeros when the factor is zero")
    return repo
