"""Export an adapter to ONNX (pooling + normalisation inside the graph) and INT8 dynamic quantisation.

Every export is verified against the PyTorch model on texts of several lengths, including lengths longer
than the tracing input, and the verification is written to export_report.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from codeintel.config import ModelConfig
from codeintel.models.base import (
    ADAPTER_SPEC_FILE,
    EmbeddingModel,
    TransformerEmbedder,
    load_adapter_spec,
    pool,
)

VERIFY_TEXTS = [
    "def add(a, b):\n    return a + b",
    "Represent this query for searching relevant code: open a file and read json",
    "export function debounce<T extends (...args: any[]) => void>(fn: T, ms = 100) {\n"
    "  let t: ReturnType<typeof setTimeout> | undefined;\n"
    "  return (...args: Parameters<T>) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };\n}",
    "class LRUCache:\n" + "".join(f"    def m{i}(self, key):\n        return self.data.get(key, {i})\n" for i in range(60)),
]


def quantize_int8(src: Path, dst: Path, fp32_pattern: str = "") -> list[str]:
    """Dynamic INT8 (per-channel weights) of MatMul nodes, keeping nodes matching `fp32_pattern` in fp32."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    graph = onnx.load(str(src), load_external_data=False).graph
    excluded = [n.name for n in graph.node if n.op_type == "MatMul" and fp32_pattern and re.search(fp32_pattern, n.name)]
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=excluded,
    )
    return excluded


class PooledEncoder(torch.nn.Module):
    def __init__(self, embedder: TransformerEmbedder):
        super().__init__()
        self.model = embedder.model
        self.pooling = embedder.spec.pooling
        self.normalize = embedder.spec.normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emb = pool(hidden, attention_mask, self.pooling)
        return F.normalize(emb, p=2, dim=-1) if self.normalize else emb


def _ort_embed(session, tokenizer, texts: list[str], max_length: int) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np", return_token_type_ids=False)
    feeds = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
    return session.run(["embedding"], feeds)[0]


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def export_onnx(
    embedder: TransformerEmbedder,
    out_dir: str | Path,
    opset: int = 17,
    quantize: bool = True,
    trace_length: int = 64,
    dynamo: bool = False,
    full_int8: bool = False,
) -> dict:
    import onnxruntime as ort

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder.model.to(device="cpu", dtype=torch.float32).eval()
    embedder.device = torch.device("cpu")
    wrapper = PooledEncoder(embedder).eval()
    dummy = embedder.tokenizer(
        ["x " * trace_length, "y"], padding="max_length", truncation=True, max_length=trace_length,
        return_tensors="pt", return_token_type_ids=False,
    )
    fp32_path = out_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["embedding"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "embedding": {0: "batch"}},
            opset_version=opset,
            dynamo=dynamo,
            do_constant_folding=True,
        )
    reference = embedder.encode_documents(VERIFY_TEXTS)
    report: dict = {"opset": opset, "max_length": embedder.max_length, "verify_texts": len(VERIFY_TEXTS), "variants": {}}
    variants = {"fp32": fp32_path}
    if quantize:
        int8_path = out_dir / "model_int8.onnx"
        excluded = quantize_int8(fp32_path, int8_path, "" if full_int8 else embedder.spec.int8_fp32_nodes)
        report["int8_fp32_nodes"] = len(excluded)
        variants["int8"] = int8_path
    for name, path in variants.items():
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = _ort_embed(session, embedder.tokenizer, VERIFY_TEXTS, embedder.max_length)
        cos = _cosines(got, reference)
        report["variants"][name] = {
            "path": path.name,
            "bytes": path.stat().st_size + sum(p.stat().st_size for p in out_dir.glob(path.name + ".data")),
            "min_cosine_vs_torch": float(cos.min()),
            "mean_cosine_vs_torch": float(cos.mean()),
        }
    embedder.tokenizer.save_pretrained(str(out_dir))
    spec = embedder.spec.to_dict() | {"query_prefix": embedder.query_prefix, "document_prefix": embedder.document_prefix}
    (out_dir / ADAPTER_SPEC_FILE).write_text(json.dumps(spec, indent=2))
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


class OnnxEmbedder(EmbeddingModel):
    """CPU inference from an export directory (model.onnx / model_int8.onnx + tokenizer + adapter spec)."""

    def __init__(self, cfg: ModelConfig, variant: str | None = None, threads: int | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.cfg, self.name = cfg, cfg.name
        directory = Path(cfg.local_dir or "")
        self.spec = load_adapter_spec(directory)
        raw = json.loads((directory / ADAPTER_SPEC_FILE).read_text())
        self.query_prefix = raw.get("query_prefix", self.spec.query_prefix) if cfg.query_prefix is None else cfg.query_prefix
        self.document_prefix = raw.get("document_prefix", self.spec.document_prefix) if cfg.document_prefix is None else cfg.document_prefix
        variant = variant or ("int8" if cfg.dtype == "int8" else "fp32")
        self.model_path = directory / ("model_int8.onnx" if variant == "int8" else "model.onnx")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory))
        self.max_length = cfg.max_length or self.spec.recommended_max_length
        self.batch_size = cfg.batch_size
        self._dim = int(self.session.get_outputs()[0].shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            import onnxruntime as ort

            from codeintel.models.base import sha256_file

            digest = sha256_file(self.model_path)[:16]
            self._fingerprint = f"onnx:{digest}|ort={ort.__version__}|maxlen={self.max_length}|docprefix={self.document_prefix!r}"
        return self._fingerprint

    _fingerprint: str | None = None

    def _encode(self, texts, prefix: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = np.argsort([-len(t) for t in texts], kind="stable")
        for i in range(0, len(order), self.batch_size):
            idx = order[i : i + self.batch_size]
            out[idx] = _ort_embed(self.session, self.tokenizer, [prefix + texts[j] for j in idx], self.max_length)
        return out

    def encode_queries(self, texts):
        return self._encode(list(texts), self.query_prefix)

    def encode_documents(self, texts):
        return self._encode(list(texts), self.document_prefix)
