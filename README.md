# codeintel-150m

A compact (~150M parameter) encoder-only bi-encoder for code: semantic code search, issue-to-code
localization, repository navigation, code-to-code retrieval, and context retrieval for coding agents.
Primary languages are Python and TypeScript; Java, Go, and Rust are parsed when their grammars are installed.

The goal is a general-purpose code embedding model with exceptional repository-localization performance,
not a localization-only model.

## Status

| Stage | State |
|---|---|
| Candidate adapters (CodeRankEmbed, gte-modernbert-base) | done; outputs match sentence-transformers reference to < 1e-6 |
| Indexing, BM25 / dense / hybrid retrieval, CLI | done |
| Benchmarks (temporal localization, docstring retrieval, CSN, CoSQA) | done; see [docs/model-selection.md](docs/model-selection.md) |
| Training stack (InfoNCE + hard negatives + GradCache + distillation + WiSE-FT) | done, smoke-tested on both candidates |
| ONNX fp32 / INT8 export | done; verified against PyTorch |
| Base checkpoint selection | both evaluated; gte-modernbert-base provisional, pending a pilot fine-tune ([docs/model-selection.md](docs/model-selection.md)) |
| Training data at scale, fine-tuning runs | next |

## Setup

```bash
uv sync --all-extras          # Python 3.10-3.13; torch, transformers>=4.48, tree-sitter, onnxruntime, ...
uv run pytest                 # offline tests (a tiny random BERT stands in for real checkpoints)
CODEINTEL_NETWORK_TESTS=1 uv run pytest -m network   # also load the real candidates
```

Runs on CPU, CUDA, and Apple Silicon (MPS); `device: auto` picks CUDA > MPS > CPU. See `.env.example`.

## Quick start

```bash
# Index a repository (omit --model for a lexical-only index)
uv run codeintel index path/to/repo --out artifacts/index/repo --model configs/models/gte-modernbert.yaml

# Search: chunk-, file-, or symbol-level results; dense, bm25, or hybrid (RRF) ranking
uv run codeintel search artifacts/index/repo "retry failed requests with backoff" \
    --model configs/models/gte-modernbert.yaml --level symbol --mode hybrid
```

Chunks are definition-aligned (functions, methods, class skeletons, module code, line windows for oversized
spans) and carry a `path::Qualified.name` header. Embeddings are cached by content hash in SQLite, so
re-indexing a repository, or indexing it at a nearby commit, only encodes changed chunks.

## Candidate checkpoints

Both candidates are pinned to full commit SHAs and each has its own adapter, which preserves its model
card's inference contract:

| | CodeRankEmbed | gte-modernbert-base |
|---|---|---|
| Hub id / revision | `nomic-ai/CodeRankEmbed` @ `3c4b608` | `Alibaba-NLP/gte-modernbert-base` @ `e7f32e3` |
| Params / dim / license | 137M / 768 / MIT | 149M / 768 / Apache-2.0 |
| Pooling | CLS | CLS |
| Normalisation | cosine (L2) | L2 |
| Query prompt | `Represent this query for searching relevant code: ` | none |
| Architecture | NomicBERT (remote code) | ModernBERT (native in transformers >= 4.48) |

The adapters check each snapshot's own `1_Pooling` and prompt config against the adapter's contract, and
loading fails if they differ.

**Remote code policy.** `trust_remote_code` is off unless a model config sets it. When it is on, the config
must also give a SHA-256 for every `.py` file in the snapshot. Loading aborts if any file is missing from
that list or its hash differs, or if `auto_map` points at code in another repository. Only `*.json`,
`*.txt`, `*.py`, `*.model`, and `*.safetensors` are downloaded, so pickle weights are never fetched. After
loading, every parameter is checked against the checkpoint keys.

These checks matter for CodeRankEmbed. Its remote `from_pretrained` ignores `revision` for weights and
falls back to `torch.load` without `weights_only`. Its local-directory branch loads with `strict=False`.
The remote code also calls `get_extended_attention_mask`, which transformers 5 removed, so
`models/coderank.py` adds a shim that reproduces the 4.x behaviour exactly.

## Benchmarks

```bash
uv run python scripts/prepare_dataset.py clone            # repos in configs/data/repos.yaml -> data/repos
uv run python scripts/prepare_dataset.py localization     # 240 post-release commits, 6 repos (py + ts)
uv run python scripts/prepare_dataset.py docstrings --languages python --out data/eval/docstrings-python
uv run python scripts/prepare_dataset.py docstrings --languages typescript --out data/eval/docstrings-typescript
uv run python scripts/prepare_dataset.py import-coir      # CodeSearchNet (py, js) NL->code, CoSQA
uv run python scripts/benchmark_models.py --config configs/evaluation/baseline.yaml --out artifacts/benchmarks/baseline
```

Task design, metrics, and the contamination policy are in [docs/evaluation.md](docs/evaluation.md).

## Training

```bash
uv run python scripts/prepare_dataset.py pairs --eval-tasks data/eval/*      # train-role repos only
uv run codeintel validate training data/train/train.jsonl
uv run codeintel train configs/training/contrastive.yaml
```

The recipe in `configs/training/contrastive.yaml` uses:

- source-homogeneous batches that skip in-batch duplicates;
- symmetric InfoNCE that masks duplicate positives;
- mined hard negatives with TopK-PercPos false-negative filtering;
- optional Matryoshka dims;
- GradCache, for large in-batch-negative counts on MPS or small GPUs;
- KL distillation toward the frozen base model, to preserve general usefulness;
- WiSE-FT weight interpolation of the final checkpoint.

Each checkpoint is written with a `codeintel_model.yaml` that reloads it through the generic `hf` adapter.

Training data is restricted to permissively licensed code (MIT, Apache-2.0, BSD, ISC, and similar; see
`codeintel.data.licenses`). Every repository in `configs/data/repos.yaml` declares an SPDX licence, which
`clone` records in `repos.lock.json`. `pairs` checks the declaration against the repository's licence files
at each revision it mines. It skips a repository, or an individual commit, that has no licence file, an
unrecognised or copyleft licence, or a mismatch, and it tags every pair with its licence.
`validate training` rejects any pair whose licence is missing or not permissive. The public evaluation
sets (CodeSearchNet, CoSQA) are never used for training.

## Export

```bash
uv run python scripts/export_onnx.py configs/models/gte-modernbert.yaml --out artifacts/onnx/gte-modernbert
```

This writes `model.onnx` (fp32, pooling and normalisation in-graph) and `model_int8.onnx`. The INT8 file
is a dynamic per-channel quantisation that keeps the FFN down-projections in fp32: quantising those
layers costs 0.04–0.08 cosine similarity against fp32. Both files are verified against PyTorch, and the
results go to `export_report.json`. Load an export with `adapter: onnx`.

## Layout

```
configs/      model, evaluation, training, and data-source configs
scripts/      prepare_dataset.py, benchmark_models.py, export_onnx.py
src/codeintel/
  models/     adapters (base, coderank, modernbert), ONNX export + runtime
  indexing/   repository walking, parsing (ast / tree-sitter), chunking, index storage + embedding cache
  retrieval/  BM25, exact dense search, hybrid fusion and file/symbol aggregation
  evaluation/ metrics + paired significance tests, contamination checks, benchmark runner
  data/       record schemas, mining (docstrings, tests, call sites, commits), licence policy, public importers,
              splits, validation
  training/   batching, losses, trainer
tests/        offline unit/integration tests; fixtures/sample_repo is a small Python + TypeScript repo
```

## License

The code is licensed under [Apache-2.0](LICENSE), the same licence as the gte-modernbert base checkpoint.
Apache-2.0 includes an explicit patent grant. For commercial use of a trained model, the licences of the
training data matter more than the code licence, which is why training only uses permissively licensed
repositories (see [Training](#training)).
