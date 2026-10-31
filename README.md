# CodeIntel

CodeIntel is a 149M-parameter embedding model for **issue-to-code localization**: given a bug report or feature
request, it retrieves the functions and classes in a Python or TypeScript repository that a fix is likely to
modify. On issues that describe only symptoms, with no reference to the relevant code, it outperforms every other
open embedding model of comparable size by **roughly 55%**. It also leads on issues that reference code directly
(+27%) and on test-to-implementation retrieval (+23%). Beyond localization, it supports code search and
call-site-to-definition navigation.

It runs locally, including as a 325 MB INT8 ONNX model on CPU, and is released under Apache-2.0. It was trained only on
permissively licensed code.

This repository contains the model's training and evaluation code: a repository indexer, BM25 / dense / hybrid
retrieval, the training pipeline, and the benchmarks.

## Results

Held-out test data, nDCG@10. Issue -> code is scored at the symbol level, over 3,804 GitHub issues from 35
repositories that were never used in training. The comparison models are open embedding models of similar size
(granite-embedding-english-r2, bge-base-en-v1.5, modernbert-embed-base, gte-modernbert-base, CodeRankEmbed,
jina-embeddings-v2-base-code, codesage-small-v2).

| benchmark | CodeIntel | best other open ~150M model |
|---|---|---|
| issue -> code, symptom-only issues (Python / TypeScript) | **0.268 / 0.274** | 0.173 / 0.178 |
| issue -> code, issues that name code | **0.516** | 0.405 |
| test -> implementation | **0.640** | 0.522 |
| call site -> definition (TypeScript) | **0.278** | 0.264 |
| CoIR StackOverflow QA | **0.922** | 0.917 |
| CoIR CodeSearchNet (Python) | **0.963** | 0.960 |

CodeIntel is built for repository-scale retrieval driven by issues, bug reports and tests. For general
natural-language code search (docstring-style queries, web questions), a general-purpose code embedding model may
suit better. All 15 benchmarks, with per-model scores, are in [docs/results/benchmarks.md](docs/results/benchmarks.md). The methodology is
in [docs/evaluation.md](docs/evaluation.md).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("vedevpatel/CodeIntel-150M")
code = [
    "# src/server/app.py::App.run\ndef run(self, host=None, port=None, debug=None):\n    ...",
    "# src/utils/date.ts::parseDate\nexport function parseDate(s: string): Date { return new Date(s); }",
]
scores = model.encode(["Server ignores the port argument when debug is enabled"]) @ model.encode(code).T
```

Documents are code definitions prefixed with a one-line `path::QualifiedName` header in the language's comment
syntax. Queries are raw issue text, and no prompt is needed. Fusing dense scores with BM25 (reciprocal-rank fusion,
dense weight 2) adds about 0.03 nDCG@10 on issue localization.

## Setup

```bash
uv sync --all-extras    # Python 3.10-3.13
uv run pytest           # offline tests
```

Runs on CPU, CUDA and Apple Silicon. `device: auto` picks CUDA, then MPS, then CPU.

## Index and search a repository

```bash
uv run codeintel index path/to/repo --out artifacts/index/repo --model configs/models/<model>.yaml
uv run codeintel search artifacts/index/repo "retry failed requests with backoff" \
    --model configs/models/<model>.yaml --level symbol --mode hybrid
```

Chunks follow definitions (functions, methods, class skeletons, module code), and oversized spans are split into
line windows. Embeddings are cached by content hash, so re-indexing a repository only encodes the chunks that
changed.

## Evaluation

```bash
uv run python scripts/prepare_dataset.py clone          # repositories in configs/data/repos.yaml
uv run python scripts/prepare_dataset.py issues         # issue -> code localization benchmark
uv run python scripts/prepare_dataset.py code2code --kind tests --languages python --out data/eval/code2code-tests-python
uv run python scripts/prepare_dataset.py code2code --kind calls --languages python --out data/eval/code2code-calls-python
uv run python scripts/prepare_dataset.py code2code --kind calls --languages typescript --out data/eval/code2code-calls-typescript
uv run python scripts/prepare_dataset.py docstrings --languages python --out data/eval/docstrings-python
uv run python scripts/prepare_dataset.py docstrings --languages typescript --out data/eval/docstrings-typescript
uv run python scripts/prepare_dataset.py import-coir    # CoSQA and the CoIR tasks
uv run codeintel evaluate configs/evaluation/issue-localization.yaml
uv run codeintel evaluate configs/evaluation/retrieval.yaml
```

Confidence intervals resample whole repositories. Every training text is checked against every evaluation text
for exact and near-duplicate matches.

## Training

```bash
uv run python scripts/mine_train_parallel.py --workers 6       # mine training pairs from train-role repositories
uv run python scripts/pack_training.py --config configs/training/codeintel.yaml --out artifacts/train/groups.json.gz
uv run codeintel train configs/training/codeintel.yaml
```

The recipe is multi-positive InfoNCE with mined hard negatives and GradCache, plus distillation toward the frozen
base model so general search ability is kept. The base model is granite-embedding-english-r2.

Training data is restricted to permissively licensed code (MIT, Apache-2.0, BSD, ISC and similar):

- **Mined pairs:** issue -> edited function, commit -> edited function, docstring -> function,
  test -> implementation and call site -> definition, from 82 GitHub repositories. Each repository's licence is
  verified at every revision code is taken from.
- **Public datasets:** licence-filtered pairs from The Vault and from CodeSearchNet's train split.

Every repository and dataset used, with its licence, is listed in the model's `TRAINING_DATA.md`.

## Export

```bash
uv run codeintel export configs/models/<model>.yaml --out artifacts/onnx/<model>
```

This writes `model.onnx` (fp32) and `model_int8.onnx`, a dynamic INT8 quantisation that keeps the feed-forward
down-projections in fp32. Both are checked against PyTorch.

## Layout

```
configs/        model, training, evaluation and data-source configs
scripts/        data preparation and mining, GPU job runners, release builder
src/codeintel/  models, indexing, retrieval, evaluation, data and training
tests/          offline tests, run against a small Python + TypeScript fixture repository
docs/           evaluation design and benchmark results
```

## License

Apache-2.0. See [LICENSE](LICENSE).
