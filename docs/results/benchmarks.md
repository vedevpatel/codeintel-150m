# Benchmarks

nDCG@10 on held-out test data for CodeIntel and seven open embedding models of similar size. Bold marks the best
score on each benchmark; — means not evaluated.

- **Issue -> code:** 3,804 GitHub issues from 35 Python and TypeScript repositories, none used in training. Scored
  at the symbol level, with BM25 fusion at each model's validation-chosen weight.
  - *Symptom-only* issues name no code from the fix.
  - *Issues that name code* mention an edited definition or file.
- **Test -> implementation, call site -> definition, docstring -> function:** built from the same held-out
  repositories. Dense retrieval.
- **CoSQA and the CoIR tasks:** the published test sets. Dense retrieval.

| benchmark | CodeIntel | granite-r2 | bge-base | modernbert-embed | gte-modernbert | CodeRankEmbed | jina-v2-code | codesage-small |
|---|---|---|---|---|---|---|---|---|
| Issue -> code, symptom-only issues (Python) | **0.268** | 0.173 | 0.155 | 0.153 | 0.146 | 0.119 | 0.143 | 0.093 |
| Issue -> code, symptom-only issues (TypeScript) | **0.274** | 0.178 | 0.176 | 0.164 | 0.153 | 0.132 | 0.174 | 0.131 |
| Issue -> code, issues that name code | **0.516** | 0.393 | 0.405 | 0.370 | 0.393 | 0.361 | 0.382 | 0.340 |
| Test -> implementation (Python) | **0.640** | 0.480 | 0.461 | 0.518 | 0.496 | 0.522 | 0.472 | 0.397 |
| Call site -> definition (Python) | 0.219 | 0.206 | **0.244** | 0.190 | 0.176 | 0.192 | 0.197 | 0.187 |
| Call site -> definition (TypeScript) | **0.278** | 0.264 | 0.246 | 0.228 | 0.213 | 0.247 | 0.226 | 0.234 |
| Docstring -> function (Python) | 0.717 | 0.719 | 0.744 | 0.748 | 0.770 | **0.863** | 0.787 | 0.813 |
| Docstring -> function (TypeScript) | 0.672 | 0.671 | 0.623 | 0.703 | 0.720 | **0.762** | 0.716 | 0.723 |
| CoSQA | 0.651 | 0.652 | 0.598 | 0.610 | 0.733 | 0.653 | **0.737** | 0.575 |
| CoIR StackOverflow QA | **0.922** | 0.917 | 0.803 | — | 0.912 | 0.831 | 0.894 | 0.771 |
| CoIR CodeSearchNet (Python) | **0.963** | 0.946 | 0.892 | — | 0.882 | 0.948 | 0.947 | 0.960 |
| CoIR CodeSearchNet (JavaScript) | 0.634 | 0.532 | 0.553 | — | 0.761 | 0.738 | **0.764** | 0.755 |
| CoIR CodeSearchNet-CCR (Python) | 0.703 | 0.514 | 0.598 | — | **0.940** | 0.764 | 0.686 | 0.751 |
| CoIR CodeSearchNet-CCR (JavaScript) | 0.688 | 0.522 | 0.542 | — | **0.943** | 0.703 | 0.625 | 0.696 |
| CoIR APPS | 0.174 | 0.140 | 0.064 | — | **0.575** | 0.235 | 0.164 | 0.319 |

Methodology: [../evaluation.md](../evaluation.md). Machine-readable scores: [benchmarks.json](benchmarks.json).
