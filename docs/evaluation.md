# Evaluation

CodeIntel is evaluated on 15 benchmarks: issue -> code localization, code-to-code retrieval inside repositories,
docstring search, CoSQA, and six CoIR tasks. Every model is scored on the same data with the same indexer.
Results are in [results/benchmarks.md](results/benchmarks.md).

## Benchmarks

| benchmark | queries | corpus | retrieval |
|---|---|---|---|
| Issue -> code localization | 3,804 GitHub issues from 35 repositories | every definition in the repository at the fix's base commit | hybrid (dense + BM25) |
| Test -> implementation (Python) | 1,320 test functions | 74k functions | dense |
| Call site -> definition (Python / TypeScript) | 1,320 / 780 call sites | 77k / 36k definitions | dense |
| Docstring -> function (Python / TypeScript) | 600 / 514 summaries | 5.9k / 3.2k functions, doc comments removed | dense |
| CoSQA | 500 web queries | 6,267 functions | dense |
| CoIR: StackOverflow QA, CodeSearchNet (Python, JavaScript), CodeSearchNet-CCR (Python, JavaScript), APPS | as published | as published | dense |

The in-repository benchmarks (issues, code-to-code, docstrings) use 35 held-out Python and TypeScript repositories.
No training data comes from them.

## Issue -> code localization

A query is a GitHub issue as its author first wrote it. A hit is a function or class that the fix edited.

- **Queries.** The title and body as opened. Later edits are undone using GitHub's edit history, and comments are
  never used. Template boilerplate, images and checkbox lines are removed. Each issue was opened on or after
  2025-03-01 and closed by exactly one merged pull request. Bot-authored issues are excluded.
- **Index.** The repository is read at the pull request's base commit and split into definition-level chunks.
- **Labels.** Gold symbols are the pre-existing definitions whose lines the fix edited. Gold files are the
  non-test, non-documentation source files it changed. Pull requests touching more than 10 source files are excluded.
  Code the fix added cannot be retrieved, so each case records a recall ceiling. The mean is 0.89.
- **Strata.** Each issue is tagged by how much of the fix's code it names. Only identifiers written as code that
  appear in at most 2% of the repository's chunks count.

  | stratum | meaning | Python | TypeScript |
  |---|---|---|---|
  | symptom-only (`descriptive`) | names none of the fixed code | 360 | 730 |
  | partial mention (`same-file`) | names other code in a fixed file, not the edited definition | 414 | 362 |
  | names fixed code (`lexical`) | names an edited definition or file | 1,176 | 762 |

  The symptom-only stratum is the headline number. It is where lexical search has nothing to match.
- **Metric.** Symbol-level nDCG@10, with Recall@5/10/20/50 also reported.

## Fusion

Hybrid retrieval combines dense and BM25 rankings by reciprocal-rank fusion (k = 60) with a dense weight. Each model
uses the weight that scored best on a separate validation split of 19 repositories. That split is also used for every
modelling decision, so the test data is scored only once per model.

## Statistics

Cases within a repository are correlated, so the repository is the unit of resampling:

- **Confidence intervals:** 95% cluster bootstrap over repositories.
- **Paired comparisons:** a repository-level sign-flip test.
- **Averages:** metrics are reported both per case and macro-averaged over repositories.

## Public benchmarks

CoIR tasks are used as published, pinned to fixed dataset revisions (`src/codeintel/data/public.py`). CoSQA's corpus
contains many identical copies of the same code. These are merged into one document so that ties do not cap the
achievable score.

## Contamination

- **No shared repositories.** Training data never comes from an evaluation or validation repository.
- **Exact and near-duplicates.** Every training text was checked against every evaluation and validation text, CoIR
  corpora included. Exact copies after normalisation, and near-duplicates (MinHash over token 5-grams, Jaccard >= 0.7),
  were removed.
- **CodeSearchNet.** Training uses CodeSearchNet's train split. CoIR's CodeSearchNet tasks use its test split, whose
  repositories are disjoint from the train split's.

## Metrics

Relevance is binary.

- **nDCG@k:** gain 2^rel − 1.
- **MRR:** over the full ranking.
- **Recall@k:** the share of gold items in the top k.

Tied scores are broken by corpus order.
