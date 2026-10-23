"""Train/eval overlap detection and training-set decontamination.

Three levels of overlap are checked:
  * exact:      identical text after normalisation (comments and whitespace removed, lower-cased);
  * near-dup:   MinHash-LSH candidates over token 5-gram shingles, confirmed by exact Jaccard;
  * repository: an evaluation repository also appearing in the training data.

Public benchmarks may additionally overlap with the *candidates'* own (pre-)training corpora, which cannot
be checked locally; `KNOWN_EXPOSURE` records what is known so reports can flag affected tasks.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

_LINE_COMMENT = re.compile(r"(?m)(^|\s)(#|//).*$")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")

# Benchmarks whose source data predates the candidates and is widely redistributed. Neither candidate's
# full training mixture is published in a form we can deduplicate against, so these are flagged rather
# than measured.
KNOWN_EXPOSURE: dict[str, str] = {
    "codesearchnet": (
        "CodeSearchNet (2019 GitHub snapshot) and its derivatives (CoIR CSN, CosQA, AdvTest) are public; "
        "CodeRankEmbed reports CSN results and is trained on CoRNStack (built from The Stack v2), and "
        "gte-modernbert's retrieval mixture is not fully published. Treat as possibly contaminated for both."
    ),
    "swe-bench": (
        "SWE-bench issues and repositories predate both candidates' releases and are public; repository code "
        "is almost certainly in pre-training corpora. Treat absolute numbers with caution."
    ),
    "temporal-commits": (
        "Commits authored after both candidates were published on the Hub (CodeRankEmbed: 2024-11-03, "
        "gte-modernbert-base: 2025-01-20). Commit messages are unseen; pre-change code may have been seen."
    ),
}


def normalize_code(text: str) -> str:
    text = _BLOCK_COMMENT.sub(" ", text)
    text = _LINE_COMMENT.sub(r"\1", text)
    return " ".join(text.lower().split())


def exact_key(text: str) -> str:
    return hashlib.sha256(normalize_code(text).encode()).hexdigest()


def shingles(text: str, n: int = 5) -> set[int]:
    tokens = _TOKEN.findall(normalize_code(text))
    if len(tokens) < n:
        tokens = tokens + [""] * (n - len(tokens))
    return {
        int.from_bytes(hashlib.blake2b("\x1f".join(tokens[i : i + n]).encode(), digest_size=8).digest(), "little")
        for i in range(len(tokens) - n + 1)
    }


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class NearDuplicateIndex:
    """MinHash (XOR-masked min-hashes) + banded LSH. Default bands/rows target Jaccard >= ~0.5 candidates."""

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 1234, ngram: int = 5):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.bands, self.rows, self.ngram = bands, num_perm // bands, ngram
        rng = np.random.default_rng(seed)
        self.masks = rng.integers(0, 2**63 - 1, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]
        self.ids: list[str] = []
        self.shingle_sets: list[set[int]] = []

    def signature(self, sh: set[int]) -> np.ndarray:
        arr = np.fromiter(sh, dtype=np.uint64, count=len(sh)) if sh else np.zeros(1, dtype=np.uint64)
        return np.bitwise_xor(arr[:, None], self.masks[None, :]).min(axis=0)

    def add(self, key: str, text: str) -> None:
        sh = shingles(text, self.ngram)
        sig = self.signature(sh)
        slot = len(self.ids)
        self.ids.append(key)
        self.shingle_sets.append(sh)
        for b in range(self.bands):
            self.buckets[b][sig[b * self.rows : (b + 1) * self.rows].tobytes()].append(slot)

    def query(self, text: str, threshold: float = 0.7) -> list[tuple[str, float]]:
        sh = shingles(text, self.ngram)
        sig = self.signature(sh)
        candidates: set[int] = set()
        for b in range(self.bands):
            candidates.update(self.buckets[b].get(sig[b * self.rows : (b + 1) * self.rows].tobytes(), ()))
        out = [(self.ids[c], jaccard(sh, self.shingle_sets[c])) for c in candidates]
        return sorted([(k, s) for k, s in out if s >= threshold], key=lambda x: -x[1])


@dataclass
class OverlapReport:
    num_eval: int
    exact: int = 0
    near_duplicate: int = 0
    repos_overlapping: list[str] = field(default_factory=list)
    examples: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return (self.exact + self.near_duplicate) / self.num_eval if self.num_eval else 0.0

    def to_dict(self) -> dict:
        return {
            "num_eval": self.num_eval,
            "exact": self.exact,
            "near_duplicate": self.near_duplicate,
            "rate": self.rate,
            "repos_overlapping": self.repos_overlapping,
            "examples": self.examples[:20],
        }


def overlap_report(
    eval_texts: Sequence[tuple[str, str]],
    train_texts: Iterable[tuple[str, str]],
    threshold: float = 0.7,
    eval_repos: Iterable[str] = (),
    train_repos: Iterable[str] = (),
) -> OverlapReport:
    """How many evaluation texts (id, text) exactly or nearly duplicate a training text (id, text)."""
    index = NearDuplicateIndex()
    exact_map: dict[str, str] = {}
    for key, text in train_texts:
        exact_map.setdefault(exact_key(text), key)
        index.add(key, text)
    report = OverlapReport(num_eval=len(eval_texts))
    for key, text in eval_texts:
        if (hit := exact_map.get(exact_key(text))) is not None:
            report.exact += 1
            report.examples.append((key, hit, 1.0))
            continue
        matches = index.query(text, threshold)
        if matches:
            report.near_duplicate += 1
            report.examples.append((key, matches[0][0], matches[0][1]))
    report.repos_overlapping = sorted(set(eval_repos) & set(train_repos))
    return report


def decontaminate(
    records: Sequence,
    eval_texts: Iterable[str],
    fields: Sequence[str] = ("query", "positive"),
    threshold: float = 0.7,
    blocked_repos: Iterable[str] = (),
) -> tuple[list, dict[str, int]]:
    """Drop training records whose `fields` exactly/nearly match any evaluation text or come from a blocked repo."""
    index = NearDuplicateIndex()
    exact = set()
    for i, text in enumerate(eval_texts):
        exact.add(exact_key(text))
        index.add(str(i), text)
    blocked = set(blocked_repos)
    kept, stats = [], {"exact": 0, "near_duplicate": 0, "repo": 0, "kept": 0}
    for rec in records:
        if blocked and getattr(rec, "repo", "") in blocked:
            stats["repo"] += 1
            continue
        texts = [getattr(rec, f) for f in fields]
        if any(exact_key(t) in exact for t in texts):
            stats["exact"] += 1
            continue
        if any(index.query(t, threshold) for t in texts):
            stats["near_duplicate"] += 1
            continue
        kept.append(rec)
    stats["kept"] = len(kept)
    return kept, stats
