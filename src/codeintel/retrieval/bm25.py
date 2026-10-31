"""Okapi BM25 over code with identifier-aware tokenization (camelCase / snake_case splitting)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have how i if in into is it its
    me my not of on or our should so such than that the their them then there these they this to was we were
    what when where which while who why will with would you your""".split()
)


def _stem(token: str) -> str:
    """Tiny suffix stripper so that 'parsing'/'parsed'/'parses' meet 'parse' in identifiers."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def code_tokens(text: str) -> list[str]:
    out: list[str] = []
    for word in _IDENT.findall(text):
        parts = [p.lower() for piece in word.split("_") if piece for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            out.append(word.lower().strip("_"))  # the full identifier, for exact-name matches
        out.extend(parts)
    return [_stem(t) for t in out if len(t) > 1 and t not in STOPWORDS]


TermArrays = tuple[np.ndarray, np.ndarray]  # (term ids, term frequencies) of one document


def term_arrays(counts: Counter, vocab: dict[str, int]) -> TermArrays:
    """One document's term counts as parallel arrays over `vocab` (grown in place)."""
    ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in counts), dtype=np.int32, count=len(counts))
    return ids, np.fromiter(counts.values(), dtype=np.float32, count=len(counts))


class BM25Index:
    """Okapi BM25 over a CSC-style postings layout: for term t, `_docs/_tfs[_starts[t]:_starts[t + 1]]`."""

    def __init__(
        self,
        documents: Sequence[str] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = code_tokens,
        *,
        vocab: dict[str, int] | None = None,
        arrays: Sequence[TermArrays] | None = None,
    ):
        """Index `documents`, or per-document `arrays` (see `term_arrays`) over a shared `vocab`.

        The array form skips all per-term Python work, which matters when the same chunks are re-indexed for
        many repository snapshots (see `codeintel.evaluation.runner.SnapshotLoader`).
        """
        self.k1, self.b, self.tokenizer = k1, b, tokenizer
        if arrays is None:
            vocab = {}
            arrays = [term_arrays(Counter(tokenizer(text)), vocab) for text in documents or []]
        assert vocab is not None
        self.vocab = vocab
        n = self.n_docs = len(arrays)
        n_terms = len(vocab)
        lengths = np.fromiter((len(ids) for ids, _ in arrays), dtype=np.int64, count=n)
        if n and lengths.sum():
            all_ids = np.concatenate([ids for ids, _ in arrays])
            all_tf = np.concatenate([tf for _, tf in arrays])
        else:
            all_ids, all_tf = np.zeros(0, np.int32), np.zeros(0, np.float32)
        doc_idx = np.repeat(np.arange(n, dtype=np.int32), lengths)
        order = np.argsort(all_ids, kind="stable")  # stable: postings stay in ascending document order
        self._docs = doc_idx[order]
        self._tfs = all_tf[order].astype(np.float32, copy=False)
        df = np.bincount(all_ids, minlength=n_terms)
        self._starts = np.concatenate([[0], np.cumsum(df)])
        doc_len = np.bincount(doc_idx, weights=all_tf, minlength=n).astype(np.float32)
        avg = float(doc_len.mean()) if n else 0.0
        self.norm = (k1 * (1 - b + b * doc_len / avg)).astype(np.float32) if avg > 0 else np.full(n, k1, np.float32)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term, qtf in Counter(self.tokenizer(query)).items():
            tid = self.vocab.get(term)
            if tid is None or tid >= len(self.idf):  # the shared vocabulary may have grown after this index
                continue
            lo, hi = self._starts[tid], self._starts[tid + 1]
            if lo == hi:
                continue
            docs, tf = self._docs[lo:hi], self._tfs[lo:hi]
            scores[docs] += qtf * self.idf[tid] * tf * (self.k1 + 1) / (tf + self.norm[docs])
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.scores(query)
        k = min(k, self.n_docs)
        if k <= 0:
            return []
        top = np.argsort(-scores, kind="stable")[:k]  # lowest index first among ties
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
