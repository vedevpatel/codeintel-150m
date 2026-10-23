"""Okapi BM25 over code with identifier-aware tokenization (camelCase / snake_case splitting)."""

from __future__ import annotations

import math
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


class BM25Index:
    def __init__(
        self,
        documents: Sequence[str],
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = code_tokens,
    ):
        self.k1, self.b, self.tokenizer = k1, b, tokenizer
        self.n_docs = len(documents)
        vocab: dict[str, int] = {}
        postings_docs: list[list[int]] = []
        postings_tfs: list[list[int]] = []
        lengths = np.zeros(self.n_docs, dtype=np.float32)
        for doc_id, text in enumerate(documents):
            counts = Counter(tokenizer(text))
            lengths[doc_id] = sum(counts.values())
            for term, tf in counts.items():
                tid = vocab.setdefault(term, len(vocab))
                if tid == len(postings_docs):
                    postings_docs.append([])
                    postings_tfs.append([])
                postings_docs[tid].append(doc_id)
                postings_tfs[tid].append(tf)
        self.vocab = vocab
        self.doc_ids = [np.asarray(d, dtype=np.int32) for d in postings_docs]
        self.tfs = [np.asarray(t, dtype=np.float32) for t in postings_tfs]
        avg = float(lengths.mean()) if self.n_docs else 0.0
        self.norm = (k1 * (1 - b + b * lengths / avg)).astype(np.float32) if avg > 0 else np.full(self.n_docs, k1, np.float32)
        self.idf = np.asarray(
            [math.log(1 + (self.n_docs - len(d) + 0.5) / (len(d) + 0.5)) for d in postings_docs], dtype=np.float32
        )

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term, qtf in Counter(self.tokenizer(query)).items():
            tid = self.vocab.get(term)
            if tid is None:
                continue
            docs, tf = self.doc_ids[tid], self.tfs[tid]
            scores[docs] += qtf * self.idf[tid] * tf * (self.k1 + 1) / (tf + self.norm[docs])
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.scores(query)
        k = min(k, self.n_docs)
        if k <= 0:
            return []
        top = np.argsort(-scores, kind="stable")[:k]  # lowest index first among ties
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
