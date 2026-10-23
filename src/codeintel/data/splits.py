"""Leakage-safe dataset splits.

Records are split by *group* (normally the repository), never individually, so near-identical code from
one repository cannot land on both sides. Assignment is a pure function of the group key and a salt, so it
is stable across runs and machines and new data never moves existing groups between splits.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")

DEFAULT_RATIOS: Mapping[str, float] = {"train": 0.9, "dev": 0.05, "test": 0.05}


def _unit_interval(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(group: str, ratios: Mapping[str, float] = DEFAULT_RATIOS, salt: str = "codeintel-v1") -> str:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    u = _unit_interval(group, salt) * total
    acc = 0.0
    for name, ratio in ratios.items():
        acc += ratio
        if u < acc:
            return name
    return list(ratios)[-1]


def split_by_group(
    records: Iterable[T],
    group_fn: Callable[[T], str],
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    salt: str = "codeintel-v1",
    pinned: Mapping[str, str] | None = None,
) -> dict[str, list[T]]:
    """Split records by group. `pinned` forces groups into a split (e.g. eval repos -> "test")."""
    pinned = pinned or {}
    out: dict[str, list[T]] = {name: [] for name in ratios}
    for rec in records:
        group = group_fn(rec)
        name = pinned.get(group) or assign_split(group, ratios, salt)
        out.setdefault(name, []).append(rec)
    return out


def temporal_split(
    records: Sequence[T], date_fn: Callable[[T], str], cutoff: str
) -> tuple[list[T], list[T]]:
    """(before, on_or_after) by ISO-8601 date string; records without a date go to `before`."""
    before, after = [], []
    for rec in records:
        date = date_fn(rec)
        (after if date and date >= cutoff else before).append(rec)
    return before, after


def group_leakage(splits: Mapping[str, Sequence[T]], group_fn: Callable[[T], str]) -> dict[str, list[str]]:
    """Groups appearing in more than one split (should be empty)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for name, recs in splits.items():
        for rec in recs:
            seen[group_fn(rec)].add(name)
    return {g: sorted(s) for g, s in seen.items() if len(s) > 1}
