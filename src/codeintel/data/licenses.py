"""Licence policy for mined repositories.

Every manifest entry declares its licence as an SPDX expression. Training pairs may only come from repositories
whose declared licence is permissive, and the declaration is checked against the licence files in the tree at
every revision code is taken from (a repository can be relicensed, or gain its licence file, partway through its
history). The declared licence is recorded on each pair and in repos.lock.json.

Detection is deliberately narrow: it recognises the standard texts, and anything else -- an unrecognised file,
extra third-party notices, a copyleft licence -- fails the check and needs a manual decision.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PERMISSIVE = frozenset(
    {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "Zlib", "Unlicense", "CC0-1.0"}
)

LICENSE_FILE = re.compile(r"^(licen[cs]e|copying)([-._][^/]*)?$", re.I)


def _phrase(text: str) -> re.Pattern:
    """Case-insensitive pattern for a licence phrase that tolerates any line wrapping between words."""
    return re.compile(r"\s+".join(map(re.escape, text.split())), re.I)


_BSD = _phrase("Redistribution and use in source and binary forms")
_BSD_ENDORSE = re.compile(r"Neither\s+the\s+name|names\s+of\s+(its|the)\s+contributors\s+may\s+be\s+used", re.I)
_ISC = re.compile(r"Permission\s+to\s+use,\s+copy,\s+modify,\s+and(/or)?\s+distribute\s+this\s+software\s+for\s+any\s+purpose", re.I)
_ISC_NOTICE = _phrase("provided that the above copyright notice")
_SIGNATURES = [
    ("Apache-2.0", re.compile(r"Apache\s+License,?\s+Version\s+2\.0", re.I)),
    ("MIT", _phrase("Permission is hereby granted, free of charge, to any person obtaining a copy")),
    ("Unlicense", _phrase("This is free and unencumbered software released into the public domain")),
    ("CC0-1.0", _phrase("CC0 1.0 Universal")),
    ("Zlib", _phrase("This software is provided 'as-is', without any express or implied warranty")),
    ("GPL-family", re.compile(r"GNU\s+(Lesser\s+|Library\s+|Affero\s+)?General\s+Public\s+License", re.I)),
    ("MPL-2.0", re.compile(r"Mozilla\s+Public\s+License,?\s+(Version|v\.?)\s*2\.0", re.I)),
    ("EPL", _phrase("Eclipse Public License")),
    ("SSPL", _phrase("Server Side Public License")),
    ("BUSL", _phrase("Business Source License")),
]


class LicenseError(ValueError):
    pass


def spdx_ids(expression: str) -> set[str]:
    return {t.strip() for t in re.split(r"\s+(?:AND|OR|WITH)\s+|[()]", expression) if t.strip()}


def is_permissive(expression: str) -> bool:
    """`A OR B` is permissive if either option is (the licensee chooses); `A AND B` only if both are."""
    expression = expression.strip()
    if not expression or "(" in expression or " WITH " in expression:
        return False  # nested expressions and exceptions are left to manual review
    return any(
        all(term.strip() in PERMISSIVE for term in option.split(" AND "))
        for option in expression.split(" OR ")
    )


def detect_licenses(text: str) -> set[str]:
    """Licences whose standard text appears in `text` (several, when notices are concatenated)."""
    found = {spdx for spdx, pattern in _SIGNATURES if pattern.search(text)}
    if _BSD.search(text):
        found.add("BSD-3-Clause" if _BSD_ENDORSE.search(text) else "BSD-2-Clause")
    if _ISC.search(text):
        found.add("ISC" if _ISC_NOTICE.search(text) else "0BSD")
    return found


def _git(repo: str | Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout.decode(
        "utf-8", errors="replace"
    )


class LicenseChecker:
    """Checks a repository's root licence files against its declared licence, per revision.

    Results are cached by licence blob, so checking every mined commit costs one `git ls-tree` per commit.
    """

    def __init__(self, repo_dir: str | Path, declared: str):
        self.repo_dir = repo_dir
        self.declared = declared
        self._by_blobs: dict[tuple[tuple[str, str], ...], str | None] = {}

    def _license_blobs(self, rev: str) -> tuple[tuple[str, str], ...]:
        blobs = []
        for line in _git(self.repo_dir, "ls-tree", rev).splitlines():
            meta, _, path = line.partition("\t")
            _, kind, sha = meta.split()
            if kind == "blob" and LICENSE_FILE.match(path):
                blobs.append((path, sha))
        return tuple(sorted(blobs))

    def problem(self, rev: str) -> str | None:
        """None if `rev` may be mined for training, otherwise the reason it may not."""
        if not is_permissive(self.declared):
            return f"declared licence {self.declared!r} is not permissive"
        blobs = self._license_blobs(rev)
        if blobs not in self._by_blobs:
            self._by_blobs[blobs] = self._check(blobs)
        return self._by_blobs[blobs]

    def _check(self, blobs: tuple[tuple[str, str], ...]) -> str | None:
        if not blobs:
            return "no licence file at the repository root"
        declared = spdx_ids(self.declared)
        for path, sha in blobs:
            found = detect_licenses(_git(self.repo_dir, "cat-file", "blob", sha))
            if not found:
                return f"{path}: unrecognised licence text"
            if not found <= declared:
                return f"{path}: found {sorted(found)}, declared {self.declared!r}"
        return None

    def require(self, rev: str) -> None:
        if (reason := self.problem(rev)) is not None:
            raise LicenseError(f"{self.repo_dir}@{rev[:12]}: {reason}")
