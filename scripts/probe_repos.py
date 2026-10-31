"""Count usable issue->PR links for candidate repositories (GitHub API via gh, cached under data/github).

  uv run python scripts/probe_repos.py owner/name [owner/name ...] [--since 2025-03-01]

Used to size candidate evaluation repositories by issue volume alone, before any model has seen them.
"""

from __future__ import annotations

import argparse
from collections import Counter

from codeintel.data.github import load_or_fetch
from codeintel.data.mining import clean_issue_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--since", default="2025-03-01")
    ap.add_argument("--cache", default="data/github")
    args = ap.parse_args()
    for slug in args.repos:
        try:
            links = load_or_fetch(slug, args.since, args.cache)
        except RuntimeError as exc:
            print(f"{slug}\tERROR\t{str(exc)[:80]}", flush=True)
            continue
        per_issue = Counter(link.issue_number for link in links)
        usable = [
            link for link in links
            if per_issue[link.issue_number] == 1 and link.issue_created_at[:10] >= args.since
            and link.issue_created_at <= link.pr_merged_at and link.original_body is not None
            and len(clean_issue_text(link.original_title, link.original_body)) >= 40
        ]
        print(f"{slug}\tlinks={len(links)}\tusable_before_diff={len(usable)}", flush=True)


if __name__ == "__main__":
    main()
