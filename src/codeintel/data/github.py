"""Fetch issue -> merged-PR links from the GitHub GraphQL API (read-only, via the authenticated `gh` CLI).

A link is an issue in the same repository listed in a merged PR's `closingIssuesReferences` (the PR says
"fixes #123" or was linked in the UI). Raw responses are cached as JSONL so that dataset builds are
reproducible and do not re-query the API.

Issue text is recovered *as it was when the issue was opened*: later body edits (which often add file paths
or links to the fix) are undone using the edit history, and title renames using the first rename event.
Comments are never used.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_VERSION = 2
# Closing references fetched per PR. 3 instead of 10 cuts the GraphQL cost per page from 11 to 4 points; PRs that
# close more than 3 issues are almost always large multi-issue PRs, which the miners discard anyway.
CLOSING_ISSUES = 3

_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: MERGED, first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title createdAt mergedAt changedFiles
        author { login }
        commits { totalCount }
        mergeCommit { oid parents(first: 2) { totalCount } }
        closingIssuesReferences(first: CLOSING_ISSUES) {
          nodes {
            number title body createdAt lastEditedAt includesCreatedEdit
            author { login }
            repository { nameWithOwner }
            userContentEdits(last: 1) { nodes { editedAt diff } }
            timelineItems(itemTypes: [RENAMED_TITLE_EVENT], first: 1) {
              nodes { ... on RenamedTitleEvent { previousTitle } }
            }
          }
        }
      }
    }
  }
}
"""


@dataclass
class LinkedIssue:
    repo: str  # owner/name
    issue_number: int
    issue_title: str
    issue_body: str
    issue_created_at: str
    issue_author: str
    pr_number: int
    pr_title: str
    pr_created_at: str
    pr_merged_at: str
    pr_author: str
    pr_changed_files: int
    pr_commits: int
    merge_commit: str
    merge_parents: int
    # Text as originally opened; `original_body` is None if the body was edited and no creation snapshot exists.
    original_title: str = ""
    original_body: str | None = None
    body_source: str = ""  # unedited | edit-history | unrecoverable


def original_issue_text(issue: dict) -> tuple[str, str | None, str]:
    """(title, body, source) as the issue was opened."""
    renames = (issue.get("timelineItems") or {}).get("nodes") or []
    title = renames[0]["previousTitle"] if renames and renames[0].get("previousTitle") is not None else issue["title"]
    if issue.get("lastEditedAt") is None:
        return title or "", issue["body"] or "", "unedited"
    oldest = ((issue.get("userContentEdits") or {}).get("nodes") or [None])[0]
    if issue.get("includesCreatedEdit") and oldest and oldest.get("diff") is not None:
        # The creation entry is stamped with the issue's own creation time.
        if (oldest.get("editedAt") or "")[:19] == issue["createdAt"][:19]:
            return title or "", oldest["diff"], "edit-history"
    if _light():
        return title or "", issue["body"] or "", "current-edited"
    return title or "", None, "unrecoverable"


_QUERY = _QUERY.replace("CLOSING_ISSUES", str(CLOSING_ISSUES))
# Training-data variant (CODEINTEL_GH_LIGHT=1): no edit history or rename events, which halves the cost again
# (2 points per page). Edited issues then keep their current text, marked body_source="current-edited"; evaluation
# and development data are always fetched with the full query, so their issue text is as opened.
_QUERY_LIGHT = _QUERY.replace("""
            userContentEdits(last: 1) { nodes { editedAt diff } }
            timelineItems(itemTypes: [RENAMED_TITLE_EVENT], first: 1) {
              nodes { ... on RenamedTitleEvent { previousTitle } }
            }""", "")
assert _QUERY_LIGHT != _QUERY


def _light() -> bool:
    return os.environ.get("CODEINTEL_GH_LIGHT") == "1"


def _graphql(query: str, retries: int = 6, **variables: str | None) -> dict:
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if value is not None:
            args += ["-F" if key == "cursor" else "-f", f"{key}={value}"]
    # GitHub intermittently answers large paginated queries with 502/504 or secondary rate limits.
    for attempt in range(retries):
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            if not data.get("errors"):
                return data["data"]
            error = f"GraphQL errors: {data['errors']}"
        else:
            error = proc.stderr.strip() or proc.stdout.strip()
        if attempt + 1 < retries:
            time.sleep(min(60, 2 ** (attempt + 2)))
    raise RuntimeError(f"GitHub GraphQL request failed after {retries} attempts: {error}")


def fetch_linked_issues(repo: str, since: str, max_pages: int = 200) -> list[LinkedIssue]:
    """Every (issue, merged PR) link for PRs created on/after `since` (ISO date), newest first."""
    owner, name = repo.split("/", 1)
    links: list[LinkedIssue] = []
    cursor = None
    for _ in range(max_pages):
        page = _graphql(_QUERY_LIGHT if _light() else _QUERY, owner=owner, name=name, cursor=cursor)["repository"]["pullRequests"]
        oldest = None
        for pr in page["nodes"]:
            oldest = pr["createdAt"]
            if pr["createdAt"][:10] < since or not pr["mergeCommit"]:
                continue
            for issue in pr["closingIssuesReferences"]["nodes"]:
                if issue["repository"]["nameWithOwner"].lower() != repo.lower():
                    continue
                title, body, body_source = original_issue_text(issue)
                links.append(
                    LinkedIssue(
                        repo=repo,
                        issue_number=issue["number"],
                        issue_title=issue["title"] or "",
                        issue_body=issue["body"] or "",
                        issue_created_at=issue["createdAt"],
                        issue_author=(issue["author"] or {}).get("login", ""),
                        pr_number=pr["number"],
                        pr_title=pr["title"] or "",
                        pr_created_at=pr["createdAt"],
                        pr_merged_at=pr["mergedAt"],
                        pr_author=(pr["author"] or {}).get("login", ""),
                        pr_changed_files=pr["changedFiles"],
                        pr_commits=pr["commits"]["totalCount"],
                        merge_commit=pr["mergeCommit"]["oid"],
                        merge_parents=pr["mergeCommit"]["parents"]["totalCount"],
                        original_title=title,
                        original_body=body,
                        body_source=body_source,
                    )
                )
        if not page["pageInfo"]["hasNextPage"] or (oldest is not None and oldest[:10] < since):
            break
        cursor = page["pageInfo"]["endCursor"]
    return links


def load_or_fetch(repo: str, since: str, cache_dir: str | Path, refresh: bool = False) -> list[LinkedIssue]:
    path = Path(cache_dir) / f"{repo.replace('/', '__')}.jsonl"
    if path.exists() and not refresh:
        meta, *rows = path.read_text().splitlines()
        if json.loads(meta).get("since") == since and json.loads(meta).get("version") == CACHE_VERSION:
            return [LinkedIssue(**json.loads(r)) for r in rows]
    links = fetch_linked_issues(repo, since)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"repo": repo, "since": since, "version": CACHE_VERSION, "closing_issues": CLOSING_ISSUES,
                            "light": _light()}) + "\n")
        for link in links:
            f.write(json.dumps(asdict(link)) + "\n")
    return links
