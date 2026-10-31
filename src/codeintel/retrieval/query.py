"""Clean issue text before it is used as a retrieval query.

Issue templates, environment dumps and long logs bury the symptom description, and the embedding models see
only the first `max_length` tokens. `clean_issue_query` keeps the title and the prose that describes the
problem, drops template questions and environment/version lines, shortens long code blocks to their first and
last lines (an exception message is usually the last line of a trace), and truncates to `max_chars`.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_DETAILS = re.compile(r"<details\b.*?</details>", re.S | re.I)
_HTML_TAG = re.compile(r"</?(?:p|b|i|u|br|hr|details|summary|pre|code|img|kbd|sub|sup|center|div|span)\b[^>]*>", re.I)
_EMPTY_ANSWER = re.compile(r"^\s*[-*]?\s*(none|n/?a|no|nothing|_?no response_?|-+|\.+)\s*\.?\s*$", re.I)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_BOLD_HEADING = re.compile(r"^\s*\*\*([^*\n]{3,90})\*\*\s*:?\s*$")
# Template section titles whose content is environment data, links, or output dumps rather than description.
_BOILERPLATE_HEADING = re.compile(
    r"\b(version|runtime|platform|environment|system info|operating system|browser|os\b|node|npm|yarn|pnpm|"
    r"pip list|pip freeze|installed packages|dependencies|checklist|screenshots?|logs?\b|stack ?trace|"
    r"debug(ging)? info\w*|reproduction (repo|repository|link|url)|minimal reproduc\w+ (repo|link|url)|"
    r"link to|codesandbox|stackblitz|playground|additional (context|information)|any other|code of conduct|"
    r"before submitting|have you|duplicate)\b",
    re.I,
)
_ENV_LINE = re.compile(
    r"^\s*[-*]?\s*(os|operating system|python|node(?:\.?js)?|npm|yarn|pnpm|pip|deno|bun|browser|version|platform|"
    r"runtime|package manager|arch(?:itecture)?|cpu|gpu|cuda|java|rust|go)\b[^\n]{0,70}\d[^\n]{0,30}$",
    re.I,
)
_PACKAGE_LINE = re.compile(r"^\s*[-*]?\s*[\w.@/\-]+\s*(==|=|:|\s)\s*v?\d+(\.\d+){1,3}[\w.\-+]*\s*,?\s*$")
_BARE_VERSION = re.compile(r"^\s*v?\d+(\.\d+){1,3}[\w.\-+()` ]{0,30}$")
_LINK_ONLY = re.compile(r"^\s*[-*]?\s*<?https?://\S+>?\s*$")


def _shorten_block(match: re.Match, max_lines: int = 10, head: int = 5, tail: int = 2) -> str:
    lines = match.group(1).rstrip("\n").split("\n")
    if len(lines) > max_lines:
        lines = lines[:head] + ["..."] + lines[-tail:]
    return "\n".join(lines) + "\n"


def _drop_boilerplate_sections(body: str) -> str:
    """Remove template headings; drop the content of sections whose heading marks environment/log boilerplate."""
    out: list[str] = []
    skipping = False
    for line in body.split("\n"):
        heading = _HEADING.match(line)
        bold = None if heading else _BOLD_HEADING.match(line)
        title = heading.group(2) if heading else (bold.group(1) if bold else None)
        if title is not None:
            skipping = bool(_BOILERPLATE_HEADING.search(title))
            continue  # template headings are questions ("What is the expected behavior?"), not description
        if not skipping:
            out.append(line)
    return "\n".join(out)


def clean_issue_query(text: str, max_chars: int = 1200, max_title_chars: int = 300) -> str:
    """Title plus the descriptive part of the body, at most about `max_chars` characters."""
    title, _, body = text.strip().partition("\n")
    title = title.strip()[:max_title_chars]
    body = _DETAILS.sub("", body)
    body = _drop_boilerplate_sections(_FENCE.sub(_shorten_block, body))
    body = _HTML_TAG.sub("", body).replace("```", "")
    kept: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped == "" and kept and kept[-1] == "":
            continue
        if stripped and (_EMPTY_ANSWER.match(line) or _ENV_LINE.match(line) or _PACKAGE_LINE.match(line) or _BARE_VERSION.match(line) or _LINK_ONLY.match(line)):
            continue
        if kept and stripped and stripped == kept[-1].strip():
            continue
        kept.append(line.rstrip())
    prose = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    cleaned = f"{title}\n\n{prose}".strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    boundary = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind(" "))
    return (cut[:boundary] if boundary > max_chars * 0.6 else cut).rstrip()
