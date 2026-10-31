"""Move per-repository evaluation checkpoints through a text-only channel, such as a remote job's stdout.

For jobs whose only output is their log, each repository's checkpoint is dictionary-encoded (ranked keys
become integers into a per-checkpoint key table), gzipped, base64-encoded, and printed between markers.
`parse_stream` recovers the checkpoints from a saved log, tolerating interleaved log lines and truncated
output (a crashed job still yields every finished repository).
"""

from __future__ import annotations

import base64
import gzip
import json
import re
from typing import Any

BEGIN = "@@CHECKPOINT"
END = "@@END"
_RANKS = ("file_rank", "sym_rank")


def pack(repo: str, checkpoint: dict[str, Any]) -> str:
    table: dict[str, int] = {}

    def code(keys: list[str]) -> list[int]:
        return [table.setdefault(k, len(table)) for k in keys]

    body = dict(checkpoint)
    for field in _RANKS:
        body[field] = {system: {q: code(keys) for q, keys in per_q.items()} for system, per_q in checkpoint[field].items()}
    body["key_table"] = list(table)
    blob = base64.b64encode(gzip.compress(json.dumps(body, separators=(",", ":")).encode(), compresslevel=9)).decode()
    # The CRC is over the encoded text, so a truncated or corrupted block is detected before decompression.
    return f"{BEGIN} {repo} {len(blob)} {_crc(blob)}\n{blob}\n{END}\n"


def unpack(blob: str) -> dict[str, Any]:
    body = json.loads(gzip.decompress(base64.b64decode(blob)))
    table = body.pop("key_table")
    for field in _RANKS:
        body[field] = {s: {q: [table[i] for i in ids] for q, ids in per_q.items()} for s, per_q in body[field].items()}
    return body


def _crc(text: str) -> str:
    import zlib

    return f"{zlib.crc32(text.encode()):08x}"


_BLOCK = re.compile(rf"^{re.escape(BEGIN)} (\S+) (\d+) ([0-9a-f]{{8}})\n(.*?)\n{re.escape(END)}$", re.M | re.S)


def parse_stream(text: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """(repo -> checkpoint, problems). Blocks that fail their length or CRC check are reported and skipped."""
    found: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for m in _BLOCK.finditer(text):
        repo, length, crc, blob = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if len(blob) != length or _crc(blob) != crc:
            problems.append(f"{repo}: block corrupted or truncated")
            continue
        found[repo] = unpack(blob)  # a repeated repo keeps the latest copy
    opened = len(re.findall(rf"^{re.escape(BEGIN)} ", text, re.M))
    if opened > len(_BLOCK.findall(text)) + len(problems):
        problems.append("stream ended inside a checkpoint block (job cut off)")
    return found, problems
