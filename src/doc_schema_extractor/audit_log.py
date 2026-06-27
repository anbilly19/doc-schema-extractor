"""Extraction run audit log - appends a JSONL record per document processed.

Each line is a self-contained JSON object with:
  - timestamp
  - document filename and type
  - template matched (or null on MISS)
  - match_score (0.0-1.0)
  - all per-template candidate scores
  - llm_used flag
  - validation result
  - field count extracted
  - duration_ms

The JSONL format means it can be tailed, grep'd, or loaded into DuckDB/pandas
for cross-document score analysis without any extra tooling.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_utils import get_logger

logger = get_logger("audit_log")

DEFAULT_AUDIT_LOG = Path("./logs/extraction_audit.jsonl")


class AuditLog:
    """Append-only JSONL audit log for extraction runs."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("AUDIT_LOG_PATH", str(DEFAULT_AUDIT_LOG)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("AuditLog initialized path=%s", self.path)

    def record(
        self,
        *,
        document_path: str,
        template_id: str | None,
        match_score: float,
        candidate_scores: dict[str, float],
        llm_used: bool,
        llm_backend: str | None,
        llm_model: str | None,
        validation_passed: bool,
        validation_errors: list[str],
        field_count: int,
        duration_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "file": Path(document_path).name,
            "file_type": Path(document_path).suffix.lstrip("."),
            "template_id": template_id,
            "match_score": round(match_score, 4),
            "candidate_scores": {k: round(v, 4) for k, v in candidate_scores.items()},
            "result": "HIT" if (not llm_used and template_id) else "MISS",
            "llm_used": llm_used,
            "llm_backend": llm_backend,
            "llm_model": llm_model,
            "validation_passed": validation_passed,
            "validation_errors": validation_errors,
            "field_count": field_count,
            "duration_ms": round(duration_ms, 1),
        }
        if extra:
            entry.update(extra)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "Audit record written file=%s template=%s score=%.4f result=%s llm=%s duration_ms=%.1f",
            entry["file"], template_id, match_score, entry["result"], llm_used, duration_ms,
        )

    def read_all(self) -> list[dict[str, Any]]:
        """Load all audit records as a list of dicts."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed audit line: %s", line[:200])
        return records
