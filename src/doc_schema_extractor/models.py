"""Pydantic models for templates, rules, fingerprints, and results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Fingerprint(BaseModel):
    required_keywords: list[str] = Field(default_factory=list)
    supplier_hint: str = ""
    doc_type: str = "unknown"
    # Quorum: minimum fraction of required_keywords that must match.
    # 1.0 = all must match (old behaviour), 0.6 = 3 of 5 suffice.
    keyword_quorum: float = Field(default=0.6, ge=0.0, le=1.0)


class ExtractionRule(BaseModel):
    field: str
    type: str = "string"
    regex: str | None = None
    anchor_regex: str | None = None
    stop_regex: str | None = None
    columns: list[str] | None = None
    strip_chars: str | None = None
    date_format: str | None = None


class ConfidenceCheck(BaseModel):
    field: str
    not_null: bool = False
    gt: float | None = None
    lt: float | None = None
    regex_match: str | None = None


class Template(BaseModel):
    template_id: str
    fingerprint: Fingerprint
    extraction_rules: list[ExtractionRule] = Field(default_factory=list)
    confidence_checks: list[ConfidenceCheck] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    hit_count: int = 0

    def increment_hit(self) -> None:
        self.hit_count += 1
        self.updated_at = datetime.utcnow()


class ExtractionResult(BaseModel):
    document_path: str = ""
    raw_text: str = ""
    template_id: str | None = None
    match_score: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)
    llm_used: bool = False
    llm_backend: str | None = None
    llm_model: str | None = None
    validation_passed: bool = False
    validation_errors: list[str] = Field(default_factory=list)
