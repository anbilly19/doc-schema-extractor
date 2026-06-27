"""Pydantic models for templates and extraction results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Fingerprint(BaseModel):
    required_keywords: list[str]
    supplier_hint: str = ""
    doc_type: str = "unknown"


class ExtractionRule(BaseModel):
    field: str
    type: Literal["string", "date", "decimal", "integer", "table", "list"]
    regex: str | None = None
    # For table rules
    anchor_regex: str | None = None
    stop_regex: str | None = None
    columns: list[str] | None = None
    # Post-processing
    strip_chars: str | None = None
    date_format: str | None = None  # e.g. "%d.%m.%Y"


class ConfidenceCheck(BaseModel):
    field: str
    not_null: bool = False
    gt: float | None = None
    lt: float | None = None
    regex_match: str | None = None


class Template(BaseModel):
    template_id: str
    fingerprint: Fingerprint
    extraction_rules: list[ExtractionRule]
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
    document_path: str
    template_id: str | None = None
    match_score: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    llm_used: bool = False
    llm_backend: str | None = None
    llm_model: str | None = None
    validation_passed: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
