"""Main Extractor orchestrator with LangSmith tracing and audit logging."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from langsmith import traceable

from .audit_log import AuditLog
from .backends.base import LLMBackend
from .backends.ollama import OllamaBackend
from .backends.openai_backend import OpenAIBackend
from .logging_utils import get_logger
from .models import ConfidenceCheck, ExtractionResult, ExtractionRule, Fingerprint, Template
from .rule_engine import RuleEngine
from .template_sanitiser import sanitise_template
from .template_store import TemplateStore
from .text_extractor import TextExtractor
from .tracing import trace_llm_call, trace_rule_engine, trace_template_match, trace_validator
from .validator import Validator

logger = get_logger("extractor")

_VALIDATION_FALLBACK_THRESHOLD = int(os.getenv("VALIDATION_FALLBACK_THRESHOLD", "2"))

# Minimum keyword hit-rate on a MISS to attempt family-based near-miss reuse.
# If the best candidate scores above this but below threshold, we treat it as
# a near-miss and pass the existing template to the LLM for updating.
_NEAR_MISS_MIN_SCORE = float(os.getenv("NEAR_MISS_MIN_SCORE", "0.3"))


def _build_default_backend() -> LLMBackend:
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "openai":
        model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        logger.info("Using default OpenAI backend model=%s", model)
        return OpenAIBackend(model=model)
    model = os.getenv("OLLAMA_DEFAULT_MODEL", "gemma4:e4b-it-qat")
    logger.info("Using default Ollama backend model=%s", model)
    return OllamaBackend(model=model)


def _resolve_store_path(raw: str | Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        p = project_root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Extractor:
    def __init__(
        self,
        backend: LLMBackend | None = None,
        store_path: str | Path | None = None,
        match_threshold: float | None = None,
        audit_log: AuditLog | None = None,
    ):
        raw_path = store_path or os.getenv("TEMPLATE_STORE_PATH", "templates/store.json")
        resolved = _resolve_store_path(raw_path)
        self._backend = backend or _build_default_backend()
        self._store = TemplateStore(resolved)
        self._text_extractor = TextExtractor()
        self._rule_engine = RuleEngine()
        self._validator = Validator()
        self._threshold = match_threshold if match_threshold is not None else float(
            os.getenv("TEMPLATE_MATCH_THRESHOLD", "0.75")
        )
        self._raw_preview_chars = int(os.getenv("LOG_RAW_TEXT_PREVIEW_CHARS", "2000"))
        self._audit = audit_log or AuditLog()
        logger.info(
            "Extractor initialized backend=%s model=%s threshold=%.2f store=%s",
            self._backend.name, self._backend.model, self._threshold, resolved,
        )

    @traceable(name="extraction_run")
    def extract(self, file_path: str | Path) -> ExtractionResult:
        logger.info("Extraction run started file=%s", file_path)
        t_start = time.monotonic()
        path = Path(file_path)
        doc = self._text_extractor.extract(path)

        logger.debug(
            "Document summary file=%s type=%s chars=%s pages=%s preview=%s",
            path, doc.file_type, len(doc.full_text), len(doc.pages),
            self._preview(doc.full_text),
        )

        result = ExtractionResult(document_path=str(path), raw_text=doc.full_text)

        template, score, candidate_scores = self._store.match_with_scores(
            doc.normalised_text, self._threshold
        )
        result.match_score = score
        trace_template_match(
            raw_text_preview=doc.full_text,
            threshold=self._threshold,
            template_count=len(self._store.list_all()),
            matched_template_id=template.template_id if template else None,
            match_score=score,
            candidate_scores=candidate_scores,
        )

        if template:
            logger.info("Template HIT template_id=%s score=%.3f", template.template_id, score)
            result.template_id = template.template_id

            data = self._rule_engine.apply(template, doc)
            trace_rule_engine(
                template_id=template.template_id,
                field_count=len(template.extraction_rules),
                extracted_fields=data,
            )

            passed, errors = self._validator.validate(data, template)
            trace_validator(
                template_id=template.template_id,
                check_count=len(template.confidence_checks),
                passed=passed,
                errors=errors,
            )

            if passed:
                result.data = data
                result.validation_passed = True
                result.llm_used = False
                template.increment_hit()
                self._store.add(template)
                self._write_audit(result, candidate_scores, t_start)
                logger.info("Extraction complete via template template_id=%s", template.template_id)
                return result

            critical_errors = [e for e in errors if "null or empty" in e]
            logger.warning(
                "Validation failed template_id=%s critical=%s/%s errors=%s",
                template.template_id, len(critical_errors), len(errors), errors,
            )

            if len(critical_errors) < _VALIDATION_FALLBACK_THRESHOLD:
                logger.info(
                    "Tolerating %s critical error(s) below threshold=%s, returning template result",
                    len(critical_errors), _VALIDATION_FALLBACK_THRESHOLD,
                )
                result.data = data
                result.validation_passed = False
                result.validation_errors = errors
                result.llm_used = False
                result.template_id = template.template_id
                self._write_audit(result, candidate_scores, t_start)
                return result

            logger.info(
                "Critical errors=%s >= threshold=%s; falling back to LLM",
                len(critical_errors), _VALIDATION_FALLBACK_THRESHOLD,
            )
            result.validation_errors = errors
            result = self._llm_extract(
                result, doc.full_text, existing_template=template, source_text=doc.full_text
            )
            self._write_audit(result, candidate_scores, t_start)
            return result

        # --- MISS path ---
        # Before calling the LLM cold, check for a near-miss: a template that
        # scored above _NEAR_MISS_MIN_SCORE but below threshold.  If found,
        # also check family similarity (doc_type + supplier) from the raw LLM
        # response is unavailable yet, so we use the best scoring candidate
        # directly.  The LLM will be instructed to reuse its template_id.
        near_miss_template: Template | None = None
        if score >= _NEAR_MISS_MIN_SCORE and candidate_scores:
            best_id = max(candidate_scores, key=lambda k: candidate_scores[k])
            best_candidate = self._store.get(best_id)
            if best_candidate:
                logger.info(
                    "Near-miss candidate id=%s score=%.3f; will pass to LLM for update",
                    best_id, score,
                )
                near_miss_template = best_candidate

        logger.info(
            "Template MISS score=%.3f near_miss=%s; falling back to LLM",
            score, near_miss_template.template_id if near_miss_template else None,
        )
        result = self._llm_extract(
            result, doc.full_text,
            existing_template=near_miss_template,
            source_text=doc.full_text,
        )
        self._write_audit(result, candidate_scores, t_start)
        return result

    def _llm_extract(
        self,
        result: ExtractionResult,
        raw_text: str,
        existing_template: Template | None = None,
        source_text: str = "",
    ) -> ExtractionResult:
        logger.info(
            "LLM extraction start backend=%s model=%s existing_template=%s",
            self._backend.name, self._backend.model,
            existing_template.template_id if existing_template else None,
        )
        try:
            llm_response = self._backend.extract_and_generate_template(
                raw_text, existing_template=existing_template
            )
            template = _build_template_from_llm_response(llm_response)
            template = sanitise_template(template, source_text or raw_text, store=self._store)

            # If the LLM ignored the existing_template_id instruction and created a new
            # template_id, but we know the near-miss family — merge into existing.
            if (
                existing_template is not None
                and template.template_id != existing_template.template_id
                and self._store.get(existing_template.template_id) is not None
            ):
                logger.warning(
                    "LLM created new id=%s despite existing_template=%s; merging into existing",
                    template.template_id, existing_template.template_id,
                )
                template = _merge_into_existing(template, existing_template)

            self._store.add(template)
            result.template_id = template.template_id
            result.llm_used = True
            result.llm_backend = self._backend.name
            result.llm_model = self._backend.model
            result.data = llm_response.get("extracted_data", {})
            result.validation_passed = True
            trace_llm_call(
                backend=self._backend.name,
                model=self._backend.model,
                existing_template_id=existing_template.template_id if existing_template else None,
                new_template_id=template.template_id,
                extracted_field_count=len(result.data),
            )
            logger.info(
                "LLM extraction complete template_id=%s fields=%s valid_rules=%s",
                template.template_id, len(result.data), len(template.extraction_rules),
            )
            return result
        except Exception:
            logger.exception(
                "LLM extraction failed backend=%s model=%s", self._backend.name, self._backend.model
            )
            raise

    def _write_audit(
        self,
        result: ExtractionResult,
        candidate_scores: dict[str, float],
        t_start: float,
    ) -> None:
        self._audit.record(
            document_path=result.document_path,
            template_id=result.template_id,
            match_score=result.match_score,
            candidate_scores=candidate_scores,
            llm_used=result.llm_used,
            llm_backend=result.llm_backend,
            llm_model=result.llm_model,
            validation_passed=result.validation_passed,
            validation_errors=result.validation_errors,
            field_count=len(result.data),
            duration_ms=(time.monotonic() - t_start) * 1000,
        )

    def _preview(self, text: str) -> str:
        return text[: self._raw_preview_chars] + ("..." if len(text) > self._raw_preview_chars else "")


def _merge_into_existing(new: Template, existing: Template) -> Template:
    """Merge a newly LLM-generated template into an existing one.

    - Reuses existing template_id, created_at, hit_count
    - Takes new extraction_rules (fresher regexes from LLM)
    - Unions keywords, deduplicates case-insensitively
    - Keeps the lower keyword_quorum of the two (more permissive)
    """
    existing_kws_lower = {kw.lower() for kw in existing.fingerprint.required_keywords}
    merged_kws = list(existing.fingerprint.required_keywords)
    for kw in new.fingerprint.required_keywords:
        if kw.lower() not in existing_kws_lower:
            merged_kws.append(kw)

    new.template_id = existing.template_id
    new.created_at = existing.created_at
    new.hit_count = existing.hit_count
    new.fingerprint.required_keywords = merged_kws
    new.fingerprint.keyword_quorum = min(
        new.fingerprint.keyword_quorum,
        existing.fingerprint.keyword_quorum,
    )
    logger.info(
        "Merged template id=%s keywords=%s->%s quorum=%.2f",
        new.template_id,
        len(existing.fingerprint.required_keywords),
        len(merged_kws),
        new.fingerprint.keyword_quorum,
    )
    return new


def _build_template_from_llm_response(resp: dict[str, Any]) -> Template:
    rules = [
        ExtractionRule(
            field=r["field"],
            type=r.get("type", "string"),
            regex=r.get("regex"),
            anchor_regex=r.get("anchor_regex"),
            stop_regex=r.get("stop_regex"),
            columns=r.get("columns"),
            date_format=r.get("date_format"),
        )
        for r in resp.get("extraction_rules", [])
    ]
    checks = [
        ConfidenceCheck(
            field=c["field"],
            not_null=c.get("not_null", False),
            gt=c.get("gt"),
            lt=c.get("lt"),
            regex_match=c.get("regex_match"),
        )
        for c in resp.get("confidence_checks", [])
    ]
    fp_raw = resp.get("fingerprint", {})
    fingerprint = Fingerprint(
        required_keywords=fp_raw.get("required_keywords", []),
        supplier_hint=fp_raw.get("supplier_hint", ""),
        doc_type=fp_raw.get("doc_type", "unknown"),
        keyword_quorum=fp_raw.get("keyword_quorum", 0.6),
    )
    return Template(
        template_id=resp.get("template_id", "unknown_v1"),
        fingerprint=fingerprint,
        extraction_rules=rules,
        confidence_checks=checks,
        metadata={"llm_generated": True},
    )
