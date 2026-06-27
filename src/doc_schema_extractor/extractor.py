"""Main Extractor orchestrator with LangSmith tracing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langsmith import traceable

from .backends.base import LLMBackend
from .backends.ollama import OllamaBackend
from .backends.openai_backend import OpenAIBackend
from .logging_utils import get_logger
from .models import ConfidenceCheck, ExtractionResult, ExtractionRule, Fingerprint, Template
from .rule_engine import RuleEngine
from .template_store import TemplateStore
from .text_extractor import TextExtractor
from .tracing import trace_llm_call, trace_rule_engine, trace_template_match, trace_validator
from .validator import Validator

logger = get_logger("extractor")


def _build_default_backend() -> LLMBackend:
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "openai":
        model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        logger.info("Using default OpenAI backend model=%s", model)
        return OpenAIBackend(model=model)
    model = os.getenv("OLLAMA_DEFAULT_MODEL", "gemma4:e4b-it-qat")
    logger.info("Using default Ollama backend model=%s", model)
    return OllamaBackend(model=model)


class Extractor:
    def __init__(self, backend: LLMBackend | None = None, store_path: str | Path = "templates/store.json", match_threshold: float | None = None):
        self._backend = backend or _build_default_backend()
        self._store = TemplateStore(store_path)
        self._text_extractor = TextExtractor()
        self._rule_engine = RuleEngine()
        self._validator = Validator()
        self._threshold = match_threshold or float(os.getenv("TEMPLATE_MATCH_THRESHOLD", "0.75"))
        self._raw_preview_chars = int(os.getenv("LOG_RAW_TEXT_PREVIEW_CHARS", "2000"))
        logger.info("Extractor initialized backend=%s model=%s threshold=%.2f store=%s", self._backend.name, self._backend.model, self._threshold, store_path)

    @traceable(name="extraction_run")
    def extract(self, file_path: str | Path) -> ExtractionResult:
        logger.info("Extraction run started file=%s", file_path)
        path = Path(file_path)
        doc = self._text_extractor.extract(path)

        logger.debug("Document summary file=%s type=%s chars=%s pages=%s preview=%s", path, doc.file_type, len(doc.full_text), len(doc.pages), self._preview(doc.full_text))

        result = ExtractionResult(document_path=str(path), raw_text=doc.full_text)

        trace_template_match(raw_text_preview=doc.full_text, threshold=self._threshold, template_count=len(self._store.list_all()))
        template, score = self._store.match(doc.full_text, self._threshold)
        result.match_score = score

        if template:
            logger.info("Template HIT template_id=%s score=%.3f", template.template_id, score)
            result.template_id = template.template_id
            trace_rule_engine(template_id=template.template_id, field_count=len(template.extraction_rules))
            data = self._rule_engine.apply(template, doc)
            trace_validator(template_id=template.template_id, check_count=len(template.confidence_checks))
            passed, errors = self._validator.validate(data, template)

            if passed:
                result.data = data
                result.validation_passed = True
                result.llm_used = False
                template.increment_hit()
                self._store.add(template)
                logger.info("Extraction completed via template template_id=%s", template.template_id)
                return result

            logger.warning("Template validation failed template_id=%s errors=%s", template.template_id, errors)
            result.validation_errors = errors
            return self._llm_extract(result, doc.full_text, existing_template=template)

        logger.info("Template MISS score=%.3f; falling back to LLM", score)
        return self._llm_extract(result, doc.full_text)

    def _llm_extract(self, result: ExtractionResult, raw_text: str, existing_template: Template | None = None) -> ExtractionResult:
        logger.info("LLM extraction start backend=%s model=%s existing_template=%s preview=%s", self._backend.name, self._backend.model, existing_template.template_id if existing_template else None, self._preview(raw_text))
        trace_llm_call(backend=self._backend.name, model=self._backend.model, existing_template_id=existing_template.template_id if existing_template else None)
        try:
            llm_response = self._backend.extract_and_generate_template(raw_text, existing_template=existing_template)
            template = _build_template_from_llm_response(llm_response)
            self._store.add(template)
            result.template_id = template.template_id
            result.llm_used = True
            result.llm_backend = self._backend.name
            result.llm_model = self._backend.model
            result.data = llm_response.get("extracted_data", {})
            result.validation_passed = True
            logger.info("LLM extraction complete template_id=%s fields=%s", template.template_id, len(result.data))
            return result
        except Exception:
            logger.exception("LLM extraction failed backend=%s model=%s", self._backend.name, self._backend.model)
            raise

    def _preview(self, text: str) -> str:
        return text[: self._raw_preview_chars] + ("..." if len(text) > self._raw_preview_chars else "")


def _build_template_from_llm_response(resp: dict[str, Any]) -> Template:
    rules = [ExtractionRule(field=r["field"], type=r.get("type", "string"), regex=r.get("regex"), anchor_regex=r.get("anchor_regex"), stop_regex=r.get("stop_regex"), columns=r.get("columns"), date_format=r.get("date_format")) for r in resp.get("extraction_rules", [])]
    checks = [ConfidenceCheck(field=c["field"], not_null=c.get("not_null", False), gt=c.get("gt"), lt=c.get("lt"), regex_match=c.get("regex_match")) for c in resp.get("confidence_checks", [])]
    fp_raw = resp.get("fingerprint", {})
    fingerprint = Fingerprint(required_keywords=fp_raw.get("required_keywords", []), supplier_hint=fp_raw.get("supplier_hint", ""), doc_type=fp_raw.get("doc_type", "unknown"))
    return Template(template_id=resp.get("template_id", "unknown_v1"), fingerprint=fingerprint, extraction_rules=rules, confidence_checks=checks, metadata={"llm_generated": True})
