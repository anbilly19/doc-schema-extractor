"""Main Extractor orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .backends.base import LLMBackend
from .backends.ollama import OllamaBackend
from .backends.openai_backend import OpenAIBackend
from .models import ExtractionResult, Template, Fingerprint, ExtractionRule, ConfidenceCheck
from .rule_engine import RuleEngine
from .template_store import TemplateStore
from .text_extractor import TextExtractor
from .validator import Validator


def _build_default_backend() -> LLMBackend:
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "openai":
        model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        return OpenAIBackend(model=model)
    model = os.getenv("OLLAMA_DEFAULT_MODEL", "gemma4:e4b-it-qat")
    return OllamaBackend(model=model)


class Extractor:
    """Main extraction orchestrator.
    
    Flow:
      1. Extract text from document
      2. Match against template store
      3a. HIT:  apply rules deterministically → validate → return
      3b. MISS: call LLM → save template → return
      4. If validation fails: fallback to LLM and update template
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        store_path: str | Path = "templates/store.json",
        match_threshold: float | None = None,
    ):
        self._backend = backend or _build_default_backend()
        self._store = TemplateStore(store_path)
        self._text_extractor = TextExtractor()
        self._rule_engine = RuleEngine()
        self._validator = Validator()
        self._threshold = match_threshold or float(
            os.getenv("TEMPLATE_MATCH_THRESHOLD", "0.75")
        )

    def extract(self, file_path: str | Path) -> ExtractionResult:
        """Extract structured data from a document."""
        path = Path(file_path)
        doc = self._text_extractor.extract(path)

        result = ExtractionResult(
            document_path=str(path),
            raw_text=doc.full_text,
        )

        # Step 1: Try to match a template
        template, score = self._store.match(doc.full_text, self._threshold)
        result.match_score = score

        if template:
            # HIT: apply deterministic rules
            result.template_id = template.template_id
            data = self._rule_engine.apply(template, doc)
            passed, errors = self._validator.validate(data, template)

            if passed:
                result.data = data
                result.validation_passed = True
                result.llm_used = False
                template.increment_hit()
                self._store.add(template)
                return result
            else:
                # Validation failed → LLM fallback + template update
                result.validation_errors = errors
                return self._llm_extract(
                    result, doc.full_text, existing_template=template
                )
        else:
            # MISS: LLM extraction + template creation
            return self._llm_extract(result, doc.full_text)

    def _llm_extract(
        self,
        result: ExtractionResult,
        raw_text: str,
        existing_template: Template | None = None,
    ) -> ExtractionResult:
        llm_response = self._backend.extract_and_generate_template(
            raw_text, existing_template=existing_template
        )

        # Build and save template from LLM response
        template = _build_template_from_llm_response(llm_response)
        self._store.add(template)

        result.template_id = template.template_id
        result.llm_used = True
        result.llm_backend = self._backend.name
        result.llm_model = self._backend.model
        result.data = llm_response.get("extracted_data", {})
        result.validation_passed = True

        return result


def _build_template_from_llm_response(resp: dict[str, Any]) -> Template:
    """Parse LLM JSON response into a Template model."""
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
    )

    return Template(
        template_id=resp.get("template_id", "unknown_v1"),
        fingerprint=fingerprint,
        extraction_rules=rules,
        confidence_checks=checks,
        metadata={
            "llm_generated": True,
            "source_response_keys": list(resp.keys()),
        },
    )
