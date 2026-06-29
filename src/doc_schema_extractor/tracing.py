"""LangSmith tracing helpers.

All functions are decorated with @traceable from the langsmith SDK.
They appear as named spans in the LangSmith UI under the configured project.

Tracing is automatically disabled when LANGSMITH_TRACING != 'true'.

IMPORTANT: each function must receive the *results* of the operation as
arguments so that LangSmith records outputs != inputs.  Callers must invoke
these functions AFTER the actual work has completed.
"""

from __future__ import annotations

from typing import Any

from langsmith import traceable


@traceable(name="template_match")
def trace_template_match(
    raw_text_preview: str,
    threshold: float,
    template_count: int,
    # ── results (must be populated by caller after match runs) ────────────
    matched_template_id: str | None,
    match_score: float,
    candidate_scores: dict[str, float],
) -> dict[str, Any]:
    """Logged when template fingerprint matching runs."""
    return {
        "raw_text_preview": raw_text_preview[:500],
        "threshold": threshold,
        "template_count": template_count,
        "matched_template_id": matched_template_id,
        "match_score": round(match_score, 4),
        "candidate_scores": {k: round(v, 4) for k, v in candidate_scores.items()},
        "hit": matched_template_id is not None,
    }


@traceable(name="rule_engine_apply")
def trace_rule_engine(
    template_id: str,
    field_count: int,
    # ── results ────────────────────────────────────────────────────────────
    extracted_fields: dict[str, Any],
) -> dict[str, Any]:
    """Logged when deterministic rule engine runs on a HIT."""
    return {
        "template_id": template_id,
        "field_count": field_count,
        "extracted_fields": extracted_fields,
        "non_null_count": sum(1 for v in extracted_fields.values() if v not in (None, "", [])),
    }


@traceable(name="validator_run")
def trace_validator(
    template_id: str,
    check_count: int,
    # ── results ────────────────────────────────────────────────────────────
    passed: bool,
    errors: list[str],
) -> dict[str, Any]:
    """Logged when pydantic confidence checks run."""
    return {
        "template_id": template_id,
        "check_count": check_count,
        "passed": passed,
        "error_count": len(errors),
        "errors": errors,
    }


@traceable(name="llm_template_generation")
def trace_llm_call(
    backend: str,
    model: str,
    existing_template_id: str | None = None,
    # ── results ────────────────────────────────────────────────────────────
    new_template_id: str | None = None,
    extracted_field_count: int = 0,
) -> dict[str, Any]:
    """Logged on every LLM fallback call."""
    return {
        "backend": backend,
        "model": model,
        "existing_template_id": existing_template_id,
        "is_update": existing_template_id is not None,
        "new_template_id": new_template_id,
        "extracted_field_count": extracted_field_count,
    }


@traceable(name="chat_turn")
def trace_chat_turn(
    question: str,
    template_id: str | None,
    backend: str,
    model: str,
    # ── results (populate AFTER answer is generated) ───────────────────────
    answer: str,
    llm_used: bool,
) -> dict[str, Any]:
    """Logged for every user message in the Streamlit chat UI."""
    return {
        "question": question,
        "template_id": template_id,
        "backend": backend,
        "model": model,
        "answer": answer,
        "llm_used": llm_used,
        "answer_chars": len(answer),
    }
