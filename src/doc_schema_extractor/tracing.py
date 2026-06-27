"""LangSmith tracing helpers.

All functions are decorated with @traceable from the langsmith SDK.
They appear as named spans in the LangSmith UI under the configured project.

Tracing is automatically disabled when LANGSMITH_TRACING != 'true'.
"""

from __future__ import annotations

from typing import Any

from langsmith import traceable


@traceable(name="template_match")
def trace_template_match(
    raw_text_preview: str,
    threshold: float,
    template_count: int,
) -> dict[str, Any]:
    """Logged when template fingerprint matching runs."""
    return {
        "raw_text_preview": raw_text_preview[:500],
        "threshold": threshold,
        "template_count": template_count,
    }


@traceable(name="rule_engine_apply")
def trace_rule_engine(
    template_id: str,
    field_count: int,
) -> dict[str, Any]:
    """Logged when deterministic rule engine runs on a HIT."""
    return {"template_id": template_id, "field_count": field_count}


@traceable(name="validator_run")
def trace_validator(
    template_id: str,
    check_count: int,
) -> dict[str, Any]:
    """Logged when pydantic confidence checks run."""
    return {"template_id": template_id, "check_count": check_count}


@traceable(name="llm_template_generation")
def trace_llm_call(
    backend: str,
    model: str,
    existing_template_id: str | None = None,
) -> dict[str, Any]:
    """Logged on every LLM fallback call."""
    return {
        "backend": backend,
        "model": model,
        "existing_template_id": existing_template_id,
        "is_update": existing_template_id is not None,
    }


@traceable(name="chat_turn")
def trace_chat_turn(
    question: str,
    template_id: str | None,
    backend: str,
    model: str,
) -> dict[str, Any]:
    """Logged for every user message in the Streamlit chat UI."""
    return {
        "question": question,
        "template_id": template_id,
        "backend": backend,
        "model": model,
    }
