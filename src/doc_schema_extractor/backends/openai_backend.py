"""OpenAI backend - supports gpt-4o, gpt-4o-mini, o4-mini, gpt-4.1 family, gpt-5 family."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from ..logging_utils import get_logger
from ..models import Template
from .base import LLMBackend, SYSTEM_PROMPT

# Models that support the json_object response_format parameter.
_JSON_MODE_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"}

# Full set of allowed model identifiers.
ALLOWED_MODELS = _JSON_MODE_MODELS | {"o4-mini", "gpt-5", "gpt-5-mini"}

logger = get_logger("backends.openai")


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None):
        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model}' not allowed. Choose from: {sorted(ALLOWED_MODELS)}"
            )
        self._model = model
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info("Initialized OpenAI backend model=%s", model)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def extract_and_generate_template(
        self, raw_text: str, existing_template: Template | None = None
    ) -> dict[str, Any]:
        update_hint = ""
        if existing_template:
            update_hint = (
                f"\n\nNote: A previous template exists (id: {existing_template.template_id}). "
                "Keep the same template_id and improve it."
            )

        user_content = (
            f"{update_hint}\n\nDocument text to extract from:\n---\n{raw_text}\n---\n\n"
            "Respond ONLY with the JSON object."
        )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        # json_object response_format is only supported on chat-completion models,
        # not on reasoning (o-series) or gpt-5 models which return JSON natively.
        if self._model in _JSON_MODE_MODELS:
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug(
            "Calling OpenAI model=%s existing_template=%s payload_chars=%s",
            self._model,
            existing_template.template_id if existing_template else None,
            len(user_content),
        )
        response = self._client.chat.completions.create(**kwargs)
        raw = (response.choices[0].message.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
        try:
            parsed = json.loads(raw)
            logger.info(
                "OpenAI response parsed model=%s keys=%s", self._model, list(parsed.keys())
            )
            return parsed
        except Exception:
            logger.exception(
                "Failed to parse OpenAI response model=%s raw_preview=%s",
                self._model,
                raw[:1000],
            )
            raise
