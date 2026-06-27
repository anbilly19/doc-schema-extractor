"""OpenAI backend - supports gpt-4o, gpt-4o-mini, o4-mini."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from ..models import Template
from .base import LLMBackend, SYSTEM_PROMPT

# Allowed models only
ALLOWED_MODELS = {"gpt-4o", "gpt-4o-mini", "o4-mini"}


class OpenAIBackend(LLMBackend):
    """OpenAI backend with JSON mode."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model}' not allowed. Choose from: {ALLOWED_MODELS}"
            )
        self._model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

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
                f"\n\nNote: A previous template exists for this document type "
                f"(id: {existing_template.template_id}). "
                f"Please update/improve it based on the new document. "
                f"Keep the same template_id."
            )

        user_content = (
            f"{update_hint}\n\nDocument text to extract from:\n"
            f"---\n{raw_text}\n---\n\nRespond ONLY with the JSON object."
        )

        # o4-mini doesn't support response_format=json_object in the same way
        # Use standard JSON mode for gpt-4o family
        kwargs: dict[str, Any] = {
            "model": self._model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        if self._model in {"gpt-4o", "gpt-4o-mini"}:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or "{}"

        # Strip markdown fences just in case
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]

        return json.loads(raw)
