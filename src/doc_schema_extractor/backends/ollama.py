"""Ollama backend - supports gemma4:e4b-it-qat, qwen3.5:2b, gemma4:e2b."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..models import Template
from .base import LLMBackend, SYSTEM_PROMPT

# Supported local models
SUPPORTED_MODELS = {
    "gemma4:e4b-it-qat",
    "qwen3.5:2b",
    "gemma4:e2b",
}


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(
        self,
        model: str = "gemma4:e4b-it-qat",
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self._model = model
        self._base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

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

        prompt = (
            f"{SYSTEM_PROMPT}{update_hint}\n\n"
            f"Document text to extract from:\n"
            f"---\n{raw_text}\n---\n\n"
            f"Respond ONLY with the JSON object."
        )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,  # Low temp for deterministic structured output
                "num_predict": 4096,
            },
        }

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()

        result = response.json()
        raw_response = result.get("response", "{}")

        # Strip markdown code fences if model wraps JSON
        raw_response = raw_response.strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.split("\n", 1)[-1]
            raw_response = raw_response.rsplit("```", 1)[0]

        return json.loads(raw_response)

    def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self._model in m for m in models)
        except Exception:
            return False
