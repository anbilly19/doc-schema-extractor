"""Ollama backend - supports gemma4:e4b-it-qat, qwen3.5:2b, gemma4:e2b."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..logging_utils import get_logger
from ..models import Template
from .base import LLMBackend, SYSTEM_PROMPT

SUPPORTED_MODELS = {"gemma4:e4b-it-qat", "qwen3.5:2b", "gemma4:e2b"}
logger = get_logger("backends.ollama")


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "gemma4:e4b-it-qat", base_url: str | None = None, timeout: float = 120.0):
        self._model = model
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._timeout = timeout
        logger.info("Initialized Ollama backend model=%s base_url=%s", model, self._base_url)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def extract_and_generate_template(self, raw_text: str, existing_template: Template | None = None) -> dict[str, Any]:
        update_hint = ""
        if existing_template:
            update_hint = f"\n\nNote: A previous template exists (id: {existing_template.template_id}). Keep the same template_id and improve it."

        prompt = f"{SYSTEM_PROMPT}{update_hint}\n\nDocument text to extract from:\n---\n{raw_text}\n---\n\nRespond ONLY with the JSON object."
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 4096},
        }
        logger.debug("Calling Ollama model=%s existing_template=%s payload_chars=%s", self._model, existing_template.template_id if existing_template else None, len(prompt))

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._base_url}/api/generate", json=payload)
            response.raise_for_status()

        result = response.json()
        raw_response = result.get("response", "{}").strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.split("\n", 1)[-1]
            raw_response = raw_response.rsplit("```", 1)[0]
        try:
            parsed = json.loads(raw_response)
            logger.info("Ollama response parsed model=%s keys=%s", self._model, list(parsed.keys()))
            return parsed
        except Exception:
            logger.exception("Failed to parse Ollama response model=%s raw_preview=%s", self._model, raw_response[:1000])
            raise
