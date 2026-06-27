"""Ollama backend - supports gemma4:e4b-it-qat, qwen3.5:2b, gemma4:e2b.

Uses streaming generation to avoid a single wall-clock timeout on large
JSON responses. Tokens are accumulated as they arrive; the connect timeout
and per-chunk read timeout are separate from the total generation time.
"""

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
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        prompt_max_chars: int | None = None,
    ):
        self._model = model or os.getenv("OLLAMA_DEFAULT_MODEL", "gemma4:e4b-it-qat")
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Configurable timeout; default 300s to handle slow local models
        self._timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT", "300"))
        # Cap document text sent to the model to keep prompts manageable
        self._prompt_max_chars = prompt_max_chars or int(os.getenv("OLLAMA_PROMPT_MAX_CHARS", "6000"))
        logger.info(
            "Initialized Ollama backend model=%s base_url=%s timeout=%s prompt_max_chars=%s",
            self._model, self._base_url, self._timeout, self._prompt_max_chars,
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def extract_and_generate_template(
        self, raw_text: str, existing_template: Template | None = None
    ) -> dict[str, Any]:
        # Truncate document text to avoid oversized prompts
        truncated = raw_text[: self._prompt_max_chars]
        if len(raw_text) > self._prompt_max_chars:
            logger.warning(
                "Document text truncated for LLM prompt original=%s limit=%s",
                len(raw_text), self._prompt_max_chars,
            )

        update_hint = ""
        if existing_template:
            update_hint = (
                f"\n\nNote: A previous template exists (id: {existing_template.template_id}). "
                f"Keep the same template_id and improve it."
            )

        prompt = (
            f"{SYSTEM_PROMPT}{update_hint}\n\n"
            f"Document text to extract from:\n---\n{truncated}\n---\n\n"
            f"Respond ONLY with the JSON object."
        )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,          # stream tokens; avoids single wall-clock timeout
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 4096},
        }

        logger.debug(
            "Calling Ollama (streaming) model=%s existing_template=%s prompt_chars=%s",
            self._model,
            existing_template.template_id if existing_template else None,
            len(prompt),
        )

        # httpx timeout: connect + each read chunk, NOT total wall-clock
        timeout = httpx.Timeout(connect=10.0, read=self._timeout, write=30.0, pool=5.0)
        chunks: list[str] = []

        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{self._base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON stream line (skipped): %s", line[:200])
                        continue
                    token = chunk.get("response", "")
                    chunks.append(token)
                    if chunk.get("done"):
                        stats = {
                            k: chunk.get(k)
                            for k in ("total_duration", "eval_count", "eval_duration")
                            if chunk.get(k) is not None
                        }
                        logger.info("Ollama stream done model=%s stats=%s", self._model, stats)
                        break

        raw_response = "".join(chunks).strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.split("\n", 1)[-1]
            raw_response = raw_response.rsplit("```", 1)[0]

        try:
            parsed = json.loads(raw_response)
            logger.info("Ollama response parsed model=%s keys=%s", self._model, list(parsed.keys()))
            return parsed
        except Exception:
            logger.exception(
                "Failed to parse Ollama response model=%s raw_preview=%s",
                self._model, raw_response[:1000],
            )
            raise

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self._model in m for m in models)
        except Exception:
            return False
