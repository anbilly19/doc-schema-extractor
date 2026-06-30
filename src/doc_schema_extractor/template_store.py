"""Persistent template store - JSON-backed."""

from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz

from .logging_utils import get_logger
from .models import Template

logger = get_logger("template_store")
DEFAULT_STORE_PATH = Path("templates/store.json")


class TemplateStore:
    def __init__(self, store_path: str | Path = DEFAULT_STORE_PATH):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, Template] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            logger.debug("Loading template store path=%s", self.store_path)
            with open(self.store_path, encoding="utf-8") as f:
                raw = json.load(f)
            for tid, data in raw.items():
                self._templates[tid] = Template.model_validate(data)
            logger.info("Loaded templates count=%s path=%s", len(self._templates), self.store_path)
        else:
            logger.info("Template store not found; starting empty path=%s", self.store_path)

    def _save(self) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            data = {tid: json.loads(t.model_dump_json()) for tid, t in self._templates.items()}
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved templates count=%s path=%s", len(self._templates), self.store_path)

    def add(self, template: Template) -> None:
        logger.debug("Adding/updating template id=%s version=%s", template.template_id, template.version)
        self._templates[template.template_id] = template
        self._save()

    def get(self, template_id: str) -> Template | None:
        logger.debug("Fetching template id=%s hit=%s", template_id, template_id in self._templates)
        return self._templates.get(template_id)

    def list_all(self) -> list[Template]:
        return list(self._templates.values())

    def delete(self, template_id: str) -> bool:
        if template_id in self._templates:
            logger.info("Deleting template id=%s", template_id)
            del self._templates[template_id]
            self._save()
            return True
        logger.warning("Delete requested for missing template id=%s", template_id)
        return False

    def match_with_scores(
        self, normalised_text: str, threshold: float = 0.75
    ) -> tuple[Template | None, float, dict[str, float]]:
        """Match template against normalised document text.

        Scoring: keyword hit-rate (0-1) + small supplier fuzzy-match boost (0-0.1).
        Both the document text and stored keywords are lowercased for comparison.
        """
        if not self._templates:
            logger.info("Template match skipped: no templates loaded")
            return None, 0.0, {}

        text_lower = normalised_text.lower()
        best_template: Template | None = None
        best_score = 0.0
        candidate_scores: dict[str, float] = {}

        for template in self._templates.values():
            fp = template.fingerprint
            keywords = fp.required_keywords
            hits = sum(1 for kw in keywords if kw.lower() in text_lower)
            keyword_score = hits / len(keywords) if keywords else 0.0
            supplier_boost = 0.0
            if fp.supplier_hint:
                ratio = fuzz.partial_ratio(fp.supplier_hint.lower(), text_lower) / 100
                supplier_boost = 0.1 * ratio
            score = min(1.0, keyword_score + supplier_boost)
            candidate_scores[template.template_id] = round(score, 4)
            logger.debug(
                "Template candidate id=%s keyword_score=%.3f supplier_boost=%.3f total=%.3f",
                template.template_id, keyword_score, supplier_boost, score,
            )
            if score > best_score:
                best_score = score
                best_template = template

        if best_score >= threshold:
            logger.info(
                "Template match HIT id=%s score=%.3f threshold=%.3f",
                best_template.template_id if best_template else None, best_score, threshold,
            )
            return best_template, best_score, candidate_scores

        logger.info(
            "Template match MISS best_id=%s score=%.3f threshold=%.3f",
            best_template.template_id if best_template else None, best_score, threshold,
        )
        return None, best_score, candidate_scores

    def match(self, normalised_text: str, threshold: float = 0.75) -> tuple[Template | None, float]:
        t, s, _ = self.match_with_scores(normalised_text, threshold)
        return t, s
