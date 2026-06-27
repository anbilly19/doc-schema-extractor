"""Persistent template store - JSON-backed, no external DB dependency."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

from .models import Template


DEFAULT_STORE_PATH = Path("templates/store.json")


class TemplateStore:
    """Load, match, and persist extraction templates."""

    def __init__(self, store_path: str | Path = DEFAULT_STORE_PATH):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, Template] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            with open(self.store_path) as f:
                raw = json.load(f)
            for tid, data in raw.items():
                self._templates[tid] = Template.model_validate(data)

    def _save(self) -> None:
        with open(self.store_path, "w") as f:
            data = {
                tid: json.loads(t.model_dump_json())
                for tid, t in self._templates.items()
            }
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def add(self, template: Template) -> None:
        """Add or overwrite a template and persist."""
        self._templates[template.template_id] = template
        self._save()

    def get(self, template_id: str) -> Template | None:
        return self._templates.get(template_id)

    def list_all(self) -> list[Template]:
        return list(self._templates.values())

    def delete(self, template_id: str) -> bool:
        if template_id in self._templates:
            del self._templates[template_id]
            self._save()
            return True
        return False

    def match(self, raw_text: str, threshold: float = 0.75) -> tuple[Template | None, float]:
        """Score raw_text against all templates and return best match above threshold.
        
        Scoring strategy:
          1. Keyword hit ratio (primary)
          2. Supplier hint fuzzy match (secondary boost)
        """
        if not self._templates:
            return None, 0.0

        best_template: Template | None = None
        best_score = 0.0

        for template in self._templates.values():
            fp = template.fingerprint
            keywords = fp.required_keywords

            # Primary: keyword overlap ratio
            hits = sum(1 for kw in keywords if kw.lower() in raw_text.lower())
            keyword_score = hits / len(keywords) if keywords else 0.0

            # Secondary: fuzzy supplier name match (boost up to +0.1)
            supplier_boost = 0.0
            if fp.supplier_hint:
                ratio = fuzz.partial_ratio(fp.supplier_hint.lower(), raw_text.lower()) / 100
                supplier_boost = 0.1 * ratio

            score = min(1.0, keyword_score + supplier_boost)

            if score > best_score:
                best_score = score
                best_template = template

        if best_score >= threshold:
            return best_template, best_score
        return None, best_score
