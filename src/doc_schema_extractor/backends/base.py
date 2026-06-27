"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import Template


SYSTEM_PROMPT = """You are a document extraction specialist. Your task is two-fold:
1. Extract structured data fields from the provided document text.
2. Generate reusable machine-readable extraction rules (regex patterns) for the same document type.

You MUST respond ONLY with valid JSON matching this schema:
{
  "template_id": "<supplier_doctype_v1 - snake_case, descriptive>",
  "fingerprint": {
    "required_keywords": ["keyword1", "keyword2", ...],  // 4-8 keywords unique to this doc type
    "supplier_hint": "<supplier company name>",
    "doc_type": "<order_confirmation|delivery_note|invoice|purchase_order|other>"
  },
  "extraction_rules": [
    {
      "field": "<field_name>",
      "type": "<string|date|decimal|integer|table|list>",
      "regex": "<python regex with one capture group, or null for tables>",
      "anchor_regex": "<regex matching table header row, for table type only>",
      "stop_regex": "<regex matching line after last table row, for table type only>",
      "columns": ["col1", "col2", ...],  // for table type only
      "date_format": "%d.%m.%Y"  // for date type, German format
    }
  ],
  "confidence_checks": [
    {"field": "order_number", "not_null": true},
    {"field": "total_gross", "gt": 0}
  ],
  "extracted_data": {
    // The actual extracted values from THIS document
    "order_number": "...",
    "order_date": "...",
    ...
  }
}

IMPORTANT:
- Regexes must be valid Python re patterns with exactly one capture group ().
- For German number formats: 1.234,56 - the regex should capture '1.234,56' (include punctuation).
- For German dates: capture as DD.MM.YYYY.
- Keep keywords specific enough to avoid false matches with other document types.
- The extracted_data dict must contain ALL fields defined in extraction_rules.
"""


class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        ...

    @abstractmethod
    def extract_and_generate_template(
        self, raw_text: str, existing_template: Template | None = None
    ) -> dict[str, Any]:
        """Call LLM to extract data AND generate template rules.
        
        Returns the full JSON response from the LLM as a dict.
        """
        ...
