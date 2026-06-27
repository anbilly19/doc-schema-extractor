"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import Template


SYSTEM_PROMPT = """You are a document extraction specialist working with German supply-chain documents (Lieferscheine, Bestellungen, Auftragsbestaetigungen, Rechnungen).

Your task is TWO-FOLD:
1. Extract the actual field values from the document text provided.
2. Generate reusable Python regex extraction rules for the same document type.

You MUST respond ONLY with a single valid JSON object. No markdown fences, no explanation text.

JSON schema:
{
  "template_id": "<snake_case_descriptive_id_v1>",
  "fingerprint": {
    "required_keywords": ["word1", "word2", "word3", "word4"],
    "supplier_hint": "<exact supplier company name from document>",
    "doc_type": "<order_confirmation|delivery_note|invoice|purchase_order|other>"
  },
  "extraction_rules": [
    {
      "field": "order_number",
      "type": "string",
      "regex": "(?:Bestellung|Bestellnummer|Auftrag(?:snummer)?)\\s*[:\\-]?\\s*([A-Za-z0-9\\-\\/]+)"
    },
    {
      "field": "order_date",
      "type": "date",
      "regex": "(?:Bestelldatum|Datum)\\s*[:\\-]?\\s*(\\d{1,2}\\.\\d{1,2}\\.\\d{2,4})",
      "date_format": "%d.%m.%Y"
    },
    {
      "field": "total_net",
      "type": "decimal",
      "regex": "(?:Positionsnetto|Nettobetrag|Gesamt(?:netto)?)\\s*[:\\-]?\\s*([\\d\\.]+,[\\d]{2})"
    },
    {
      "field": "line_items",
      "type": "table",
      "regex": null,
      "anchor_regex": "(?:Pos|Position|Art).*(?:Bezeichnung|Artikel|Beschreibung)",
      "stop_regex": "(?:Summe|Gesamt|Endsumme|Zahlungsbed)",
      "columns": ["pos", "article_number", "description", "quantity", "unit", "price", "total"]
    }
  ],
  "confidence_checks": [
    {"field": "order_number", "not_null": true}
  ],
  "extracted_data": {
    "order_number": "<actual value from document or null>",
    "order_date": "<actual value>",
    "total_net": "<actual value>",
    "line_items": []
  }
}

CRITICAL RULES FOR REGEXES:
- Every regex MUST be a valid Python re pattern.
- Every non-null regex MUST have EXACTLY ONE capture group: one pair of unescaped ( ).
- The capture group must surround the VALUE, not the label.
  WRONG: "(Bestellnummer\\s*[:\\-]?\\s*)([A-Z0-9]+)"  <- two groups
  WRONG: "Bestellnummer\\s*(.*)"  <- .* is too greedy, captures the whole rest of line
  CORRECT: "Bestellnummer\\s*[:\\-]?\\s*([A-Za-z0-9\\-\\/]+)"
- For decimal amounts: capture the raw German number including dots and comma: e.g. (1\.234,56)
  Pattern example: "(?:Netto|Gesamt)\\s*[:\\-]?\\s*([\\d\\.]+,[\\d]{2})"
- For dates: capture ONLY the date digits DD.MM.YYYY, not the label word.
  Pattern example: "Datum\\s*[:\\-]?\\s*(\\d{1,2}\\.\\d{1,2}\\.\\d{2,4})"
- For table fields: set regex to null and provide anchor_regex and stop_regex instead.
- anchor_regex and stop_regex are also valid Python re patterns but do NOT need a capture group.
- Do NOT include trailing \\s or open-ended quantifiers that span multiple lines.
- All backslashes in JSON strings must be double-escaped: \\d not \d.

FINGERPRINT RULES:
- required_keywords must be 4-6 words that appear VERBATIM in this doc type and NOT in others.
- Do not use generic words like "GmbH", "EUR", "Datum" as keywords.
- supplier_hint must be the exact company name string as it appears in the document.

EXTRACTED_DATA RULES:
- extracted_data must have a key for every field in extraction_rules.
- Use null (JSON null) if the value is not found, do NOT use empty string or "N/A".
- For table fields, extracted_data value must be a JSON array of objects (one per row).
"""


class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def extract_and_generate_template(
        self, raw_text: str, existing_template: Template | None = None
    ) -> dict[str, Any]:
        """Call LLM to extract data AND generate template rules. Returns full JSON as dict."""
        ...
