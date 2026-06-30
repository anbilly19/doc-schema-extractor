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
    "doc_type": "<order_confirmation|delivery_note|invoice|purchase_order|other>",
    "keyword_quorum": 0.6
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
- For decimal amounts: capture the raw German number including dots and comma: e.g. (1\\.234,56)
  Pattern example: "(?:Netto|Gesamt)\\s*[:\\-]?\\s*([\\d\\.]+,[\\d]{2})"
- For dates: capture ONLY the date digits DD.MM.YYYY, not the label word.
  Pattern example: "Datum\\s*[:\\-]?\\s*(\\d{1,2}\\.\\d{1,2}\\.\\d{2,4})"
- For table fields: set regex to null and provide anchor_regex and stop_regex instead.
- anchor_regex and stop_regex are also valid Python re patterns but do NOT need a capture group.
- Do NOT include trailing \\s or open-ended quantifiers that span multiple lines.
- All backslashes in JSON strings must be double-escaped: \\\\d not \\d.

DATE REGEX ANTI-PATTERNS — never produce any of these:
  WRONG: hallucinated placeholder with dashes like "([A-Za-z]{4}-... ...)"  <- never
  WRONG: garbage like "(0q.0v.2&4)"                <- not a regex
  WRONG: "YYYY-MM-DD"                              <- not a format string
  CORRECT for ISO dates:    "(\\d{4}-\\d{2}-\\d{2})"           date_format: "%Y-%m-%d"
  CORRECT for German dates: "(\\d{1,2}\\.\\d{1,2}\\.\\d{4})"  date_format: "%d.%m.%Y"
  date_format MUST use Python strptime codes only: %d %m %Y %y %H %M %S — nothing else.
  If the document has no recognisable date, set regex to null and extracted_data value to null.

FINGERPRINT RULES:
- required_keywords must be 4-6 words that appear VERBATIM in this doc type and NOT in others.
- Do not use generic words like "GmbH", "EUR", "Datum" as keywords.
- supplier_hint must be the exact company name string as it appears in the document.
- keyword_quorum: float between 0.0 and 1.0 — what fraction of required_keywords must match
  for this template to be selected. Use 0.6 as default (3 of 5 keywords suffice).
  Use 0.8+ only if ALL your keywords are truly invariant across every instance of this doc type.
  IMPORTANT: keywords that appear only on SOME instances (recipient names, addresses,
  greetings, product names) LOWER the effective quorum — prefer structural keywords
  (column headers, form labels, company name, document type headers) that appear on EVERY instance.

EXTRACTED_DATA RULES:
- extracted_data must have a key for every field in extraction_rules.
- Use null (JSON null) if the value is not found, do NOT use empty string or "N/A".
- For table fields, extracted_data value must be a JSON array of objects (one per row).

SELF-CHECK — complete these steps mentally before producing output:
For every extraction_rule where regex is not null:
  1. Find the exact value string in the document text above (e.g. "7076182617" or "18.06.2026").
  2. Write your regex so that applying it to the document text returns that exact string as group(1).
  3. Set extracted_data for that field to the SAME value your regex would capture.
  4. If you cannot construct a regex that matches the actual value in the text,
     set regex to null and still populate extracted_data with the literal value.

Example grounding check:
  Document contains: "Bestelldatum  18.06.2026"
  extracted_data["order_date"] = "18.06.2026"
  regex = "Bestelldatum\\s*(\\d{1,2}\\.\\d{1,2}\\.\\d{4})"
  Verify: does (\\d{1,2}\\.\\d{1,2}\\.\\d{4}) match "18.06.2026"? YES -> ship it.
"""

EXISTING_TEMPLATE_PREFIX = """IMPORTANT: A similar template already exists in the store for this document family.
Existing template_id: {template_id}
Existing keywords:    {keywords}
Existing doc_type:    {doc_type}
Existing supplier:    {supplier_hint}

You MUST:
1. Reuse the SAME template_id: "{template_id}" — do NOT invent a new one.
2. In required_keywords, produce the UNION of the existing keywords and any new structural
   keywords you find in this document. Remove keywords that are document-instance specific
   (recipient names, addresses, product names, greetings). Keep only structural keywords
   that appear on EVERY document of this type (form labels, column headers, company name,
   document type headers).
3. Set keyword_quorum to 0.6 unless you are certain all keywords are invariant.
4. Keep extraction_rules compatible with both documents — use alternation (A|B) in regexes
   where the label text differs slightly between instances.
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
