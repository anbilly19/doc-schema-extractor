"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import Template


SYSTEM_PROMPT = """You are a document extraction specialist working with German supply-chain documents (Lieferscheine, Bestellungen, Auftragsbestaetigungen, Rechnungen, Aktionsmeldungen, Listungsmeldungen).

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
    "doc_type": "<order_confirmation|delivery_note|invoice|purchase_order|promotion_form|listing_form|other>",
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
  WRONG: "Bestellnummer\\s*(.*)"  <- .* is too greedy
  CORRECT: "Bestellnummer\\s*[:\\-]?\\s*([A-Za-z0-9\\-\\/]+)"
- For decimal amounts: capture the raw German number including dots and comma.
  Pattern example: "(?:Netto|Gesamt)\\s*[:\\-]?\\s*([\\d\\.]+,[\\d]{2})"
- For dates: capture ONLY the date digits, not the label.
  Pattern example: "Datum\\s*[:\\-]?\\s*(\\d{1,2}\\.\\d{1,2}\\.\\d{2,4})"
- For table fields: set regex to null and provide anchor_regex and stop_regex instead.
- anchor_regex and stop_regex do NOT need a capture group.
- Do NOT include trailing \\s or open-ended quantifiers that span multiple lines.
- All backslashes in JSON strings must be double-escaped: \\\\d not \\d.

DATE REGEX ANTI-PATTERNS - never produce any of these:
  WRONG: hallucinated placeholder  WRONG: garbage  WRONG: "YYYY-MM-DD" as a regex
  CORRECT for ISO dates:    "(\\d{4}-\\d{2}-\\d{2})"           date_format: "%Y-%m-%d"
  CORRECT for German dates: "(\\d{1,2}\\.\\d{1,2}\\.\\d{4})"  date_format: "%d.%m.%Y"
  date_format MUST use Python strptime codes only: %d %m %Y %y %H %M %S.
  If the document has no recognisable date, set regex to null and extracted_data value to null.

FINGERPRINT RULES:
- required_keywords must be 4-6 strings that appear VERBATIM in this doc type and NOT in others.
- Do not use generic words like "GmbH", "EUR", "Datum" as keywords.
- supplier_hint must be the exact company name string as it appears in the document.
- keyword_quorum: fraction of required_keywords that must match (default 0.6).
  IMPORTANT: prefer structural keywords (column headers, form labels, company name, email
  addresses, document type headers) that appear on EVERY instance of this doc type.
  Avoid instance-specific values (recipient names, addresses, product names, dates, KW numbers).

XLSX-SPECIFIC RULES (apply when the input text contains [Sheet: ...] markers):
- NEVER use the sheet name as a keyword. Sheet names like "Aktion", "Neulistung-Auslistung",
  "Tabelle1" vary between instances of the SAME form and will break matching.
- Use cell label text that is structurally invariant across all instances of this form.
  Good examples for Uplegger forms: "Aktionsnr. lt. Plan", "Vertrieb Innendienst",
  "KAM Pflichfelder", "Berechnungspreis Kunde", "aktion@uplegger.de".
- Dates in XLSX cells arrive as ISO strings: "2025-05-13 00:00:00".
  Write date regexes to match that format: regex "(\\d{4}-\\d{2}-\\d{2})" date_format "%Y-%m-%d".
  Do NOT write German dot-format date regexes for XLSX dates.
- For XLSX table extraction use anchor_regex on the column header row and stop_regex on the
  footer/remarks row (e.g. "Bemerkungen").

EXTRACTED_DATA RULES:
- extracted_data must have a key for every field in extraction_rules.
- Use null (JSON null) if the value is not found, never empty string or "N/A".
- For table fields, extracted_data value must be a JSON array of objects.

SELF-CHECK - complete these steps mentally before producing output:
For every extraction_rule where regex is not null:
  1. Find the exact value string in the document text above.
  2. Verify your regex returns that exact string as group(1).
  3. Set extracted_data to the SAME value.
  4. If you cannot verify, set regex to null and populate extracted_data with the literal value.
"""

EXISTING_TEMPLATE_PREFIX = """IMPORTANT: A similar template already exists in the store for this document family.
Existing template_id: {template_id}
Existing keywords:    {keywords}
Existing doc_type:    {doc_type}
Existing supplier:    {supplier_hint}

You MUST:
1. Reuse the SAME template_id: "{template_id}" - do NOT invent a new one.
2. In required_keywords, produce the UNION of the existing keywords and any new structural
   keywords you find in this document. Remove instance-specific values (recipient names,
   addresses, product names, greetings, KW numbers, SHEET NAMES). Keep only structural
   keywords present on EVERY instance of this doc type.
3. Set keyword_quorum to 0.6 unless all keywords are truly invariant.
4. Keep extraction_rules compatible with both documents - use alternation (A|B) where labels differ.
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
