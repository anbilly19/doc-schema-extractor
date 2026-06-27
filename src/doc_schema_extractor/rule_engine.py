"""Deterministic rule engine - applies saved extraction rules without LLM."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import ExtractionRule, Template
from .text_extractor import DocumentContent


class RuleEngine:
    """Apply template extraction rules deterministically."""

    def apply(self, template: Template, doc: DocumentContent) -> dict[str, Any]:
        result: dict[str, Any] = {}
        text = doc.full_text

        for rule in template.extraction_rules:
            try:
                value = self._apply_rule(rule, text, doc)
                result[rule.field] = value
            except Exception as e:
                result[rule.field] = None
                result[f"{rule.field}.__error"] = str(e)

        return result

    def _apply_rule(self, rule: ExtractionRule, text: str, doc: DocumentContent) -> Any:
        match rule.type:
            case "table":
                return self._extract_table(rule, doc)
            case "string":
                return self._extract_regex(rule, text)
            case "date":
                return self._extract_date(rule, text)
            case "decimal":
                return self._extract_decimal(rule, text)
            case "integer":
                raw = self._extract_regex(rule, text)
                return int(raw.replace(".", "").replace(",", "")) if raw else None
            case "list":
                return self._extract_list(rule, text)
            case _:
                return self._extract_regex(rule, text)

    def _extract_regex(self, rule: ExtractionRule, text: str) -> str | None:
        if not rule.regex:
            return None
        match = re.search(rule.regex, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            return None
        value = match.group(1).strip()
        if rule.strip_chars:
            value = value.strip(rule.strip_chars)
        return value

    def _extract_date(self, rule: ExtractionRule, text: str) -> str | None:
        raw = self._extract_regex(rule, text)
        if not raw:
            return None
        # Try common German date formats
        formats = [rule.date_format] if rule.date_format else [
            "%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y"
        ]
        for fmt in formats:
            try:
                return date.strftime(date.strptime(raw, fmt), "%Y-%m-%d")
            except ValueError:
                continue
        return raw  # return raw string if parsing fails

    def _extract_decimal(self, rule: ExtractionRule, text: str) -> float | None:
        raw = self._extract_regex(rule, text)
        if not raw:
            return None
        # Normalize German number format: 1.234,56 → 1234.56
        cleaned = raw.replace(".", "").replace(",", ".")
        try:
            return float(Decimal(cleaned))
        except InvalidOperation:
            return None

    def _extract_list(self, rule: ExtractionRule, text: str) -> list[str]:
        if not rule.regex:
            return []
        return re.findall(rule.regex, text, re.IGNORECASE | re.MULTILINE)

    def _extract_table(
        self, rule: ExtractionRule, doc: DocumentContent
    ) -> list[dict[str, str]]:
        """Find a table by its anchor regex and extract rows until stop_regex."""
        columns = rule.columns or []

        # Strategy 1: try pdfplumber-extracted tables first
        for page in doc.pages:
            for table in page.tables:
                if not table:
                    continue
                header_text = " ".join(str(c) for c in (table[0] or []))
                if rule.anchor_regex and not re.search(
                    rule.anchor_regex, header_text, re.IGNORECASE
                ):
                    continue
                rows = []
                for row in table[1:]:
                    if rule.stop_regex and any(
                        re.search(rule.stop_regex, str(c), re.IGNORECASE) for c in row
                    ):
                        break
                    if any(c for c in row):
                        row_dict = {
                            col: (row[i] if i < len(row) else "")
                            for i, col in enumerate(columns)
                        }
                        rows.append(row_dict)
                if rows:
                    return rows

        # Strategy 2: fallback to regex-based line scanning
        if rule.anchor_regex and rule.stop_regex:
            return self._extract_table_from_text(
                doc.full_text, rule.anchor_regex, rule.stop_regex, columns
            )
        return []

    def _extract_table_from_text(
        self,
        text: str,
        anchor_regex: str,
        stop_regex: str,
        columns: list[str],
    ) -> list[dict[str, str]]:
        lines = text.split("\n")
        capturing = False
        rows = []
        for line in lines:
            if not capturing and re.search(anchor_regex, line, re.IGNORECASE):
                capturing = True
                continue
            if capturing:
                if re.search(stop_regex, line, re.IGNORECASE):
                    break
                parts = re.split(r"\s{2,}", line.strip())
                if parts and parts[0]:
                    row_dict = {
                        col: (parts[i] if i < len(parts) else "")
                        for i, col in enumerate(columns)
                    }
                    rows.append(row_dict)
        return rows
