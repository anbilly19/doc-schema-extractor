"""Deterministic rule engine - applies saved extraction rules without LLM."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .logging_utils import get_logger
from .models import ExtractionRule, Template
from .text_extractor import DocumentContent

logger = get_logger("rule_engine")


class RuleEngine:
    def apply(self, template: Template, doc: DocumentContent) -> dict[str, Any]:
        logger.info("Applying rule engine template_id=%s field_count=%s", template.template_id, len(template.extraction_rules))
        result: dict[str, Any] = {}
        text = doc.full_text

        for rule in template.extraction_rules:
            try:
                value = self._apply_rule(rule, text, doc)
                result[rule.field] = value
                logger.debug("Rule applied field=%s type=%s extracted=%s", rule.field, rule.type, self._preview(value))
            except Exception as e:
                result[rule.field] = None
                result[f"{rule.field}.__error"] = str(e)
                logger.exception("Rule failed field=%s type=%s", rule.field, rule.type)

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
            logger.debug("Regex rule missing pattern field=%s", rule.field)
            return None
        match = re.search(rule.regex, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            logger.debug("Regex no match field=%s pattern=%s", rule.field, rule.regex)
            return None
        value = match.group(1).strip()
        if rule.strip_chars:
            value = value.strip(rule.strip_chars)
        return value

    def _extract_date(self, rule: ExtractionRule, text: str) -> str | None:
        raw = self._extract_regex(rule, text)
        if not raw:
            return None
        formats = [rule.date_format] if rule.date_format else ["%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        logger.warning("Date parse failed field=%s raw=%s", rule.field, raw)
        return raw

    def _extract_decimal(self, rule: ExtractionRule, text: str) -> float | None:
        raw = self._extract_regex(rule, text)
        if not raw:
            return None
        cleaned = raw.replace(".", "").replace(",", ".")
        try:
            return float(Decimal(cleaned))
        except InvalidOperation:
            logger.warning("Decimal parse failed field=%s raw=%s cleaned=%s", rule.field, raw, cleaned)
            return None

    def _extract_list(self, rule: ExtractionRule, text: str) -> list[str]:
        if not rule.regex:
            return []
        matches = re.findall(rule.regex, text, re.IGNORECASE | re.MULTILINE)
        logger.debug("List extracted field=%s count=%s", rule.field, len(matches))
        return matches

    def _extract_table(self, rule: ExtractionRule, doc: DocumentContent) -> list[dict[str, str]]:
        columns = rule.columns or []
        logger.debug("Table extraction start field=%s columns=%s", rule.field, columns)

        for page in doc.pages:
            for table in page.tables:
                if not table:
                    continue
                header_text = " ".join(str(c) for c in (table[0] or []))
                if rule.anchor_regex and not re.search(rule.anchor_regex, header_text, re.IGNORECASE):
                    continue
                rows = []
                for row in table[1:]:
                    if rule.stop_regex and any(re.search(rule.stop_regex, str(c), re.IGNORECASE) for c in row):
                        break
                    if any(c for c in row):
                        row_dict = {col: (row[i] if i < len(row) else "") for i, col in enumerate(columns)}
                        rows.append(row_dict)
                if rows:
                    logger.info("Table extracted from page tables field=%s rows=%s", rule.field, len(rows))
                    return rows

        if rule.anchor_regex and rule.stop_regex:
            rows = self._extract_table_from_text(doc.full_text, rule.anchor_regex, rule.stop_regex, columns)
            logger.info("Table extracted from text fallback field=%s rows=%s", rule.field, len(rows))
            return rows
        logger.warning("Table extraction failed field=%s no anchor/stop or no rows", rule.field)
        return []

    def _extract_table_from_text(self, text: str, anchor_regex: str, stop_regex: str, columns: list[str]) -> list[dict[str, str]]:
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
                    row_dict = {col: (parts[i] if i < len(parts) else "") for i, col in enumerate(columns)}
                    rows.append(row_dict)
        return rows

    def _preview(self, value: Any) -> str:
        text = str(value)
        return text[:300] + ("..." if len(text) > 300 else "")
