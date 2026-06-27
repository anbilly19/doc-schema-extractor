"""Tests for the deterministic rule engine."""

import pytest
from doc_schema_extractor.rule_engine import RuleEngine
from doc_schema_extractor.models import Template, Fingerprint, ExtractionRule
from doc_schema_extractor.text_extractor import DocumentContent, PageContent


def _make_doc(text: str) -> DocumentContent:
    return DocumentContent(
        path="test.pdf",
        file_type="pdf",
        full_text=text,
        pages=[PageContent(page_num=1, text=text, tables=[])],
    )


def test_extract_string():
    engine = RuleEngine()
    template = Template(
        template_id="test",
        fingerprint=Fingerprint(required_keywords=[]),
        extraction_rules=[
            ExtractionRule(
                field="order_number",
                type="string",
                regex=r"Bestellung:\s*(\d+)",
            )
        ],
    )
    doc = _make_doc("Bestellung: 8859\nDatum: 18.06.2026")
    result = engine.apply(template, doc)
    assert result["order_number"] == "8859"


def test_extract_date():
    engine = RuleEngine()
    template = Template(
        template_id="test",
        fingerprint=Fingerprint(required_keywords=[]),
        extraction_rules=[
            ExtractionRule(
                field="delivery_date",
                type="date",
                regex=r"Wunschtermin:\s*(\d{2}\.\d{2}\.\d{4})",
            )
        ],
    )
    doc = _make_doc("Wunschtermin: 24.06.2026")
    result = engine.apply(template, doc)
    assert result["delivery_date"] == "2026-06-24"


def test_extract_decimal_german_format():
    engine = RuleEngine()
    template = Template(
        template_id="test",
        fingerprint=Fingerprint(required_keywords=[]),
        extraction_rules=[
            ExtractionRule(
                field="total_gross",
                type="decimal",
                regex=r"Summe\s+([\d\.\,]+)\s+EUR",
            )
        ],
    )
    doc = _make_doc("Summe 1.344,60 EUR")
    result = engine.apply(template, doc)
    assert result["total_gross"] == pytest.approx(1344.60)


def test_missing_field_returns_none():
    engine = RuleEngine()
    template = Template(
        template_id="test",
        fingerprint=Fingerprint(required_keywords=[]),
        extraction_rules=[
            ExtractionRule(
                field="missing_field",
                type="string",
                regex=r"NOTHERE:\s*(\w+)",
            )
        ],
    )
    doc = _make_doc("Completely different text")
    result = engine.apply(template, doc)
    assert result["missing_field"] is None
