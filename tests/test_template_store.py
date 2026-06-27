"""Tests for template store matching logic."""

import pytest
from doc_schema_extractor.template_store import TemplateStore
from doc_schema_extractor.models import Template, Fingerprint, ExtractionRule


def _make_template(tid: str, keywords: list[str], supplier: str) -> Template:
    return Template(
        template_id=tid,
        fingerprint=Fingerprint(
            required_keywords=keywords,
            supplier_hint=supplier,
            doc_type="order_confirmation",
        ),
        extraction_rules=[
            ExtractionRule(field="order_number", type="string", regex=r"Bestellung:\s*(\d+)")
        ],
    )


def test_match_hit(tmp_path):
    store = TemplateStore(tmp_path / "store.json")
    t = _make_template(
        "redefine_meat_v1",
        ["Bestellung:", "Nettopreis", "Nettowert", "Umsatzsteuer"],
        "Redefine Meat",
    )
    store.add(t)

    text = "Bestellung: 8859\nNettopreis 10,96 EUR\nNettowert 526,08 EUR\nUmsatzsteuer 7,00%"
    matched, score = store.match(text, threshold=0.75)
    assert matched is not None
    assert matched.template_id == "redefine_meat_v1"
    assert score >= 0.75


def test_match_miss(tmp_path):
    store = TemplateStore(tmp_path / "store.json")
    t = _make_template(
        "redefine_meat_v1",
        ["Bestellung:", "Nettopreis", "Nettowert", "Umsatzsteuer"],
        "Redefine Meat",
    )
    store.add(t)

    text = "Completely different document with no matching keywords"
    matched, score = store.match(text, threshold=0.75)
    assert matched is None
    assert score < 0.75


def test_persist_and_reload(tmp_path):
    store_path = tmp_path / "store.json"
    store = TemplateStore(store_path)
    t = _make_template("test_v1", ["keyword1", "keyword2"], "Test Supplier")
    store.add(t)

    # Reload from disk
    store2 = TemplateStore(store_path)
    loaded = store2.get("test_v1")
    assert loaded is not None
    assert loaded.fingerprint.supplier_hint == "Test Supplier"


def test_hit_counter(tmp_path):
    store = TemplateStore(tmp_path / "store.json")
    t = _make_template("counter_v1", ["keyword1", "keyword2"], "Supplier")
    store.add(t)

    for _ in range(3):
        loaded = store.get("counter_v1")
        loaded.increment_hit()
        store.add(loaded)

    reloaded = store.get("counter_v1")
    assert reloaded.hit_count == 3
