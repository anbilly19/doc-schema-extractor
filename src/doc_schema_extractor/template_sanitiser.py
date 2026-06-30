"""Sanitise and validate LLM-generated template rules before saving.

Problems fixed here:
- Invalid Python regex (syntax error) -> rule dropped
- Regex with != 1 capture group -> rule dropped
- Regex that matches nothing against the source document -> rule flagged (kept but warned)
- Table rule missing anchor_regex or stop_regex -> rule dropped
- confidence_checks referencing fields not in extraction_rules -> check dropped
- Fingerprint keywords that are not present in source text -> dropped
- Back-fill with discriminative mined tokens only when LLM proposed too few valid keywords
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from .logging_utils import get_logger
from .models import ConfidenceCheck, ExtractionRule, Template

if TYPE_CHECKING:
    from .template_store import TemplateStore

logger = get_logger("template_sanitiser")

_MIN_KW_LEN = 5
_RE_TOKEN = re.compile(r"[A-Za-z\u00c0-\u024f][A-Za-z\u00c0-\u024f0-9]{%d,}" % (_MIN_KW_LEN - 1))

_STOP_WORDS = {
    "gmbh", "datum", "seite", "bitte", "sehr", "damen", "herren", "liefern",
    "artikel", "menge", "preis", "summe", "gesamt", "netto", "brutto",
    "bestellung", "lieferung", "rechnung", "auftrag", "position", "nummer",
    "anzahl", "stück", "euro", "their", "with", "from", "this", "that",
}


def _tokenise(text: str) -> list[str]:
    return [m.group(0) for m in _RE_TOKEN.finditer(text)]


def _score_keywords(
    candidates: list[str],
    source_text: str,
    existing_templates: list[Template],
) -> list[tuple[str, float]]:
    """Score candidate tokens by discriminativeness for back-fill use only.

    Score = frequency_in_source / (1 + number_of_existing_templates_that_contain_exact_keyword).
    """
    src_lower = source_text.lower()
    src_counter = Counter(t.lower() for t in candidates)

    scored: list[tuple[str, float]] = []
    seen_lower: set[str] = set()
    for token in candidates:
        tl = token.lower()
        if tl in seen_lower or tl in _STOP_WORDS:
            continue
        seen_lower.add(tl)
        freq = src_counter[tl]
        # Exact full-string match against existing keyword lists
        collision = sum(
            1 for tmpl in existing_templates
            if any(tl == kw.lower() for kw in tmpl.fingerprint.required_keywords)
        )
        score = freq / (1.0 + collision)
        if tl in src_lower:
            scored.append((token, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _filter_keywords(
    proposed: list[str],
    source_text: str,
    existing_templates: list[Template],
    target_count: int = 5,
) -> list[str]:
    """Validate and back-fill fingerprint keywords.

    Policy:
    - Keep ALL proposed keywords that actually appear in the source text.
      We do NOT drop them for cross-template collision — specificity is the
      LLM's job and proposed keywords are almost always more specific than
      anything we can mine automatically.
    - Only back-fill with mined tokens when fewer than target_count proposed
      keywords survived the source-text presence check.
    """
    src_lower = source_text.lower()

    kept: list[str] = []
    for kw in proposed:
        if kw.lower() in src_lower:
            kept.append(kw)
        else:
            logger.warning("Keyword '%s' not found in source text, dropping", kw)

    if len(kept) >= target_count:
        logger.debug("Keeping %s proposed keywords (all present in source)", len(kept))
        return kept

    # Back-fill only to reach target_count
    needed = target_count - len(kept)
    logger.info(
        "Only %s/%s proposed keywords present in source; mining %s back-fill tokens",
        len(kept), len(proposed), needed,
    )
    candidates = _tokenise(source_text)
    scored = _score_keywords(candidates, source_text, existing_templates)
    kept_lower = {k.lower() for k in kept}
    for token, score in scored:
        if token.lower() in kept_lower:
            continue
        kept.append(token)
        kept_lower.add(token.lower())
        logger.info("Back-fill keyword '%s' (score=%.2f)", token, score)
        needed -= 1
        if needed == 0:
            break

    logger.info(
        "Final keywords for template: %s (proposed=%s existing_templates=%s)",
        kept, len(proposed), len(existing_templates),
    )
    return kept


# ---------------------------------------------------------------------------
# Rule sanitisation
# ---------------------------------------------------------------------------

def _count_capture_groups(pattern: str) -> int:
    try:
        return re.compile(pattern).groups
    except re.error:
        return -1


def _is_valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def sanitise_rule(rule: ExtractionRule, source_text: str) -> ExtractionRule | None:
    if rule.type == "table":
        if not rule.anchor_regex or not rule.stop_regex:
            logger.warning(
                "Dropping table rule field=%s: missing anchor_regex or stop_regex", rule.field,
            )
            return None
        for label, pat in (("anchor_regex", rule.anchor_regex), ("stop_regex", rule.stop_regex)):
            if not _is_valid_regex(pat):
                logger.warning(
                    "Dropping table rule field=%s: invalid %s pattern=%s", rule.field, label, pat,
                )
                return None
        return rule

    if not rule.regex:
        logger.warning("Dropping rule field=%s type=%s: no regex", rule.field, rule.type)
        return None

    if not _is_valid_regex(rule.regex):
        logger.warning(
            "Dropping rule field=%s: invalid regex pattern=%s", rule.field, rule.regex,
        )
        return None

    n_groups = _count_capture_groups(rule.regex)
    if n_groups != 1:
        logger.warning(
            "Dropping rule field=%s: expected 1 capture group, got %s, pattern=%s",
            rule.field, n_groups, rule.regex,
        )
        return None

    try:
        match = re.search(rule.regex, source_text, re.IGNORECASE | re.MULTILINE)
        if not match:
            logger.warning(
                "Rule field=%s regex matches nothing in source text (kept) pattern=%s",
                rule.field, rule.regex,
            )
        else:
            logger.debug(
                "Rule field=%s smoke-test passed, extracted=%s", rule.field, match.group(1)[:100],
            )
    except Exception as exc:
        logger.warning("Rule field=%s smoke-test exception: %s", rule.field, exc)

    return rule


def sanitise_template(
    template: Template,
    source_text: str,
    store: "TemplateStore | None" = None,
) -> Template:
    """Sanitise all rules, confidence checks, and fingerprint keywords."""
    if store is not None:
        existing = [t for t in store.list_all() if t.template_id != template.template_id]
        template.fingerprint.required_keywords = _filter_keywords(
            proposed=template.fingerprint.required_keywords,
            source_text=source_text,
            existing_templates=existing,
        )

    original_count = len(template.extraction_rules)
    valid_rules: list[ExtractionRule] = []

    for rule in template.extraction_rules:
        sanitised = sanitise_rule(rule, source_text)
        if sanitised is not None:
            valid_rules.append(sanitised)

    dropped = original_count - len(valid_rules)
    if dropped:
        logger.warning(
            "Sanitiser dropped %s/%s rules for template_id=%s",
            dropped, original_count, template.template_id,
        )
    else:
        logger.info(
            "Sanitiser: all %s rules valid for template_id=%s", original_count, template.template_id,
        )

    valid_field_names = {r.field for r in valid_rules}

    valid_checks: list[ConfidenceCheck] = []
    for check in template.confidence_checks:
        if check.field in valid_field_names:
            valid_checks.append(check)
        else:
            logger.warning(
                "Dropping confidence_check for field=%s: not in valid rules", check.field,
            )

    template.extraction_rules = valid_rules
    template.confidence_checks = valid_checks
    return template
