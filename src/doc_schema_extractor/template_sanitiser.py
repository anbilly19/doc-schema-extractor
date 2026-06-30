"""Sanitise and validate LLM-generated template rules before saving.

Problems fixed here:
- Invalid Python regex (syntax error) -> rule dropped
- Regex with != 1 capture group -> rule dropped
- Regex that matches nothing against the source document -> rule flagged (kept but warned)
- Table rule missing anchor_regex or stop_regex -> rule dropped
- confidence_checks referencing fields not in extraction_rules -> check dropped
- Fingerprint keywords that collide with existing templates -> replaced with
  discriminative alternatives mined from the source text (TF-IDF-style scoring)
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

# Minimum token length and character requirement to be a candidate keyword
_MIN_KW_LEN = 5
_RE_TOKEN = re.compile(r"[A-Za-z\u00c0-\u024f][A-Za-z\u00c0-\u024f0-9]{%d,}" % (_MIN_KW_LEN - 1))

# Stop-words that are too generic to discriminate between templates
_STOP_WORDS = {
    "gmbh", "datum", "seite", "bitte", "sehr", "damen", "herren", "liefern",
    "artikel", "menge", "preis", "summe", "gesamt", "netto", "brutto",
    "bestellung", "lieferung", "rechnung", "auftrag", "position", "nummer",
    "anzahl", "stück", "euro", "their", "with", "from", "this", "that",
}


def _tokenise(text: str) -> list[str]:
    """Extract candidate keyword tokens from text."""
    return [m.group(0) for m in _RE_TOKEN.finditer(text)]


def _score_keywords(
    candidates: list[str],
    source_text: str,
    existing_templates: list[Template],
) -> list[tuple[str, float]]:
    """Score candidate keywords by discriminativeness.

    Score = (frequency in source_text) / (1 + count of existing templates
    whose keywords contain this token case-insensitively).
    Higher score = appears often here, rarely in other templates.
    """
    src_lower = source_text.lower()
    src_counter = Counter(t.lower() for t in candidates)

    # Build a set of all tokens already used across existing templates
    existing_kw_tokens: set[str] = set()
    for tmpl in existing_templates:
        for kw in tmpl.fingerprint.required_keywords:
            existing_kw_tokens.update(t.lower() for t in _tokenise(kw))

    scored: list[tuple[str, float]] = []
    seen_lower: set[str] = set()
    for token in candidates:
        tl = token.lower()
        if tl in seen_lower or tl in _STOP_WORDS:
            continue
        seen_lower.add(tl)
        freq = src_counter[tl]
        collision = sum(
            1 for tmpl in existing_templates
            if any(tl in kw.lower() for kw in tmpl.fingerprint.required_keywords)
        )
        score = freq / (1.0 + collision)
        if tl in src_lower:  # must actually appear in source
            scored.append((token, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _filter_keywords(
    proposed: list[str],
    source_text: str,
    existing_templates: list[Template],
    target_count: int = 5,
) -> list[str]:
    """Return discriminative keywords for this template.

    Keeps proposed keywords that are not over-represented in existing templates,
    then back-fills with high-scoring tokens mined from source_text.
    """
    src_lower = source_text.lower()

    # Keep proposed keywords that are present in source and low-collision
    kept: list[str] = []
    for kw in proposed:
        if kw.lower() not in src_lower:
            logger.warning("Keyword '%s' not found in source text, dropping", kw)
            continue
        collision = sum(
            1 for tmpl in existing_templates
            if any(kw.lower() in existing_kw.lower() for existing_kw in tmpl.fingerprint.required_keywords)
        )
        if collision == 0:
            kept.append(kw)
        else:
            logger.info(
                "Keyword '%s' collides with %s existing template(s), will try to replace",
                kw, collision,
            )

    if len(kept) >= target_count:
        logger.debug("All %s proposed keywords are discriminative", len(kept))
        return kept[:target_count]

    # Back-fill with high-scoring mined tokens
    candidates = _tokenise(source_text)
    scored = _score_keywords(candidates, source_text, existing_templates)
    kept_lower = {k.lower() for k in kept}
    for token, score in scored:
        if token.lower() in kept_lower:
            continue
        kept.append(token)
        kept_lower.add(token.lower())
        logger.info("Added discriminative keyword '%s' (score=%.2f)", token, score)
        if len(kept) >= target_count:
            break

    logger.info(
        "Final keywords for new template: %s (from %s proposed, %s existing templates)",
        kept, len(proposed), len(existing_templates),
    )
    return kept


# ---------------------------------------------------------------------------
# Rule sanitisation (unchanged logic)
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
    """Validate one extraction rule. Returns None if rule should be dropped."""

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
    """Sanitise all rules, confidence checks, and fingerprint keywords.

    Pass `store` to enable discriminative keyword filtering; without it,
    keyword sanitisation is skipped (backward compatible).
    """
    # --- keyword filtering ---
    if store is not None:
        existing = [t for t in store.list_all() if t.template_id != template.template_id]
        template.fingerprint.required_keywords = _filter_keywords(
            proposed=template.fingerprint.required_keywords,
            source_text=source_text,
            existing_templates=existing,
        )

    # --- rule sanitisation ---
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
