"""Sanitise and validate LLM-generated template rules before saving.

Problems fixed here:
- Invalid Python regex (syntax error) -> rule dropped
- Regex with != 1 capture group -> rule dropped  
- Regex that matches nothing against the source document -> rule flagged (kept but warned)
- Table rule missing anchor_regex or stop_regex -> rule dropped
- confidence_checks referencing fields not in extraction_rules -> check dropped
"""

from __future__ import annotations

import re
from typing import Any

from .logging_utils import get_logger
from .models import ConfidenceCheck, ExtractionRule, Template

logger = get_logger("template_sanitiser")


def _count_capture_groups(pattern: str) -> int:
    """Count the number of capture groups in a regex pattern."""
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
        # Table rules must have anchor + stop regex; they must also be valid patterns
        if not rule.anchor_regex or not rule.stop_regex:
            logger.warning(
                "Dropping table rule field=%s: missing anchor_regex or stop_regex",
                rule.field,
            )
            return None
        for label, pat in (("anchor_regex", rule.anchor_regex), ("stop_regex", rule.stop_regex)):
            if not _is_valid_regex(pat):
                logger.warning(
                    "Dropping table rule field=%s: invalid %s pattern=%s",
                    rule.field, label, pat,
                )
                return None
        return rule

    # Non-table rules must have a regex
    if not rule.regex:
        logger.warning("Dropping rule field=%s type=%s: no regex", rule.field, rule.type)
        return None

    # Must be valid Python regex
    if not _is_valid_regex(rule.regex):
        logger.warning(
            "Dropping rule field=%s: invalid regex pattern=%s",
            rule.field, rule.regex,
        )
        return None

    # Must have exactly one capture group
    n_groups = _count_capture_groups(rule.regex)
    if n_groups != 1:
        logger.warning(
            "Dropping rule field=%s: expected 1 capture group, got %s, pattern=%s",
            rule.field, n_groups, rule.regex,
        )
        return None

    # Smoke-test against source text; warn but keep if no match (doc may vary)
    try:
        match = re.search(rule.regex, source_text, re.IGNORECASE | re.MULTILINE)
        if not match:
            logger.warning(
                "Rule field=%s regex matches nothing in source text (kept, may work on future docs) pattern=%s",
                rule.field, rule.regex,
            )
        else:
            logger.debug(
                "Rule field=%s smoke-test passed, extracted=%s",
                rule.field, match.group(1)[:100],
            )
    except Exception as e:
        logger.warning("Rule field=%s smoke-test exception: %s", rule.field, e)

    return rule


def sanitise_template(template: Template, source_text: str) -> Template:
    """Sanitise all rules and confidence checks in a template in-place."""
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
            "Sanitiser: all %s rules valid for template_id=%s",
            original_count, template.template_id,
        )

    valid_field_names = {r.field for r in valid_rules}

    # Drop confidence checks referencing fields that no longer exist
    valid_checks: list[ConfidenceCheck] = []
    for check in template.confidence_checks:
        if check.field in valid_field_names:
            valid_checks.append(check)
        else:
            logger.warning(
                "Dropping confidence_check for field=%s: not in valid rules",
                check.field,
            )

    template.extraction_rules = valid_rules
    template.confidence_checks = valid_checks
    return template
