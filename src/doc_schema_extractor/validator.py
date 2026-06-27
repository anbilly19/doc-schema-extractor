"""Validation of extracted data against template confidence checks."""

from __future__ import annotations

import re
from typing import Any

from .logging_utils import get_logger
from .models import Template

logger = get_logger("validator")


class Validator:
    def validate(self, data: dict[str, Any], template: Template) -> tuple[bool, list[str]]:
        errors: list[str] = []
        logger.info("Running validation template_id=%s checks=%s", template.template_id, len(template.confidence_checks))

        for check in template.confidence_checks:
            field = check.field
            value = data.get(field)

            if check.not_null and (value is None or value == ""):
                msg = f"Required field '{field}' is null or empty"
                errors.append(msg)
                logger.warning(msg)
                continue

            if value is None:
                continue

            if check.gt is not None:
                try:
                    if float(value) <= check.gt:
                        msg = f"Field '{field}' value {value} is not > {check.gt}"
                        errors.append(msg)
                        logger.warning(msg)
                except (TypeError, ValueError):
                    msg = f"Field '{field}' is not numeric for gt check"
                    errors.append(msg)
                    logger.warning(msg)

            if check.lt is not None:
                try:
                    if float(value) >= check.lt:
                        msg = f"Field '{field}' value {value} is not < {check.lt}"
                        errors.append(msg)
                        logger.warning(msg)
                except (TypeError, ValueError):
                    msg = f"Field '{field}' is not numeric for lt check"
                    errors.append(msg)
                    logger.warning(msg)

            if check.regex_match is not None and not re.match(check.regex_match, str(value)):
                msg = f"Field '{field}' value '{value}' does not match pattern '{check.regex_match}'"
                errors.append(msg)
                logger.warning(msg)

        logger.info("Validation complete template_id=%s passed=%s error_count=%s", template.template_id, len(errors) == 0, len(errors))
        return len(errors) == 0, errors
