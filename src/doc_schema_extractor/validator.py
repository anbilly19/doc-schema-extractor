"""Validation of extracted data against template confidence checks."""

from __future__ import annotations

import re
from typing import Any

from .models import ConfidenceCheck, Template


class Validator:
    """Validate extracted data against template confidence checks."""

    def validate(
        self, data: dict[str, Any], template: Template
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []

        for check in template.confidence_checks:
            field = check.field
            value = data.get(field)

            if check.not_null and (value is None or value == ""):
                errors.append(f"Required field '{field}' is null or empty")
                continue

            if value is None:
                continue

            if check.gt is not None:
                try:
                    if float(value) <= check.gt:
                        errors.append(
                            f"Field '{field}' value {value} is not > {check.gt}"
                        )
                except (TypeError, ValueError):
                    errors.append(f"Field '{field}' is not numeric for gt check")

            if check.lt is not None:
                try:
                    if float(value) >= check.lt:
                        errors.append(
                            f"Field '{field}' value {value} is not < {check.lt}"
                        )
                except (TypeError, ValueError):
                    errors.append(f"Field '{field}' is not numeric for lt check")

            if check.regex_match is not None:
                if not re.match(check.regex_match, str(value)):
                    errors.append(
                        f"Field '{field}' value '{value}' does not match pattern '{check.regex_match}'"
                    )

        passed = len(errors) == 0
        return passed, errors
