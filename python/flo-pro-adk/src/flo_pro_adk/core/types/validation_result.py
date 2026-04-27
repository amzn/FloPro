"""Validation result types for data validation.

Used by CounterpartyInputData.validate() and assertion error reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity level for validation findings."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validation check."""

    valid: bool
    severity: ValidationSeverity
    message: str
    field: str | None = None
