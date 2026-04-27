"""Abstract base for counterparty input data.

Domain packs provide concrete implementations with domain-specific fields.
The ABC enforces a validate() contract so data integrity is checked before
injection into mock counterparty agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flo_pro_adk.core.types.validation_result import (
        ValidationResult,
    )


class CounterpartyInputData(ABC):
    """Base container for mock counterparty input data.

    Each domain pack defines a concrete subclass with domain-specific fields
    (e.g., demand arrays, cost parameters). ``validate()`` is called by the
    SimulationDataGenerator before injecting data into a InMemoryDataLoader.
    """

    @abstractmethod
    def validate(self) -> list[ValidationResult]:
        """Validate internal consistency of the input data.

        Returns a list of ValidationResult. An empty list means valid.
        """
        ...
