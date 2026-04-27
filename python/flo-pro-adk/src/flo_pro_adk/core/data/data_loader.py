"""DataLoader ABC — abstracts agent input data sourcing.

Implementations handle different tech stacks (local disk, S3, Snowflake, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataLoader(ABC):
    """Abstract base for agent input data loading.

    Implementations provide:
    - ``load()`` — read data from the source
    - ``snapshot(run_id)`` — persist a point-in-time copy for debugging
    """

    @abstractmethod
    def load(self) -> Any:
        """Read agent input data from the source."""
        ...

    @abstractmethod
    def snapshot(self, run_id: str) -> None:
        """Persist a point-in-time copy of the data for debugging."""
        ...
