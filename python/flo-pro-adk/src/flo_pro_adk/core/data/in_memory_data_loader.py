"""InMemoryDataLoader — wraps pre-generated data for simulation and testing."""

from __future__ import annotations

from typing import Any

from flo_pro_adk.core.data.data_loader import DataLoader


class InMemoryDataLoader(DataLoader):
    """In-memory DataLoader for simulation and testing."""

    def __init__(self, data: Any) -> None:
        self._data = data

    def load(self) -> Any:
        return self._data

    def snapshot(self, run_id: str) -> None:
        """No-op — data is already in memory."""
