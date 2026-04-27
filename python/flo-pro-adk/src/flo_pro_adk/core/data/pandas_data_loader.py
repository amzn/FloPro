"""PandasDataLoader — read CSV into a DataFrame with auto-snapshot.

Usage::

    loader = PandasDataLoader("my_costs.csv")
    data = loader.load()  # reads CSV + snapshots for debugging
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd

from flo_pro_adk.core.data.data_loader import DataLoader

logger = logging.getLogger(__name__)

_DEFAULT_SNAPSHOT_DIR = Path.home() / ".vadk" / "snapshots"


class PandasDataLoader(DataLoader):
    """Read a CSV file into a DataFrame with auto-snapshot on load().

    Args:
        path: Path to a CSV file.
        snapshot_dir: Override snapshot directory. Defaults to ~/.vadk/snapshots/.
    """

    def __init__(self, path: str | Path, snapshot_dir: Path | None = None) -> None:
        self._path = Path(path)
        self._snapshot_dir = snapshot_dir or _DEFAULT_SNAPSHOT_DIR
        self._data: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Read CSV on first call, auto-snapshot, cache for subsequent calls."""
        if self._data is None:
            self._data = pd.read_csv(self._path)
            self.snapshot(f"{self._path.stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        return self._data

    def snapshot(self, run_id: str) -> None:
        """Write CSV snapshot for debugging."""
        if self._data is None:
            return
        output_dir = self._snapshot_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self._data.to_csv(output_dir / "data.csv", index=False)
        logger.info("Snapshot: %s/%s/data.csv", self._snapshot_dir, run_id)
