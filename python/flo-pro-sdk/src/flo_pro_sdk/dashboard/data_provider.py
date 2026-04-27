"""Read-only data provider for dashboard consumption.

DashboardDataProvider reads from the Parquet datasets written by
FileSystemBackend. Each read opens a fresh pyarrow.dataset scan, so
new L0/L1/L2 files are discovered automatically on every call.

The provider is stateless and never writes. It returns pandas DataFrames
for tabular data and numpy arrays for vector data. Missing data returns
empty DataFrames or None -- never raises exceptions for absent files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from flo_pro_sdk.core.persistence_backend import (
    AGENT_SOLUTIONS_DIR,
    CONSENSUS_VARS_DIR,
    CONVERGENCE_DIR,
    METADATA_DIR,
    VAR_METADATA_DIR,
    VAR_METADATA_FILE,
    _dedup_table,
)

logger = logging.getLogger(__name__)


class DashboardDataProvider:
    """Read-only provider that reads Parquet datasets for dashboard views.

    Args:
        run_dir: Path to a single run directory containing the Parquet
            datasets (convergence/, consensus_vars/, agent_solutions/,
            metadata/).
    """

    def __init__(self, run_dir: Path | str) -> None:
        self._run_dir = Path(run_dir)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def get_convergence_data(self) -> pd.DataFrame:
        """Read convergence scalars (iteration, timestamp, residuals).

        Returns an empty DataFrame if no convergence data exists.
        """
        conv_dir = self._run_dir / CONVERGENCE_DIR
        if not conv_dir.exists():
            return pd.DataFrame()
        try:
            dataset = ds.dataset(str(conv_dir), format="parquet")
            table = dataset.to_table()
            if table.num_rows == 0:
                return pd.DataFrame()
            table = _dedup_table(table.sort_by("iteration"))
            return table.to_pandas()
        except Exception:
            logger.exception("Error reading convergence data")
            return pd.DataFrame()

    def get_agent_solutions(
        self,
        agent_id: str | None = None,
        columns: List[str] | None = None,
    ) -> pd.DataFrame:
        """Read agent solutions with optional partition filter and column projection.

        Args:
            agent_id: If provided, only load this agent's partition.
            columns: If provided, only load these columns. The 'iteration'
                column is always included.

        Returns an empty DataFrame if no agent solution data exists.
        """
        sol_dir = self._run_dir / AGENT_SOLUTIONS_DIR
        if not sol_dir.exists():
            return pd.DataFrame()
        try:
            dataset = ds.dataset(str(sol_dir), format="parquet", partitioning="hive")

            row_filter = None
            if agent_id is not None:
                row_filter = pc.field("agent_id") == agent_id

            proj_columns = None
            if columns is not None:
                proj_columns = list(dict.fromkeys(["iteration"] + columns))
                if agent_id is None and "agent_id" not in proj_columns:
                    proj_columns.append("agent_id")

            table = dataset.to_table(filter=row_filter, columns=proj_columns)
            if table.num_rows == 0:
                return pd.DataFrame()
            table = table.sort_by("iteration")
            key_cols = (
                ("iteration", "agent_id")
                if "agent_id" in table.column_names
                else ("iteration",)
            )
            table = _dedup_table(table, key_columns=key_cols)
            return table.to_pandas()
        except Exception:
            logger.exception("Error reading agent solutions")
            return pd.DataFrame()

    def get_consensus_vars(self, iteration: int) -> Optional[np.ndarray]:
        """Read deserialized consensus variable array for a single iteration.

        Returns None if the iteration is not found or data is missing.
        """
        cv_dir = self._run_dir / CONSENSUS_VARS_DIR
        if not cv_dir.exists():
            return None
        try:
            dataset = ds.dataset(str(cv_dir), format="parquet")
            table = dataset.to_table(filter=pc.field("iteration") == iteration)
            if table.num_rows == 0:
                return None
            table = _dedup_table(table)
            row = table.to_pydict()
            return np.array(row["consensus_vars"][-1])
        except Exception:
            logger.exception("Error reading consensus vars for iteration %d", iteration)
            return None

    def get_all_consensus_vars(self) -> Dict[int, np.ndarray]:
        """Read all consensus variable arrays, keyed by iteration.

        Returns an empty dict if no consensus data exists. Single dataset
        scan — much more efficient than calling get_consensus_vars() in a loop.
        """
        cv_dir = self._run_dir / CONSENSUS_VARS_DIR
        if not cv_dir.exists():
            return {}
        try:
            dataset = ds.dataset(str(cv_dir), format="parquet")
            table = dataset.to_table()
            if table.num_rows == 0:
                return {}
            table = _dedup_table(table.sort_by("iteration"))
            data = table.to_pydict()
            return {
                it: np.array(cv)
                for it, cv in zip(data["iteration"], data["consensus_vars"])
            }
        except Exception:
            logger.exception("Error reading all consensus vars")
            return {}

    def get_problem_metadata(self) -> Optional[Dict[str, Any]]:
        """Read problem metadata (problem.json + var_metadata Parquet).

        Returns None if metadata has not been written yet.
        """
        problem_path = self._run_dir / METADATA_DIR / "problem.json"
        if not problem_path.exists():
            return None
        try:
            with open(problem_path) as f:
                metadata = json.load(f)

            vm_dir = self._run_dir / METADATA_DIR / VAR_METADATA_DIR
            var_metadata: Dict[str, Any] = {}
            if vm_dir.exists():
                for group_dir in sorted(vm_dir.iterdir()):
                    if group_dir.is_dir() and group_dir.name.startswith("group_name="):
                        group_name = group_dir.name.split("=", 1)[1]
                        data_path = group_dir / VAR_METADATA_FILE
                        if data_path.exists():
                            df = pq.read_table(data_path).to_pandas()
                            df = df.drop(columns=["group_name"], errors="ignore")
                            var_metadata[group_name] = df

            metadata["var_metadata"] = var_metadata
            return metadata
        except Exception:
            logger.exception("Error reading problem metadata")
            return None

    def get_manifest(self) -> Optional[Dict[str, Any]]:
        """Read the run manifest. Returns None if not found."""
        manifest_path = self._run_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception:
            logger.exception("Error reading manifest")
            return None

    def list_agent_ids(self) -> List[str]:
        """Discover agent IDs from the agent_solutions dataset.

        Returns an empty list if no agent solution data exists.
        """
        sol_dir = self._run_dir / AGENT_SOLUTIONS_DIR
        if not sol_dir.exists():
            return []
        try:
            dataset = ds.dataset(str(sol_dir), format="parquet", partitioning="hive")
            table = dataset.to_table(columns=["agent_id"])
            if table.num_rows == 0:
                return []
            return sorted(pc.unique(table.column("agent_id")).to_pylist())
        except Exception:
            logger.exception("Error listing agent IDs")
            return []
