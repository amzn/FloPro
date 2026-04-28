"""Persistence backends for durable state storage.

Provides the PersistenceBackend ABC and FileSystemBackend, which writes
algorithm state to a partitioned Parquet dataset layout with LSM-style
tiered compaction:

    base_dir/coordination_id/run_id/
        convergence/iter_NNNNNN.parquet          (L0 — per-iteration)
        convergence/chunk_NNNNNN.parquet         (L1 — compacted chunks)
        convergence/convergence.parquet          (L2 — final, on close)
        consensus_vars/...                       (same tiers)
        agent_solutions/agent_id=X/...           (same tiers, per partition)
        metadata/problem.json
        metadata/var_metadata/group_name=X/var_metadata.parquet
        manifest.json

Compaction tiers:
    L0: Per-iteration files, written immediately. Crash-safe.
    L1: Chunk files, produced by merging chunk_size L0 files.
    L2: Single file per dataset, produced on close().

See Also:
    - PersistenceWriter: Non-blocking background persistence
    - StoreConfig: Configuration for state store setup
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from flo_pro_sdk.core.state import AgentPlan, CoreState, State
    from flo_pro_sdk.core.coordination_run import CoordinationRun
    from flo_pro_sdk.core.registry import AgentRegistry

logger = logging.getLogger(__name__)

CONVERGENCE_DIR = "convergence"
CONSENSUS_VARS_DIR = "consensus_vars"
AGENT_SOLUTIONS_DIR = "agent_solutions"
AGENT_STATE_DIR = "agent_state"  # Reserved for per-agent prices/rho/targets (future)
METADATA_DIR = "metadata"
VAR_METADATA_DIR = "var_metadata"
DEFAULT_CHUNK_SIZE = 50

# ── Parquet file name patterns ─────────────────────────────────────────
# L0: Per-iteration files, written immediately.
L0_FILE_PATTERN = "iter_{iteration:06d}.parquet"
# L1: Chunk files, produced by merging chunk_size L0 files.
L1_FILE_PATTERN = "chunk_{index:06d}.parquet"
# L2: Final compacted file per dataset, produced on close().
L2_CONVERGENCE_FILE = "convergence.parquet"
L2_CONSENSUS_VARS_FILE = "consensus_vars.parquet"
L2_AGENT_SOLUTIONS_FILE = "agent_solutions.parquet"
# Metadata files
PROBLEM_METADATA_FILE = "problem.json"
VAR_METADATA_FILE = "var_metadata.parquet"


# ── Array serialization helpers ────────────────────────────────────────


def _array_to_list(arr: np.ndarray) -> list[float]:
    """Convert a numpy array to a list of float64 values for Parquet storage."""
    return arr.astype(np.float64).tolist()


def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
    """Write a Parquet file atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        os.close(fd)
        pq.write_table(table, tmp_path)
        tmp_path.rename(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _list_l0_files(directory: Path) -> list[Path]:
    """List Level 0 (iter_*.parquet) files sorted by name."""
    return sorted(directory.glob("iter_*.parquet"))


def _next_chunk_index(directory: Path) -> int:
    """Return the next chunk index for a directory."""
    existing = list(directory.glob("chunk_*.parquet"))
    if not existing:
        return 0
    indices = [int(f.stem.split("_", 1)[1]) for f in existing]
    return max(indices) + 1


COMPACTION_MARKER = ".compaction_done"


def _recover_incomplete_compaction(directory: Path, l2_filename: str) -> None:
    """Clean up source files left behind by a crash during compaction.

    If a compaction marker exists, the compacted file was written
    successfully but the process crashed before deleting the source
    files. Delete them now.
    """
    marker = directory / COMPACTION_MARKER
    if not marker.exists():
        return
    # L0→L1 leftovers
    for f in _list_l0_files(directory):
        f.unlink(missing_ok=True)
    # L1→L2 leftovers (only if L2 file exists — L2 compaction)
    if (directory / l2_filename).exists():
        for f in sorted(directory.glob("chunk_*.parquet")):
            f.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    logger.info("Recovered incomplete compaction in %s", directory)


def _dedup_table(
    table: pa.Table, key_columns: tuple[str, ...] = ("iteration",)
) -> pa.Table:
    """Remove duplicate rows by key columns, keeping the last occurrence.

    This is a safety net for the case where a crash left both compacted
    and source files in the same directory.
    """
    if table.num_rows == 0:
        return table
    if len(key_columns) == 1:
        keys = table.column(key_columns[0]).to_pylist()
    else:
        cols = [table.column(c).to_pylist() for c in key_columns]
        keys = list(zip(*cols))
    seen: dict = {}
    for idx, key in enumerate(keys):
        seen[key] = idx  # last occurrence wins
    if len(seen) == table.num_rows:
        return table  # no duplicates, fast path
    removed = table.num_rows - len(seen)
    logger.warning(
        "Dedup removed %d duplicate row(s) by %s — possible incomplete compaction recovery",
        removed,
        key_columns,
    )
    indices = sorted(seen.values())
    return table.take(indices)


def _compact_l0_to_l1(directory: Path) -> bool:
    """Compact all L0 files in directory into a single L1 chunk.

    Uses a marker file for crash safety: the marker is written after
    the chunk file, and source L0 files are deleted after the marker.
    If a crash occurs between writing the chunk and deleting L0 files,
    the marker enables cleanup on next startup.

    Returns True if compaction succeeded, False if it failed or
    there were no files to compact.
    """
    l0_files = _list_l0_files(directory)
    if not l0_files:
        return False
    try:
        tables = [pq.ParquetFile(f).read() for f in l0_files]
        merged = pa.concat_tables(tables)
        chunk_idx = _next_chunk_index(directory)
        chunk_path = directory / L1_FILE_PATTERN.format(index=chunk_idx)
        _atomic_write_parquet(merged, chunk_path)
        # Marker: compacted file is durable, safe to delete sources
        marker = directory / COMPACTION_MARKER
        marker.touch()
        for f in l0_files:
            f.unlink()
        marker.unlink(missing_ok=True)
        return True
    except Exception:
        logger.exception(
            "L0→L1 compaction failed for %s; retaining L0 files", directory
        )
        return False


def _compact_to_l2(directory: Path, l2_filename: str) -> bool:
    """Compact all L0 + L1 files into a single L2 file.

    Uses a marker file for crash safety (same pattern as L0→L1).

    Returns True if compaction succeeded, False otherwise.
    """
    l0_files = _list_l0_files(directory)
    l1_files = sorted(directory.glob("chunk_*.parquet"))
    all_files = l1_files + l0_files
    if not all_files:
        return False
    try:
        tables = [pq.ParquetFile(f).read() for f in all_files]
        merged = pa.concat_tables(tables).sort_by("iteration")
        data_path = directory / l2_filename
        _atomic_write_parquet(merged, data_path)
        # Marker: compacted file is durable, safe to delete sources
        marker = directory / COMPACTION_MARKER
        marker.touch()
        for f in all_files:
            f.unlink()
        marker.unlink(missing_ok=True)
        return True
    except Exception:
        logger.exception("L1→L2 compaction failed for %s; retaining files", directory)
        return False


# ── Abstract base class ───────────────────────────────────────────────


class PersistenceBackend(ABC):
    """Abstract base class for persistent state storage backends."""

    @abstractmethod
    def write_state(self, iteration: int, state: "State", timestamp: float) -> None:
        """Write algorithm state for an iteration.

        Args:
            iteration: The iteration number.
            state: The algorithm state.
            timestamp: UTC epoch timestamp of when the state was produced.
        """
        pass

    @abstractmethod
    def write_agent_plan(
        self, iteration: int, agent_id: str, plan: "AgentPlan"
    ) -> None:
        """Write an agent's plan for an iteration."""
        pass

    @abstractmethod
    def write_metadata(self, registry: "AgentRegistry") -> None:
        """Write static problem metadata after registration finalization."""
        pass

    @abstractmethod
    def read_state(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Read state data for a specific iteration."""
        pass

    @abstractmethod
    def read_agent_plans(self, iteration: int) -> Dict[str, Any]:
        """Read all agent plans for an iteration."""
        pass

    @abstractmethod
    def read_metadata(self) -> Optional[Dict[str, Any]]:
        """Read problem metadata. Returns None if not written yet."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending writes."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close resources and ensure all writes complete."""
        pass


# ── Filesystem backend ────────────────────────────────────────────────


class FileSystemBackend(PersistenceBackend):
    """Filesystem persistence with partitioned Parquet dataset layout.

    Uses LSM-style tiered compaction:

        run_dir/
            convergence/iter_NNNNNN.parquet          (L0 — per-iteration)
            convergence/chunk_NNNNNN.parquet         (L1 — compacted chunks)
            convergence/convergence.parquet           (L2 — final, on close)
            consensus_vars/...                        (same tiers)
            agent_solutions/agent_id=X/...            (same tiers, per partition)
            manifest.json

    L0 files are written per-iteration. After chunk_size L0 writes,
    they are merged into an L1 chunk. On close(), all files are merged
    into a single L2 file per dataset.
    """

    def __init__(
        self,
        base_dir: str,
        identity: Optional["CoordinationRun"] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        from flo_pro_sdk.core.coordination_run import CoordinationRun, write_manifest

        self._base_dir = Path(base_dir)
        self._identity = identity or CoordinationRun()
        self._run_dir = self._identity.run_dir(self._base_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._chunk_size = chunk_size
        # Track L0 write counts for compaction triggers.
        # Keys: directory paths (convergence, consensus_vars, agent partitions)
        self._l0_counts: Dict[Path, int] = {}

        for d in (CONVERGENCE_DIR, CONSENSUS_VARS_DIR, AGENT_SOLUTIONS_DIR):
            (self._run_dir / d).mkdir(parents=True, exist_ok=True)

        self._recover()

        write_manifest(
            self._run_dir,
            self._identity,
            status="running",
            started_at=self._started_at,
        )
        logger.info("FileSystemBackend initialized: %s", self._run_dir)

    # Map dataset directory names to their L2 file names.
    _L2_FILENAMES = {
        CONVERGENCE_DIR: L2_CONVERGENCE_FILE,
        CONSENSUS_VARS_DIR: L2_CONSENSUS_VARS_FILE,
    }

    def _recover(self) -> None:
        """Recover from incomplete compaction and restore L0 counts + iteration count."""
        # Recover from any incomplete compaction (crash between write and delete)
        for d in (CONVERGENCE_DIR, CONSENSUS_VARS_DIR):
            _recover_incomplete_compaction(self._run_dir / d, self._L2_FILENAMES[d])
        for agent_dir in self._agent_partition_dirs():
            _recover_incomplete_compaction(agent_dir, L2_AGENT_SOLUTIONS_FILE)

        # Recover L0 counts from any existing files (resume scenario)
        for d in (CONVERGENCE_DIR, CONSENSUS_VARS_DIR):
            d_path = self._run_dir / d
            existing = len(_list_l0_files(d_path))
            if existing > 0:
                self._l0_counts[d_path] = existing
        for agent_dir in self._agent_partition_dirs():
            existing = len(_list_l0_files(agent_dir))
            if existing > 0:
                self._l0_counts[agent_dir] = existing

    def _agent_partition_dirs(self) -> list[Path]:
        """List agent_id=X partition directories under agent_solutions/."""
        sol_dir = self._run_dir / AGENT_SOLUTIONS_DIR
        if not sol_dir.exists():
            return []
        return [
            d
            for d in sorted(sol_dir.iterdir())
            if d.is_dir() and d.name.startswith("agent_id=")
        ]

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def run_id(self) -> str:
        return self._identity.run_id

    @property
    def coordination_id(self) -> str:
        return self._identity.coordination_id

    @property
    def identity(self) -> "CoordinationRun":
        return self._identity

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _increment_l0(self, directory: Path) -> None:
        """Increment L0 count for a directory and trigger compaction if needed."""
        count = self._l0_counts.get(directory, 0) + 1
        if count >= self._chunk_size:
            if _compact_l0_to_l1(directory):
                self._l0_counts[directory] = 0
            else:
                self._l0_counts[directory] = count
        else:
            self._l0_counts[directory] = count

    # ── Writes ─────────────────────────────────────────────────────────

    def write_state(self, iteration: int, state: "State", timestamp: float) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to closed FileSystemBackend")

        core = state.get_core_state()

        # convergence — scalar residuals
        conv_dir = self._run_dir / CONVERGENCE_DIR
        conv_path = conv_dir / L0_FILE_PATTERN.format(iteration=iteration)
        _atomic_write_parquet(
            self._convergence_table(iteration, core, timestamp), conv_path
        )
        self._increment_l0(conv_dir)

        # consensus_vars — array as binary
        cv_dir = self._run_dir / CONSENSUS_VARS_DIR
        cv_path = cv_dir / L0_FILE_PATTERN.format(iteration=iteration)
        _atomic_write_parquet(self._consensus_vars_table(iteration, core), cv_path)
        self._increment_l0(cv_dir)

        # TODO: Write per-agent state (prices, rho, targets) to agent_state/agent_id=X/.
        # These are inputs to the agent solve and live on State, not AgentPlan.
        # Deferred — will be a separate dataset written from write_state() once
        # the interface is extended to support it.
        logger.debug("Wrote state for iteration %d", iteration)

    def write_agent_plan(
        self, iteration: int, agent_id: str, plan: "AgentPlan"
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to closed FileSystemBackend")

        agent_dir = self._run_dir / AGENT_SOLUTIONS_DIR / f"agent_id={agent_id}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        path = agent_dir / L0_FILE_PATTERN.format(iteration=iteration)
        _atomic_write_parquet(self._agent_solution_table(iteration, plan), path)
        self._increment_l0(agent_dir)
        logger.debug("Wrote plan for agent %s iteration %d", agent_id, iteration)

    def write_metadata(self, registry: "AgentRegistry") -> None:
        """Write static problem metadata after registration finalization.

        Writes:
            metadata/problem.json — agents, variable_groups, subscriptions
            metadata/var_metadata/group_name=X/var_metadata.parquet — per-group metadata
        """
        if self._closed:
            raise RuntimeError("Cannot write to closed FileSystemBackend")

        meta_dir = self._run_dir / METADATA_DIR
        meta_dir.mkdir(parents=True, exist_ok=True)

        layout = registry.get_layout()
        global_vars = registry.get_all_subscribed_vars()

        # Build problem.json
        agents = []
        for agent_id in registry.list_agents():
            agents.append(
                {
                    "agent_id": agent_id,
                    "metadata": registry.get_metadata(agent_id),
                }
            )

        variable_groups = []
        for group_name in sorted(layout.group_slices.keys()):
            s = layout.group_slices[group_name]
            variable_groups.append(
                {
                    "name": str(group_name),
                    "slice_start": s.start,
                    "slice_end": s.stop,
                    "count": s.stop - s.start,
                }
            )

        subscriptions: Dict[str, Dict[str, list]] = {}
        for group_name in sorted(layout.group_slices.keys()):
            agent_indices = registry.get_agent_indices_by_var_group(group_name)
            for agent_id, indices in agent_indices.items():
                subscriptions.setdefault(agent_id, {})[str(group_name)] = (
                    indices.tolist()
                )

        problem_json = {
            "agents": agents,
            "variable_groups": variable_groups,
            "total_variable_count": layout.total_size,
            "subscriptions": subscriptions,
        }

        problem_path = meta_dir / PROBLEM_METADATA_FILE
        fd, tmp = tempfile.mkstemp(dir=meta_dir, suffix=".tmp")
        tmp_path = Path(tmp)
        try:
            os.close(fd)
            tmp_path.write_text(json.dumps(problem_json, indent=2))
            tmp_path.rename(problem_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        # Write per-group var_metadata as Parquet
        vm_dir = meta_dir / VAR_METADATA_DIR
        for group_name, group_meta in global_vars.items():
            group_dir = vm_dir / f"group_name={group_name}"
            group_dir.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(group_meta.var_metadata, preserve_index=False)
            _atomic_write_parquet(table, group_dir / VAR_METADATA_FILE)

        logger.info("Wrote problem metadata to %s", meta_dir)

    def read_metadata(self) -> Optional[Dict[str, Any]]:
        """Read problem metadata. Returns None if not written yet."""
        problem_path = self._run_dir / METADATA_DIR / PROBLEM_METADATA_FILE
        if not problem_path.exists():
            return None

        with open(problem_path) as f:
            metadata = json.load(f)

        # Load var_metadata Parquet files
        vm_dir = self._run_dir / METADATA_DIR / VAR_METADATA_DIR
        var_metadata: Dict[str, Any] = {}
        if vm_dir.exists():
            for group_dir in sorted(vm_dir.iterdir()):
                if group_dir.is_dir() and group_dir.name.startswith("group_name="):
                    group_name = group_dir.name.split("=", 1)[1]
                    data_path = group_dir / VAR_METADATA_FILE
                    if data_path.exists():
                        df = pq.read_table(data_path).to_pandas()
                        # Drop Hive partition column if pyarrow inferred it
                        df = df.drop(columns=["group_name"], errors="ignore")
                        var_metadata[group_name] = df

        metadata["var_metadata"] = var_metadata
        return metadata

    # ── Reads (per-iteration) ──────────────────────────────────────────

    def read_state(self, iteration: int) -> Optional[Dict[str, Any]]:
        conv_dir = self._run_dir / CONVERGENCE_DIR
        if not conv_dir.exists() or not any(conv_dir.glob("*.parquet")):
            return None

        # Read from whatever tier files exist (L0, L1, L2)
        dataset = ds.dataset(str(conv_dir), format="parquet")
        table = dataset.to_table(filter=pc.field("iteration") == iteration)
        if table.num_rows == 0:
            return None
        # Dedup safety: keep last occurrence (consistent with _dedup_table)
        conv = table.to_pydict()
        result = {k: v[-1] for k, v in conv.items()}

        cv_dir = self._run_dir / CONSENSUS_VARS_DIR
        if cv_dir.exists() and any(cv_dir.glob("*.parquet")):
            cv_dataset = ds.dataset(str(cv_dir), format="parquet")
            cv_table = cv_dataset.to_table(filter=pc.field("iteration") == iteration)
            if cv_table.num_rows > 0:
                cv = cv_table.to_pydict()
                result["consensus_vars"] = np.array(cv["consensus_vars"][-1])

        return result

    def read_agent_plans(self, iteration: int) -> Dict[str, Any]:
        plans: Dict[str, Any] = {}
        sol_dir = self._run_dir / AGENT_SOLUTIONS_DIR
        if not sol_dir.exists():
            return plans

        for agent_dir in sorted(sol_dir.iterdir()):
            if not agent_dir.is_dir() or not agent_dir.name.startswith("agent_id="):
                continue
            agent_id = agent_dir.name.split("=", 1)[1]
            dataset = ds.dataset(str(agent_dir), format="parquet")
            table = dataset.to_table(filter=pc.field("iteration") == iteration)
            if table.num_rows > 0:
                row = table.to_pydict()
                plans[agent_id] = {k: v[-1] for k, v in row.items()}

        return plans

    # ── Reads (full dataset via pyarrow.dataset) ───────────────────────

    def read_convergence_dataset(self) -> Optional[pa.Table]:
        """Read full convergence dataset. Returns None if directory doesn't exist."""
        conv_dir = self._run_dir / CONVERGENCE_DIR
        if not conv_dir.exists():
            return None
        dataset = ds.dataset(str(conv_dir), format="parquet")
        table = dataset.to_table()
        if table.num_rows > 0:
            table = _dedup_table(table.sort_by("iteration"))
        return table

    def read_agent_solutions_dataset(
        self, agent_id: Optional[str] = None
    ) -> Optional[pa.Table]:
        """Read agent solutions with Hive partition discovery.

        If agent_id is provided, only that agent's partition is loaded.
        """
        sol_dir = self._run_dir / AGENT_SOLUTIONS_DIR
        if not sol_dir.exists():
            return None

        dataset = ds.dataset(str(sol_dir), format="parquet", partitioning="hive")
        if agent_id is not None:
            table = dataset.to_table(filter=pc.field("agent_id") == agent_id)
        else:
            table = dataset.to_table()
        if table.num_rows > 0:
            table = table.sort_by("iteration")
            key_cols = (
                ("iteration", "agent_id")
                if "agent_id" in table.column_names
                else ("iteration",)
            )
            table = _dedup_table(table, key_columns=key_cols)
        return table

    # ── Lifecycle ──────────────────────────────────────────────────────

    def _get_max_iteration(self) -> int:
        """Read the maximum iteration from the convergence dataset."""
        conv_dir = self._run_dir / CONVERGENCE_DIR
        if not any(conv_dir.glob("*.parquet")):
            return 0
        try:
            dataset = ds.dataset(str(conv_dir), format="parquet")
            table = dataset.to_table(columns=["iteration"])
            if table.num_rows > 0:
                return pc.max(table.column("iteration")).as_py()
        except Exception:
            logger.debug(
                "Could not read max iteration from %s", conv_dir, exc_info=True
            )
        return 0

    def flush(self) -> None:
        pass  # Writes are synchronous

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # L1→L2 final compaction for each dataset
        _compact_to_l2(self._run_dir / CONVERGENCE_DIR, L2_CONVERGENCE_FILE)
        _compact_to_l2(self._run_dir / CONSENSUS_VARS_DIR, L2_CONSENSUS_VARS_FILE)
        for agent_dir in self._agent_partition_dirs():
            _compact_to_l2(agent_dir, L2_AGENT_SOLUTIONS_FILE)

        from flo_pro_sdk.core.coordination_run import write_manifest

        write_manifest(
            self._run_dir,
            self._identity,
            status="completed",
            started_at=self._started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            final_iteration=self._get_max_iteration(),
        )
        logger.info("FileSystemBackend closed: %s", self._run_dir)

    # ── Table builders ─────────────────────────────────────────────────

    @staticmethod
    def _convergence_table(
        iteration: int, core: "CoreState", timestamp: float
    ) -> pa.Table:
        primal = float(core.residuals.primal) if core.residuals else None
        dual = float(core.residuals.dual) if core.residuals else None
        return pa.table(
            {
                "iteration": pa.array([iteration], type=pa.int64()),
                "timestamp": pa.array([float(timestamp)], type=pa.float64()),
                "primal_residual": pa.array([primal], type=pa.float64()),
                "dual_residual": pa.array([dual], type=pa.float64()),
            }
        )

    @staticmethod
    def _consensus_vars_table(iteration: int, core: "CoreState") -> pa.Table:
        return pa.table(
            {
                "iteration": pa.array([iteration], type=pa.int64()),
                "consensus_vars": pa.array(
                    [_array_to_list(core.consensus_vars)], type=pa.list_(pa.float64())
                ),
            }
        )

    @staticmethod
    def _agent_solution_table(iteration: int, plan: "AgentPlan") -> pa.Table:
        sol = plan.solution
        obj = sol.objective

        # Build a struct column where each field is a variable group.
        # Sorted by group name for deterministic field ordering (mirrors VarLayout).
        # The struct schema is self-describing: field names are group names,
        # visible in Parquet schema metadata without reading any rows.
        sorted_pv = dict(sorted(sol.preferred_vars.items()))
        struct_fields = [
            pa.field(str(name), pa.list_(pa.float64())) for name in sorted_pv.keys()
        ]
        struct_value = {
            str(name): _array_to_list(arr) for name, arr in sorted_pv.items()
        }

        return pa.table(
            {
                "iteration": pa.array([iteration], type=pa.int64()),
                "utility": pa.array([float(obj.utility)], type=pa.float64()),
                "subsidy": pa.array([float(obj.subsidy)], type=pa.float64()),
                "proximal": pa.array([float(obj.proximal)], type=pa.float64()),
                "preferred_vars": pa.array(
                    [struct_value], type=pa.struct(struct_fields)
                ),
            }
        )
