"""Discover and browse coordination runs under a given path.

RunBrowser scans a directory tree for run directories (identified by
manifest.json), groups them by coordination_id, and provides
DashboardDataProvider instances on demand.

Directory structure::

    base_dir/
      coord_A/
        run_1/manifest.json
        run_2/manifest.json
      coord_B/
        run_3/manifest.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider
from flo_pro_sdk.dashboard.metrics import DashboardMetricsComputer

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Summary of a single run, read from its manifest."""

    run_dir: Path
    coordination_id: str
    run_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    final_iteration: Optional[int] = None
    resumed_from: Optional[str] = None
    duration_seconds: Optional[float] = None

    @staticmethod
    def from_manifest(run_dir: Path) -> Optional["RunInfo"]:
        """Read manifest.json and build a RunInfo. Returns None on failure."""
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read manifest: %s", manifest_path)
            return None

        duration = None
        started = m.get("started_at")
        completed = m.get("completed_at")
        if started and completed:
            try:
                t0 = datetime.fromisoformat(str(started))
                t1 = datetime.fromisoformat(str(completed))
                duration = (t1 - t0).total_seconds()
            except (ValueError, TypeError):
                pass

        return RunInfo(
            run_dir=run_dir,
            coordination_id=m.get("coordination_id", run_dir.parent.name),
            run_id=m.get("run_id", run_dir.name),
            status=m.get("status", "unknown"),
            started_at=started,
            completed_at=completed,
            final_iteration=m.get("final_iteration"),
            resumed_from=m.get("resumed_from"),
            duration_seconds=duration,
        )


@dataclass
class CoordinationInfo:
    """A coordination_id and its runs."""

    coordination_id: str
    runs: List[RunInfo] = field(default_factory=list)

    @property
    def latest_run(self) -> Optional[RunInfo]:
        return self.runs[-1] if self.runs else None

    @property
    def total_iterations(self) -> int:
        return sum(r.final_iteration or 0 for r in self.runs)


class RunBrowser:
    """Discover and browse runs under a path.

    Handles three input cases:
    1. Path is a run directory (has manifest.json) → single run
    2. Path is a coordination directory (children are run dirs) → one coordination
    3. Path is a base directory (children are coordination dirs) → multiple coordinations

    ``is_single_run`` returns True only for Case 1 (user pointed directly
    at a run directory). Cases 2 and 3 always show the overview page even
    if only one run exists currently, since more may appear later.

    Call ``refresh()`` to re-scan the directory tree for new runs.

    Usage::

        browser = RunBrowser("/path/to/base_dir")
        for coord in browser.coordinations:
            for run in coord.runs:
                provider = browser.get_provider(run.run_dir)
    """

    # Discovery case constants
    _CASE_RUN_DIR = 1
    _CASE_COORD_DIR = 2
    _CASE_BASE_DIR = 3

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).resolve()
        self._coordinations: List[CoordinationInfo] = []
        self._providers: Dict[Path, DashboardDataProvider] = {}
        self._computers: Dict[Path, DashboardMetricsComputer] = {}
        self._case: int = self._CASE_BASE_DIR
        self._discover()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def coordinations(self) -> List[CoordinationInfo]:
        return self._coordinations

    @property
    def is_single_run(self) -> bool:
        """True only when the path pointed directly at a run directory."""
        return self._case == self._CASE_RUN_DIR

    @property
    def total_runs(self) -> int:
        return sum(len(c.runs) for c in self._coordinations)

    @property
    def all_runs(self) -> List[RunInfo]:
        """Flat list of all runs across all coordinations."""
        return [r for c in self._coordinations for r in c.runs]

    def refresh(self) -> None:
        """Re-scan the directory tree for new or updated runs.

        Preserves cached providers and computers. Only re-discovers
        for Cases 2 and 3 (coordination/base dirs); Case 1 (single
        run dir) is static.
        """
        if self._case == self._CASE_RUN_DIR:
            # Re-read manifest to pick up status changes
            info = RunInfo.from_manifest(self._path)
            if info is not None:
                self._coordinations = [
                    CoordinationInfo(coordination_id=info.coordination_id, runs=[info]),
                ]
            return
        self._discover()

    def get_provider(self, run_dir: Path) -> DashboardDataProvider:
        """Get or create a DashboardDataProvider for a run directory."""
        run_dir = run_dir.resolve()
        if run_dir not in self._providers:
            self._providers[run_dir] = DashboardDataProvider(run_dir)
        return self._providers[run_dir]

    def get_computer(self, run_dir: Path) -> DashboardMetricsComputer:
        """Get or create a DashboardMetricsComputer for a run directory."""
        run_dir = run_dir.resolve()
        if run_dir not in self._computers:
            provider = self.get_provider(run_dir)
            self._computers[run_dir] = DashboardMetricsComputer(provider)
        return self._computers[run_dir]

    def find_run(self, coordination_id: str, run_id: str) -> Optional[RunInfo]:
        """Look up a specific run by coordination_id and run_id."""
        for coord in self._coordinations:
            if coord.coordination_id == coordination_id:
                for run in coord.runs:
                    if run.run_id == run_id:
                        return run
        return None

    def _discover(self) -> None:
        """Scan the path and populate coordinations."""
        path = self._path

        # Case 1: path is itself a run dir
        info = RunInfo.from_manifest(path)
        if info is not None:
            self._case = self._CASE_RUN_DIR
            coord = CoordinationInfo(coordination_id=info.coordination_id, runs=[info])
            self._coordinations = [coord]
            return

        # Case 2: children are run dirs (path is a coordination dir)
        children = sorted(p for p in path.iterdir() if p.is_dir())
        run_infos = [RunInfo.from_manifest(c) for c in children]
        run_infos_valid = [r for r in run_infos if r is not None]
        if run_infos_valid:
            self._case = self._CASE_COORD_DIR
            coord_id = run_infos_valid[0].coordination_id
            coord = CoordinationInfo(coordination_id=coord_id, runs=run_infos_valid)
            self._coordinations = [coord]
            return

        # Case 3: children are coordination dirs (path is a base dir)
        self._case = self._CASE_BASE_DIR
        coords: Dict[str, CoordinationInfo] = {}
        for child in children:
            if not child.is_dir():
                continue
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir():
                    continue
                info = RunInfo.from_manifest(grandchild)
                if info is not None:
                    cid = info.coordination_id
                    if cid not in coords:
                        coords[cid] = CoordinationInfo(coordination_id=cid)
                    coords[cid].runs.append(info)

        self._coordinations = sorted(coords.values(), key=lambda c: c.coordination_id)
