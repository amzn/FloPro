"""Run identity model for persistence backends.

Provides RunIdentity — a shared dataclass that holds the two-level identity
hierarchy (coordination_id + run_id) used by all persistence backends to
organize persisted data. Also provides manifest read/write utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _generate_timestamp_id(prefix: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}{ts}" if prefix else ts


@dataclass
class CoordinationRun:
    """Two-level identity for a persistence run.

    Attributes:
        coordination_id: Logical grouping for related runs (retries,
            parameter sweeps). Auto-generated if not provided.
        run_id: Unique identifier for this execution attempt.
            Auto-generated (timestamp-based) if not provided.
        resumed_from: If this run is a restart of a crashed run,
            the run_id of the crashed run.
    """

    coordination_id: str = field(
        default_factory=lambda: _generate_timestamp_id("coord_")
    )
    run_id: str = field(default_factory=lambda: _generate_timestamp_id())
    resumed_from: Optional[str] = None

    def run_dir(self, base_dir: Path) -> Path:
        """Compute the run directory: base_dir / coordination_id / run_id."""
        return base_dir / self.coordination_id / self.run_id


def write_manifest(
    run_dir: Path,
    identity: CoordinationRun,
    status: str,
    started_at: str,
    *,
    completed_at: Optional[str] = None,
    final_iteration: Optional[int] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Write manifest.json to the run directory."""
    manifest: Dict[str, Any] = {
        "coordination_id": identity.coordination_id,
        "run_id": identity.run_id,
        "status": status,
        "started_at": started_at,
    }
    optional = {
        "completed_at": completed_at,
        "final_iteration": final_iteration,
        "resumed_from": identity.resumed_from,
        "additional_attributes": additional_attributes,
    }
    manifest.update({k: v for k, v in optional.items() if v is not None})

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read manifest.json from a run directory. Returns None if not found."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)
