"""Compute resource specification for agents and coordinators."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ComputeSpec:
    """Specification for compute resources required by an agent or coordinator."""

    num_cpus: Optional[float] = None
    num_gpus: Optional[float] = None
    memory_mb: Optional[int] = None
