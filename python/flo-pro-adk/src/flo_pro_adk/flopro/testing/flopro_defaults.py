"""Centralized default configuration for Flo Pro test scenarios.

All tunable constants for scenario generation live here. No magic numbers
should appear inline in scenario definitions or data generators.
"""

from __future__ import annotations

from typing import TypedDict


class FloProScenarioDefaults(TypedDict):
    """Default values for Flo Pro scenario construction."""

    base_seed: int
    n_variables: int
    n_groups: int
    price_range: tuple[float, float]
    rho: float
    n_asins: int
    n_inbound_nodes: int
    n_weeks: int


FLOPRO_SCENARIO_DEFAULTS: FloProScenarioDefaults = {
    "base_seed": 42,
    "n_variables": 18,
    "n_groups": 1,
    "price_range": (0.0, 10.0),
    "rho": 1.0,
    "n_asins": 3,
    "n_inbound_nodes": 2,
    "n_weeks": 3,
}
