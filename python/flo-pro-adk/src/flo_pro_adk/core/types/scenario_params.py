"""Scenario parameter types for test data generation.

ScenarioParams configures the SimulationDataGenerator. ScenarioDomainParams is the
extensible TypedDict base that domain packs inherit to add domain-specific keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


class ScenarioDomainParams(TypedDict, total=False):
    """Base for domain-specific scenario parameters.

    Domain packs extend via inheritance to add required keys.
    ``total=False`` allows the base to be empty while subclasses
    define their own required fields.
    """


@dataclass(frozen=True)
class ScenarioParams:
    """Immutable parameter set configuring a test scenario.

    Consumed by SimulationDataGenerator and build_problem.
    """

    name: str
    seed: int
    n_variables: int
    n_groups: int
    price_distribution: Literal["uniform", "normal", "lognormal"]
    price_range: tuple[float, float]
    rho: float
    domain_params: ScenarioDomainParams

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ScenarioParams.name must not be empty")
        if self.n_variables <= 0:
            raise ValueError("ScenarioParams.n_variables must be positive")
        if self.n_groups <= 0:
            raise ValueError("ScenarioParams.n_groups must be positive")
        if self.price_range[0] > self.price_range[1]:
            raise ValueError("price_range lower bound must not exceed upper bound")
        if self.rho <= 0:
            raise ValueError("ScenarioParams.rho must be positive")
