"""Unit test runners — entry points for 1-iteration directional correctness tests.

Calls agent.solve() directly with generated (consensus_plan, prices, rho)
inputs to validate core ADMM functionality without a coordination loop.
No counterparty or coordinator involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from flo_pro_sdk.agent.agent_definition import (
    AgentDefinition,
    Objective,
    Solution,
)
from flo_pro_sdk.core.types import JsonValue
from flo_pro_sdk.core.variables import Prices, PublicVarValues, RhoValues

from flo_pro_adk.core.types.scenario_params import (
    ScenarioParams,
)

if TYPE_CHECKING:
    from flo_pro_adk.core.testing.simulation_data_generator import (
        SimulationDataGenerator,
    )


@dataclass(frozen=True)
class UnitTestResult:
    """Structured output from a single-iteration solve() call."""

    solution: Solution
    consensus_vars: PublicVarValues
    prices: Prices
    rho: RhoValues

    @property
    def objective(self) -> Objective:
        return self.solution.objective

    @property
    def preferred_vars(self) -> PublicVarValues:
        return self.solution.preferred_vars


def run_unit_test(
    agent_class: type[AgentDefinition],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
) -> UnitTestResult:
    """Run a single solve() call with generated inputs."""
    generator = data_generator_class(scenario)
    consensus_vars = generator.generate_consensus_vars()
    prices = generator.generate_prices()
    rho = generator.generate_rho()

    agent = agent_class.create(agent_params or {})
    solution = agent.solve(consensus_vars, prices, rho)

    return UnitTestResult(
        solution=solution,
        consensus_vars=consensus_vars,
        prices=prices,
        rho=rho,
    )


def run_unit_test_with_inputs(
    agent_class: type[AgentDefinition],
    consensus_vars: PublicVarValues,
    prices: Prices,
    rho: RhoValues,
    *,
    agent_params: dict[str, JsonValue] | None = None,
) -> UnitTestResult:
    """Run a single solve() call with explicit inputs."""
    agent = agent_class.create(agent_params or {})
    solution = agent.solve(consensus_vars, prices, rho)

    return UnitTestResult(
        solution=solution,
        consensus_vars=consensus_vars,
        prices=prices,
        rho=rho,
    )


def run_rho_sensitivity(
    agent_class: type[AgentDefinition],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
    n_points: int = 10,
) -> list[UnitTestResult]:
    """Run solve() across a range of rho values for sensitivity testing."""
    generator = data_generator_class(scenario)
    consensus_vars = generator.generate_consensus_vars()
    prices = generator.generate_prices()
    rho_series = generator.generate_rho_series(n_points)

    agent = agent_class.create(agent_params or {})

    return [
        UnitTestResult(
            solution=agent.solve(consensus_vars, prices, rho),
            consensus_vars=consensus_vars,
            prices=prices,
            rho=rho,
        )
        for rho in rho_series
    ]


def run_price_sensitivity(
    agent_class: type[AgentDefinition],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
    n_variants: int = 5,
) -> list[UnitTestResult]:
    """Run solve() across price variants for directional testing."""
    generator = data_generator_class(scenario)
    consensus_vars = generator.generate_consensus_vars()
    rho = generator.generate_rho()
    price_variants = generator.generate_price_variants(n_variants)

    agent = agent_class.create(agent_params or {})

    return [
        UnitTestResult(
            solution=agent.solve(consensus_vars, prices, rho),
            consensus_vars=consensus_vars,
            prices=prices,
            rho=rho,
        )
        for prices in price_variants
    ]
