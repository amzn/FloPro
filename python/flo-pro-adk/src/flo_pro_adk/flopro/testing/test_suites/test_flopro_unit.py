"""Pre-built Flo Pro unit test suite.

Tests cover 1-iteration directional correctness:
- Solution validity (shape, finiteness)
- Determinism (same inputs → same output)
- Rho sensitivity (L2 distance decreases with higher rho)

Also validates that the built-in counterparty agents (RetailerAgent,
VendorAgent) produce valid solutions for the same scenarios.

The agent_class fixture is provided by FloProSimulationSuite at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from flo_pro_adk.core.assertions.agent_assertions import (
    AgentAssertions,
)
from flo_pro_adk.core.counterparty.counterparty_agent import (
    _var_metadata_registry,
)
from flo_pro_adk.core.data.in_memory_data_loader import (
    InMemoryDataLoader,
)
from flo_pro_adk.core.testing.unit_test_runner import (
    run_rho_sensitivity,
    run_unit_test,
)
from flo_pro_adk.flopro.counterparty.retailer_agent import (
    RetailerAgent,
)
from flo_pro_adk.flopro.counterparty.vendor_agent import (
    VendorAgent,
)
from flo_pro_adk.flopro.testing.flopro_data_generator import (
    FloProSimulationDataGenerator,
)
from flo_pro_adk.flopro.testing.flopro_scenarios import (
    BASE_SCENARIO,
    DEMAND_SPIKE_SCENARIO,
    NORMAL_SCENARIO,
    SUPPLY_CONSTRAINED_SCENARIO,
)

assertions = AgentAssertions()


def _xpress_available() -> bool:
    try:
        import xpress  # noqa: F401
        return True
    except ImportError:
        return False


requires_xpress = pytest.mark.skipif(
    not _xpress_available(),
    reason="xpress not installed",
)


class TestSolutionValidity:
    """Validate that solve() returns structurally correct output."""

    def test_base_scenario(self, agent_class):
        result = run_unit_test(agent_class, FloProSimulationDataGenerator, BASE_SCENARIO)
        assertions.assert_solution_valid(result.solution, result.consensus_vars)

    def test_demand_spike_scenario(self, agent_class):
        result = run_unit_test(agent_class, FloProSimulationDataGenerator, DEMAND_SPIKE_SCENARIO)
        assertions.assert_solution_valid(result.solution, result.consensus_vars)

    def test_supply_constrained_scenario(self, agent_class):
        result = run_unit_test(agent_class, FloProSimulationDataGenerator, SUPPLY_CONSTRAINED_SCENARIO)
        assertions.assert_solution_valid(result.solution, result.consensus_vars)


class TestDeterminism:
    """Validate that solve() is deterministic."""

    def test_deterministic_output(self, agent_class):
        result = run_unit_test(agent_class, FloProSimulationDataGenerator, BASE_SCENARIO)
        agent = agent_class.create({})
        assertions.assert_deterministic(
            agent.solve,
            result.consensus_vars,
            result.prices,
            result.rho,
        )


class TestRhoSensitivity:
    """Validate that L2 distance to consensus decreases with higher rho."""

    def test_rho_sensitivity(self, agent_class):
        results = run_rho_sensitivity(
            agent_class, FloProSimulationDataGenerator, BASE_SCENARIO, n_points=5,
        )
        if len(results) >= 2:
            agent = agent_class.create({})
            assertions.assert_l2_distance_decreases(
                agent.solve,
                results[0].consensus_vars,
                results[0].prices,
                results[0].rho,
                results[-1].rho,
            )


@requires_xpress
class TestRetailerAgentSolve:
    """Validate RetailerAgent produces valid solutions standalone."""

    def test_base_scenario_valid(self):
        gen = FloProSimulationDataGenerator(BASE_SCENARIO)
        data = gen.generate_counterparty_input_data()
        consensus = gen.generate_consensus_vars()
        prices = gen.generate_prices()
        rho = gen.generate_rho()

        _var_metadata_registry[RetailerAgent] = gen.generate_variable_group_metadata()
        agent = RetailerAgent(agent_params={}, data_loader=InMemoryDataLoader(data))
        agent.register()
        solution = agent.solve(consensus, prices, rho)
        assertions.assert_solution_valid(solution, consensus)

    def test_demand_spike_valid(self):
        gen = FloProSimulationDataGenerator(DEMAND_SPIKE_SCENARIO)
        data = gen.generate_counterparty_input_data()
        consensus = gen.generate_consensus_vars()
        prices = gen.generate_prices()
        rho = gen.generate_rho()

        _var_metadata_registry[RetailerAgent] = gen.generate_variable_group_metadata()
        agent = RetailerAgent(agent_params={}, data_loader=InMemoryDataLoader(data))
        agent.register()
        solution = agent.solve(consensus, prices, rho)
        assertions.assert_solution_valid(solution, consensus)


@requires_xpress
class TestVendorAgentSolve:
    """Validate VendorAgent produces valid solutions standalone."""

    def test_normal_scenario_valid(self):
        gen = FloProSimulationDataGenerator(NORMAL_SCENARIO)
        data = gen.generate_vendor_input_data()
        consensus = gen.generate_consensus_vars()
        prices = gen.generate_prices()
        rho = gen.generate_rho()

        _var_metadata_registry[VendorAgent] = gen.generate_variable_group_metadata()
        agent = VendorAgent(agent_params={}, data_loader=InMemoryDataLoader(data))
        agent.register()
        solution = agent.solve(consensus, prices, rho)
        assertions.assert_solution_valid(solution, consensus)

    def test_demand_spike_valid(self):
        gen = FloProSimulationDataGenerator(DEMAND_SPIKE_SCENARIO)
        data = gen.generate_vendor_input_data()
        consensus = gen.generate_consensus_vars()
        prices = gen.generate_prices()
        rho = gen.generate_rho()

        _var_metadata_registry[VendorAgent] = gen.generate_variable_group_metadata()
        agent = VendorAgent(agent_params={}, data_loader=InMemoryDataLoader(data))
        agent.register()
        solution = agent.solve(consensus, prices, rho)
        assertions.assert_solution_valid(solution, consensus)

    def test_rho_sensitivity(self):
        gen = FloProSimulationDataGenerator(NORMAL_SCENARIO)
        data = gen.generate_vendor_input_data()
        consensus = gen.generate_consensus_vars()
        prices = gen.generate_prices()

        deviations = []
        for rho_val in [0.1, 1.0, 10.0, 100.0]:
            rho = {k: np.full_like(v, rho_val) for k, v in gen.generate_rho().items()}
            _var_metadata_registry[VendorAgent] = gen.generate_variable_group_metadata()
            agent = VendorAgent(agent_params={}, data_loader=InMemoryDataLoader(data))
            agent.register()
            sol = agent.solve(consensus, prices, rho)
            for group_name in consensus:
                dev = float(np.linalg.norm(
                    sol.preferred_vars[group_name] - consensus[group_name]
                ))
                deviations.append(dev)

        for i in range(1, len(deviations)):
            assert deviations[i] <= deviations[i - 1] + 1e-6
