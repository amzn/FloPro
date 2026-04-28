"""Pre-built Flo Pro E2E test suite.

Tests run full multi-iteration ADMM coordination with counterparty agents
(RetailerAgent and VendorAgent) and validate convergence.

The agent_class fixture is provided by FloProSimulationSuite at runtime.

NOTE: These tests require counterparty solve() to be implemented.
They will be skipped if the agent is not yet available.
"""

from __future__ import annotations

import pytest

from flo_pro_adk.core.assertions.coordination_assertions import (
    CoordinationAssertions,
)
from flo_pro_adk.core.testing.e2e_test_runner import (
    run_e2e_test,
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
    NON_CONVERGENCE_SCENARIO,
    NORMAL_SCENARIO,
    SUPPLY_CONSTRAINED_SCENARIO,
)

assertions = CoordinationAssertions()


def _retailer_agent_available() -> bool:
    """Check if RetailerAgent.solve() is implemented."""
    try:
        import numpy as np
        from flo_pro_sdk.core.variables import PublicVarGroupName
        group = PublicVarGroupName("asin_vendor_inbound_periods")
        agent = RetailerAgent(agent_params={})
        agent.solve(
            {group: np.zeros(1)},
            {group: np.zeros(1)},
            {group: np.ones(1)},
        )
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True


def _vendor_agent_available() -> bool:
    """Check if VendorAgent.solve() is implemented."""
    try:
        import numpy as np
        from flo_pro_sdk.core.variables import PublicVarGroupName
        group = PublicVarGroupName("asin_vendor_inbound_periods")
        agent = VendorAgent(agent_params={})
        agent.solve(
            {group: np.zeros(1)},
            {group: np.zeros(1)},
            {group: np.ones(1)},
        )
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True


requires_retailer_agent = pytest.mark.skipif(
    not _retailer_agent_available(),
    reason="RetailerAgent.solve() not yet implemented",
)

requires_vendor_agent = pytest.mark.skipif(
    not _vendor_agent_available(),
    reason="VendorAgent.solve() not yet implemented",
)


@requires_retailer_agent
class TestConvergenceWithRetailer:
    """Validate coordination convergence with RetailerAgent as counterparty."""

    def test_base_scenario_converges(self, agent_class):
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=RetailerAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=BASE_SCENARIO,
            max_iterations=500,
        )
        assertions.assert_convergence(result.final_state)

    def test_demand_spike_converges(self, agent_class):
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=RetailerAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=DEMAND_SPIKE_SCENARIO,
            max_iterations=500,
        )
        assertions.assert_convergence(result.final_state)

    def test_supply_constrained_converges(self, agent_class):
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=RetailerAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=SUPPLY_CONSTRAINED_SCENARIO,
            max_iterations=500,
        )
        assertions.assert_convergence(result.final_state)


@requires_vendor_agent
class TestConvergenceWithVendor:
    """Validate coordination convergence with VendorAgent as counterparty."""

    def test_normal_scenario_converges(self, agent_class):
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=VendorAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=NORMAL_SCENARIO,
            max_iterations=500,
        )
        assertions.assert_convergence(result.final_state)

    def test_demand_spike_converges(self, agent_class):
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=VendorAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=DEMAND_SPIKE_SCENARIO,
            max_iterations=500,
        )
        assertions.assert_convergence(result.final_state)

    def test_supply_constrained_converges(self, agent_class):
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=VendorAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=SUPPLY_CONSTRAINED_SCENARIO,
            max_iterations=500,
        )
        assertions.assert_convergence(result.final_state)


@requires_retailer_agent
class TestNonConvergenceWithRetailer:
    """Validate behavior under adversarial scenarios with RetailerAgent."""

    def test_non_convergence_scenario_runs(self, agent_class):
        """Should complete without error even if it doesn't converge."""
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=RetailerAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=NON_CONVERGENCE_SCENARIO,
            max_iterations=100,
        )
        assert result.n_iterations > 0


@requires_vendor_agent
class TestNonConvergenceWithVendor:
    """Validate behavior under adversarial scenarios with VendorAgent."""

    def test_non_convergence_scenario_runs(self, agent_class):
        """Should complete without error even if it doesn't converge."""
        result = run_e2e_test(
            agent_class=agent_class,
            counterparty_class=VendorAgent,
            data_generator_class=FloProSimulationDataGenerator,
            scenario=NON_CONVERGENCE_SCENARIO,
            max_iterations=100,
        )
        assert result.n_iterations > 0
