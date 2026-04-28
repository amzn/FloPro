# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the pre-built Flo Pro unit test suite.

Tests the discovery-based suite logic with StubAgent (via conftest fixture)
and the counterparty agent tests with real RetailerAgent/VendorAgent.
"""

from __future__ import annotations

import numpy as np
import pytest

from flo_pro_adk.core.assertions.agent_assertions import AgentAssertions
from flo_pro_adk.core.counterparty.counterparty_agent import _var_metadata_registry
from flo_pro_adk.core.data.in_memory_data_loader import InMemoryDataLoader
from flo_pro_adk.core.testing.unit_test_runner import run_rho_sensitivity, run_unit_test
from flo_pro_adk.flopro.counterparty.retailer_agent import RetailerAgent
from flo_pro_adk.flopro.counterparty.vendor_agent import VendorAgent
from flo_pro_adk.flopro.testing.flopro_data_generator import FloProSimulationDataGenerator
from flo_pro_adk.flopro.testing.flopro_scenarios import (
    BASE_SCENARIO,
    DEMAND_SPIKE_SCENARIO,
    NORMAL_SCENARIO,
    SUPPLY_CONSTRAINED_SCENARIO,
)
from tests.flopro.testing.conftest import requires_xpress

assertions = AgentAssertions()


# -- Discovery-based tests (use StubAgent via conftest agent_class fixture) --


@pytest.mark.parametrize("scenario", [BASE_SCENARIO, DEMAND_SPIKE_SCENARIO, SUPPLY_CONSTRAINED_SCENARIO], ids=lambda s: s.name)
def test_solution_validity(agent_class, scenario):
    result = run_unit_test(agent_class, FloProSimulationDataGenerator, scenario)
    assertions.assert_solution_valid(result.solution, result.consensus_vars)


def test_determinism(agent_class):
    result = run_unit_test(agent_class, FloProSimulationDataGenerator, BASE_SCENARIO)
    agent = agent_class.create({})
    assertions.assert_deterministic(agent.solve, result.consensus_vars, result.prices, result.rho)


def test_rho_sensitivity(agent_class):
    results = run_rho_sensitivity(agent_class, FloProSimulationDataGenerator, BASE_SCENARIO, n_points=5)
    assert len(results) >= 2
    agent = agent_class.create({})
    assertions.assert_l2_distance_decreases(
        agent.solve, results[0].consensus_vars, results[0].prices, results[0].rho, results[-1].rho,
    )


# -- RetailerAgent standalone tests --


@requires_xpress
@pytest.mark.parametrize("scenario", [BASE_SCENARIO, DEMAND_SPIKE_SCENARIO], ids=lambda s: s.name)
def test_retailer_solve_valid(scenario):
    gen = FloProSimulationDataGenerator(scenario)
    _var_metadata_registry[RetailerAgent] = gen.generate_variable_group_metadata()
    agent = RetailerAgent(agent_params={}, data_loader=InMemoryDataLoader(gen.generate_counterparty_input_data()))
    agent.register()
    solution = agent.solve(gen.generate_consensus_vars(), gen.generate_prices(), gen.generate_rho())
    assertions.assert_solution_valid(solution, gen.generate_consensus_vars())


# -- VendorAgent standalone tests --


@requires_xpress
@pytest.mark.parametrize("scenario", [NORMAL_SCENARIO, DEMAND_SPIKE_SCENARIO], ids=lambda s: s.name)
def test_vendor_solve_valid(scenario):
    gen = FloProSimulationDataGenerator(scenario)
    _var_metadata_registry[VendorAgent] = gen.generate_variable_group_metadata()
    agent = VendorAgent(agent_params={}, data_loader=InMemoryDataLoader(gen.generate_vendor_input_data()))
    agent.register()
    solution = agent.solve(gen.generate_consensus_vars(), gen.generate_prices(), gen.generate_rho())
    assertions.assert_solution_valid(solution, gen.generate_consensus_vars())


@requires_xpress
@pytest.mark.parametrize("rho_val", [1.0, 10.0, 100.0])
def test_vendor_rho_sensitivity(rho_val):
    gen = FloProSimulationDataGenerator(NORMAL_SCENARIO)
    _var_metadata_registry[VendorAgent] = gen.generate_variable_group_metadata()
    agent = VendorAgent(agent_params={}, data_loader=InMemoryDataLoader(gen.generate_vendor_input_data()))
    agent.register()
    consensus = gen.generate_consensus_vars()
    prices = gen.generate_prices()

    rho_low = {k: np.full_like(v, 0.1) for k, v in gen.generate_rho().items()}
    rho_high = {k: np.full_like(v, rho_val) for k, v in gen.generate_rho().items()}

    sol_low = agent.solve(consensus, prices, rho_low)
    sol_high = agent.solve(consensus, prices, rho_high)

    for g in consensus:
        dist_low = float(np.linalg.norm(sol_low.preferred_vars[g] - consensus[g]))
        dist_high = float(np.linalg.norm(sol_high.preferred_vars[g] - consensus[g]))
        assert dist_high <= dist_low + 1e-6, f"rho={rho_val}: dist_high={dist_high} > dist_low={dist_low}"
