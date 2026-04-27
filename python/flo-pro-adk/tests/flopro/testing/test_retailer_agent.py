"""Tests for RetailerAgent data generation and solve."""

from __future__ import annotations

import numpy as np
import pytest

from flo_pro_adk.core.counterparty.counterparty_agent import _var_metadata_registry
from flo_pro_adk.core.data.in_memory_data_loader import InMemoryDataLoader
from flo_pro_adk.flopro.counterparty.retailer_agent import RetailerAgent
from flo_pro_adk.flopro.testing.flopro_data_generator import FloProSimulationDataGenerator
from flo_pro_adk.flopro.testing.flopro_scenarios import BASE_SCENARIO
from tests.flopro.testing.conftest import requires_xpress


@pytest.fixture()
def gen() -> FloProSimulationDataGenerator:
    return FloProSimulationDataGenerator(BASE_SCENARIO)


def test_generates_valid_data(gen: FloProSimulationDataGenerator):
    assert len(gen.generate_counterparty_input_data().validate()) == 0


def test_reproducibility():
    d1 = FloProSimulationDataGenerator(BASE_SCENARIO).generate_counterparty_input_data()
    d2 = FloProSimulationDataGenerator(BASE_SCENARIO).generate_counterparty_input_data()
    np.testing.assert_array_equal(d1.demand, d2.demand)


@requires_xpress
def test_solve_returns_valid_solution(gen: FloProSimulationDataGenerator):
    _var_metadata_registry[RetailerAgent] = gen.generate_variable_group_metadata()
    agent = RetailerAgent(agent_params={}, data_loader=InMemoryDataLoader(gen.generate_counterparty_input_data()))
    solution = agent.solve(gen.generate_consensus_vars(), gen.generate_prices(), gen.generate_rho())

    for group_name, z in gen.generate_consensus_vars().items():
        po = solution.preferred_vars[group_name]
        assert po.shape == z.shape, f"Shape mismatch for {group_name}"
        assert np.all(np.isfinite(po)), f"Non-finite values in {group_name}"
        assert np.all(po >= -1e-6), f"Negative values in {group_name}"

    assert solution.objective.proximal >= 0


@requires_xpress
@pytest.mark.parametrize("rho_val", [0.1, 1.0, 10.0, 100.0])
def test_higher_rho_pulls_closer_to_consensus(gen: FloProSimulationDataGenerator, rho_val: float):
    _var_metadata_registry[RetailerAgent] = gen.generate_variable_group_metadata()
    agent = RetailerAgent(agent_params={}, data_loader=InMemoryDataLoader(gen.generate_counterparty_input_data()))
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
