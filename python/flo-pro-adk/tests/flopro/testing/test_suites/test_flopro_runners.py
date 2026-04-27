# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Flo Pro-specific test runner invocations."""

from __future__ import annotations

from flo_pro_adk.core.testing.e2e_test_runner import run_e2e_test
from flo_pro_adk.core.testing.unit_test_runner import run_rho_sensitivity, run_unit_test
from flo_pro_adk.flopro.counterparty.retailer_agent import RetailerAgent
from flo_pro_adk.flopro.counterparty.vendor_agent import VendorAgent
from flo_pro_adk.flopro.registration import FLOPRO_GROUP_NAME
from flo_pro_adk.flopro.testing.flopro_data_generator import FloProSimulationDataGenerator
from flo_pro_adk.flopro.testing.flopro_scenarios import BASE_SCENARIO, NORMAL_SCENARIO
from tests.conftest import StubAgent
from tests.flopro.testing.conftest import requires_xpress


def test_unit_runner_returns_valid_result():
    result = run_unit_test(StubAgent, FloProSimulationDataGenerator, BASE_SCENARIO)
    assert FLOPRO_GROUP_NAME in result.solution.preferred_vars
    assert result.consensus_vars[FLOPRO_GROUP_NAME].shape == result.preferred_vars[FLOPRO_GROUP_NAME].shape


def test_rho_sensitivity_returns_multiple():
    results = run_rho_sensitivity(StubAgent, FloProSimulationDataGenerator, BASE_SCENARIO, n_points=3)
    assert len(results) == 3


@requires_xpress
def test_e2e_runner_with_retailer():
    result = run_e2e_test(
        agent_class=StubAgent,
        counterparty_class=RetailerAgent,
        data_generator_class=FloProSimulationDataGenerator,
        scenario=BASE_SCENARIO,
        max_iterations=50,
    )
    assert result.n_iterations > 0


@requires_xpress
def test_e2e_runner_with_vendor():
    result = run_e2e_test(
        agent_class=StubAgent,
        counterparty_class=VendorAgent,
        data_generator_class=FloProSimulationDataGenerator,
        scenario=NORMAL_SCENARIO,
        max_iterations=50,
    )
    assert result.n_iterations > 0
