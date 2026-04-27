# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Flo Pro E2E coordination with real RetailerAgent and VendorAgent."""

from __future__ import annotations

import pytest

from flo_pro_adk.core.assertions.coordination_assertions import CoordinationAssertions
from flo_pro_adk.core.testing.e2e_test_runner import run_e2e_test
from flo_pro_adk.flopro.counterparty.retailer_agent import RetailerAgent
from flo_pro_adk.flopro.counterparty.vendor_agent import VendorAgent
from flo_pro_adk.flopro.testing.flopro_data_generator import FloProSimulationDataGenerator
from flo_pro_adk.flopro.testing.flopro_scenarios import NORMAL_SCENARIO, NON_CONVERGENCE_SCENARIO
from tests.flopro.testing.conftest import requires_xpress

assertions = CoordinationAssertions()
MAX_ITERATIONS = 2000


@requires_xpress
@pytest.mark.parametrize("agent_cls,counterparty_cls,tol", [
    (RetailerAgent, VendorAgent, 1e-2),
    (VendorAgent, RetailerAgent, 1e-4),
], ids=["retailer_vs_vendor", "vendor_vs_retailer"])
def test_normal_scenario_converges(agent_cls, counterparty_cls, tol):
    result = run_e2e_test(
        agent_class=agent_cls,
        counterparty_class=counterparty_cls,
        data_generator_class=FloProSimulationDataGenerator,
        scenario=NORMAL_SCENARIO,
        max_iterations=MAX_ITERATIONS,
    )
    assert result.n_iterations > 1, "Expected multi-iteration coordination"
    assertions.assert_convergence(result.final_state, primal_tol=tol, dual_tol=tol)


@requires_xpress
def test_non_convergence_completes():
    result = run_e2e_test(
        agent_class=RetailerAgent,
        counterparty_class=VendorAgent,
        data_generator_class=FloProSimulationDataGenerator,
        scenario=NON_CONVERGENCE_SCENARIO,
        max_iterations=100,
    )
    assert result.n_iterations > 1, "Expected multi-iteration coordination"
