"""Tests for core E2E test runner (domain-agnostic)."""

from __future__ import annotations

import pytest
from flo_pro_sdk.core.variables import PublicVarGroupName

from flo_pro_adk.core.testing.e2e_test_runner import (
    E2ETestResult,
    run_e2e_test,
)
from flo_pro_adk.core.testing.simulation_data_generator import (
    SimulationDataGenerator,
)
from flo_pro_adk.core.types.scenario_params import (
    ScenarioParams,
)
from tests.conftest import StubAgent, StubCounterpartyAgent

GROUP = PublicVarGroupName("test_var")

_SCENARIO = ScenarioParams(
    name="core_e2e_test",
    seed=99,
    n_variables=4,
    n_groups=1,
    price_distribution="uniform",
    price_range=(0.0, 1.0),
    rho=1.0,
    domain_params={},
)


class _StubDataGenerator(SimulationDataGenerator):
    def get_group_names(self) -> list[PublicVarGroupName]:
        return [GROUP]

    def generate_counterparty_input_data(self):  # type: ignore[override]
        return None


@pytest.fixture()
def e2e_result() -> E2ETestResult:
    return run_e2e_test(
        agent_class=StubAgent,
        counterparty_class=StubCounterpartyAgent,
        data_generator_class=_StubDataGenerator,
        scenario=_SCENARIO,
        max_iterations=10,
    )


def test_returns_e2e_result(e2e_result: E2ETestResult):
    assert isinstance(e2e_result, E2ETestResult)
    assert e2e_result.n_iterations > 0


def test_respects_max_iterations():
    result = run_e2e_test(
        agent_class=StubAgent,
        counterparty_class=StubCounterpartyAgent,
        data_generator_class=_StubDataGenerator,
        scenario=_SCENARIO,
        max_iterations=5,
    )
    assert result.n_iterations <= 5


def test_final_residuals_are_floats(e2e_result: E2ETestResult):
    assert e2e_result.final_residuals is not None
    primal, dual = e2e_result.final_residuals
    assert isinstance(primal, float)
    assert isinstance(dual, float)
