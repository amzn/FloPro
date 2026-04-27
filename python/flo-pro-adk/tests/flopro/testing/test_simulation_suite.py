"""Tests for FloProSimulationSuite."""

from __future__ import annotations

from flo_pro_sdk.agent.agent_definition import (
    AgentDefinition,
    Objective,
    Solution,
)
from flo_pro_sdk.core.variables import Prices, PublicVarValues, RhoValues

from flo_pro_adk.flopro.testing.simulation_suite import (
    FloProSimulationSuite,
)


class _StubAgent(AgentDefinition):
    """Returns consensus as preferred — always produces valid solutions."""

    def __init__(self, **kwargs: object) -> None:
        pass

    def solve(self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues) -> Solution:
        return Solution(
            preferred_vars={g: v.copy() for g, v in public_vars.items()},
            objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
        )


class _BrokenAgent(AgentDefinition):
    """Always raises — simulates a broken agent implementation."""

    def __init__(self, **kwargs: object) -> None:
        pass

    def solve(self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues) -> Solution:
        raise RuntimeError("Agent is broken")


def test_run_unit_passes_with_valid_agent():
    """Unit test suite should pass when agent returns structurally valid solutions."""
    exit_code = FloProSimulationSuite(_StubAgent).run_unit()
    assert exit_code == 0, "Expected all unit tests to pass with a valid stub agent"


def test_run_unit_fails_with_broken_agent():
    """Unit test suite should fail when agent raises during solve()."""
    exit_code = FloProSimulationSuite(_BrokenAgent).run_unit()
    assert exit_code != 0, "Expected unit tests to fail with a broken agent"


def test_run_all_returns_int():
    """run_all() should return a pytest exit code as int."""
    exit_code = FloProSimulationSuite(_StubAgent).run_all()
    assert isinstance(exit_code, int), f"Expected int exit code, got {type(exit_code)}"
