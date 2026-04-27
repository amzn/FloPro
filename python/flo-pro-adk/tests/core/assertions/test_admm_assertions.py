"""Tests for AgentAssertions."""

from __future__ import annotations

import numpy as np
import pytest

from flo_pro_sdk.agent.agent_definition import Objective, Solution
from flo_pro_sdk.core.variables import PublicVarGroupName

from flo_pro_adk.core.assertions.agent_assertions import AgentAssertions
from flo_pro_adk.core.exceptions.assertion_errors import VADKAssertionError

G = PublicVarGroupName("g")


@pytest.fixture()
def assertions() -> AgentAssertions:
    return AgentAssertions()


def _solution(preferred: dict) -> Solution:
    return Solution(preferred_vars=preferred, objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0))


def test_valid_solution_passes(assertions: AgentAssertions):
    assertions.assert_solution_valid(_solution({G: np.array([1.5, 2.5])}), {G: np.array([1.0, 2.0])})


def test_missing_group_raises(assertions: AgentAssertions):
    with pytest.raises(VADKAssertionError, match="Missing"):
        assertions.assert_solution_valid(_solution({}), {G: np.array([1.0])})


def test_shape_mismatch_raises(assertions: AgentAssertions):
    with pytest.raises(VADKAssertionError, match="Shape"):
        assertions.assert_solution_valid(_solution({G: np.array([1.0])}), {G: np.array([1.0, 2.0])})


def test_nan_raises(assertions: AgentAssertions):
    with pytest.raises(VADKAssertionError, match="Non-finite"):
        assertions.assert_solution_valid(_solution({G: np.array([float("nan")])}), {G: np.array([1.0])})


def test_deterministic_passes(assertions: AgentAssertions):
    def solve_fn(pv, p, r):
        return _solution({G: np.array([1.0])})

    assertions.assert_deterministic(solve_fn, {G: np.array([1.0])}, {G: np.zeros(1)}, {G: np.ones(1)})


def test_nondeterministic_raises(assertions: AgentAssertions):
    call_count = 0

    def solve_fn(pv, p, r):
        nonlocal call_count
        call_count += 1
        return _solution({G: np.array([float(call_count)])})

    with pytest.raises(VADKAssertionError, match="Non-deterministic"):
        assertions.assert_deterministic(solve_fn, {G: np.array([1.0])}, {G: np.zeros(1)}, {G: np.ones(1)})
