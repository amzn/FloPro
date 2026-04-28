"""Tests for InMemoryStateStore implementation."""

import numpy as np

from flo_pro_sdk.core.in_memory_state_store import InMemoryStateStore
from flo_pro_sdk.core.state import ConsensusState, AgentPlan
from flo_pro_sdk.agent.agent_definition import Objective, Solution
from flo_pro_sdk.core.variables import PublicVarGroupName


def _s(i: int) -> "ConsensusState":
    return ConsensusState(
        iteration=i,
        consensus_vars=np.array([float(i)]),
        agent_preferred_vars={"a": np.array([float(i)])},
        prices={"a": np.array([float(i * 0.1)])},
        rho={"a": np.array([1.0])},
    )


def _plan(agent_id: str, iteration: int) -> AgentPlan:
    return AgentPlan(
        agent_id=agent_id,
        iteration=iteration,
        solution=Solution(
            preferred_vars={PublicVarGroupName("y"): np.array([1.0])},
            objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
        ),
    )


class TestInMemoryStateStore:
    def test_store_and_get_state(self):
        board = InMemoryStateStore(cache_size=5)
        board.store_state(0, _s(0))
        assert board.get_state(0).iteration == 0

    def test_get_state_returns_none_for_missing(self):
        board = InMemoryStateStore(cache_size=5)
        assert board.get_state(0) is None

    def test_recent_states_descending(self):
        board = InMemoryStateStore(cache_size=5)
        for i in range(4):
            board.store_state(i, _s(i))
        recent = board.get_recent_states(3)
        assert [s.iteration for s in recent] == [3, 2, 1]

    def test_recent_states_fewer_than_requested(self):
        board = InMemoryStateStore(cache_size=5)
        board.store_state(0, _s(0))
        assert len(board.get_recent_states(10)) == 1

    def test_eviction(self):
        board = InMemoryStateStore(cache_size=3)
        for i in range(4):
            board.store_state(i, _s(i))
        assert board.get_state(0) is None
        assert board.get_state(3) is not None

    def test_eviction_maintains_bound(self):
        board = InMemoryStateStore(cache_size=3)
        for i in range(10):
            board.store_state(i, _s(i))
            # Verify cache size is maintained
            assert len(board._iteration_order) <= 3

    def test_agent_plan_crud(self):
        board = InMemoryStateStore(cache_size=5)
        board.store_agent_plan(0, "a1", _plan("a1", 0))
        assert board.get_agent_plan(0, "a1").agent_id == "a1"
        assert board.get_agent_plan(0, "missing") is None

    def test_multiple_agents_same_iteration(self):
        board = InMemoryStateStore(cache_size=5)
        board.store_agent_plan(0, "a1", _plan("a1", 0))
        board.store_agent_plan(0, "a2", _plan("a2", 0))
        assert board.get_agent_plan(0, "a1").agent_id == "a1"
        assert board.get_agent_plan(0, "a2").agent_id == "a2"

    def test_plans_evicted_with_states(self):
        board = InMemoryStateStore(cache_size=2)
        board.store_state(0, _s(0))
        board.store_agent_plan(0, "a1", _plan("a1", 0))
        board.store_state(1, _s(1))
        board.store_state(2, _s(2))
        assert board.get_agent_plan(0, "a1") is None

    def test_update_existing_state(self):
        board = InMemoryStateStore(cache_size=5)
        board.store_state(0, _s(0))
        board.store_state(
            0,
            ConsensusState(
                iteration=0,
                consensus_vars=np.array([999.0]),
                agent_preferred_vars={"a": np.zeros(1)},
                prices={"a": np.zeros(1)},
                rho={"a": np.ones(1)},
            ),
        )
        assert board.get_state(0).consensus_vars[0] == 999.0
        # Verify only one iteration is stored (not duplicated)
        assert len(board._iteration_order) == 1
