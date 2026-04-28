"""Tests for PersistenceWriter and PersistingStoreWrapper."""

import time
from queue import Queue
from unittest.mock import Mock

import numpy as np

from flo_pro_sdk.core.in_memory_state_store import InMemoryStateStore
from flo_pro_sdk.core.persistence import PersistenceWriter, PersistingStoreWrapper
from flo_pro_sdk.core.state import ConsensusState, AgentPlan
from flo_pro_sdk.core.state_store import DirectRef
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


class TestPersistingStoreWrapper:
    def test_store_state_delegates_and_enqueues(self):
        inner = InMemoryStateStore(cache_size=5)
        q: Queue = Queue()
        wrapper = PersistingStoreWrapper(inner, q)

        wrapper.store_state(0, _s(0))
        assert inner.get_state(0).iteration == 0

        item = q.get_nowait()
        assert item[0] == "state" and item[1] == 0 and item[3] is None

    def test_store_agent_plan_delegates_and_enqueues(self):
        inner = InMemoryStateStore(cache_size=5)
        q: Queue = Queue()
        wrapper = PersistingStoreWrapper(inner, q)

        wrapper.store_agent_plan(0, "a1", _plan("a1", 0))
        assert inner.get_agent_plan(0, "a1") is not None

        item = q.get_nowait()
        assert item[:3] == ("plan", 0, "a1")

    def test_reads_delegate_to_inner(self):
        inner = InMemoryStateStore(cache_size=5)
        wrapper = PersistingStoreWrapper(inner, Queue())

        wrapper.store_state(0, _s(0))
        assert wrapper.get_state(0).iteration == 0
        assert wrapper.get_state(99) is None
        assert len(wrapper.get_recent_states(5)) == 1


class TestPersistenceWriter:
    def _mock_store(self):
        store = Mock()
        store.write_state = Mock()
        store.write_agent_plan = Mock()
        store.close = Mock()
        return store

    def test_writes_state_to_store(self):
        store = self._mock_store()
        writer = PersistenceWriter(store, InMemoryStateStore())
        state = _s(0)
        writer.queue.put(("state", 0, DirectRef(state), None))
        writer.finalize()
        store.write_state.assert_called_once_with(0, state, timestamp=None)

    def test_writes_agent_plan_to_store(self):
        store = self._mock_store()
        writer = PersistenceWriter(store, InMemoryStateStore())
        plan = _plan("a1", 0)
        writer.queue.put(("plan", 0, "a1", DirectRef(plan)))
        writer.finalize()
        store.write_agent_plan.assert_called_once_with(0, "a1", plan)

    def test_finalize_drains_queue_and_closes(self):
        store = self._mock_store()
        writer = PersistenceWriter(store, InMemoryStateStore())
        for i in range(3):
            writer.queue.put(("state", i, DirectRef(_s(i)), None))
            writer.queue.put(("plan", i, f"a{i}", DirectRef(_plan(f"a{i}", i))))
        writer.finalize()
        assert store.write_state.call_count == 3
        assert store.write_agent_plan.call_count == 3
        store.close.assert_called_once()

    def test_write_behind_is_non_blocking(self):
        store = self._mock_store()
        store.write_state = Mock(side_effect=lambda *_: time.sleep(0.1))
        writer = PersistenceWriter(store, InMemoryStateStore())

        t0 = time.time()
        writer.queue.put(("state", 0, DirectRef(_s(0)), None))
        assert time.time() - t0 < 0.05

        writer.finalize()
