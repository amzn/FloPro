"""Property-based tests for InMemoryStateStore and persistence.

Covers:
- Cache size invariant (store never exceeds configured size)
- Recent states ordering (descending iteration order)
- All states persisted after finalize (via PersistingStoreWrapper)
"""

import tempfile

from hypothesis import given, settings, strategies as st

from flo_pro_sdk.core.in_memory_state_store import InMemoryStateStore
from flo_pro_sdk.core.persistence import PersistenceWriter, PersistingStoreWrapper
from flo_pro_sdk.core.state import ConsensusState
from flo_pro_sdk.core.persistence_backend import FileSystemBackend

import numpy as np


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------


@st.composite
def state_sequence_strategy(draw, min_size=0, max_size=50):
    """Generate a sequence of states with sequential iteration numbers."""
    num_states = draw(st.integers(min_value=min_size, max_value=max_size))
    return [
        ConsensusState(
            iteration=i,
            consensus_vars=np.array(
                [draw(st.floats(allow_nan=False, allow_infinity=False))]
            ),
            agent_preferred_vars={"a": np.array([0.0])},
            prices={
                "a": np.array([draw(st.floats(allow_nan=False, allow_infinity=False))])
            },
            rho={
                "a": np.array(
                    [
                        draw(
                            st.floats(
                                min_value=0.0, allow_nan=False, allow_infinity=False
                            )
                        )
                    ]
                )
            },
        )
        for i in range(num_states)
    ]


def _make_state(i: int) -> "ConsensusState":
    return ConsensusState(
        iteration=i,
        consensus_vars=np.array([float(i)]),
        agent_preferred_vars={"a": np.array([float(i)])},
        prices={"a": np.array([float(i * 0.1)])},
        rho={"a": np.array([1.0])},
    )


# ---------------------------------------------------------------------------
# Cache size invariant
# ---------------------------------------------------------------------------


class TestCacheSizeInvariant:
    @given(
        states=state_sequence_strategy(min_size=0, max_size=100),
        cache_size=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_cache_never_exceeds_configured_size(self, states, cache_size):
        board = InMemoryStateStore(cache_size=cache_size)
        for s in states:
            board.store_state(s.iteration, s)
            assert len(board._iteration_order) <= cache_size

    @given(cache_size=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=None)
    def test_eviction_preserves_most_recent(self, cache_size):
        board = InMemoryStateStore(cache_size=cache_size)
        n = cache_size * 3
        for i in range(n):
            board.store_state(i, _make_state(i))

        for i in range(n - cache_size, n):
            assert board.get_state(i) is not None
        for i in range(n - cache_size):
            assert board.get_state(i) is None


# ---------------------------------------------------------------------------
# Recent states ordering
# ---------------------------------------------------------------------------


class TestRecentStatesOrdering:
    @given(
        states=state_sequence_strategy(min_size=1, max_size=100),
        cache_size=st.integers(min_value=1, max_value=50),
        request_count=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100, deadline=None)
    def test_recent_states_descending_order(self, states, cache_size, request_count):
        board = InMemoryStateStore(cache_size=cache_size)
        for s in states:
            board.store_state(s.iteration, s)

        recent = board.get_recent_states(request_count)
        iters = [s.iteration for s in recent]
        assert iters == sorted(iters, reverse=True)

    @given(
        num_states=st.integers(min_value=2, max_value=100),
        cache_size=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_most_recent_first_and_consecutive(self, num_states, cache_size):
        board = InMemoryStateStore(cache_size=cache_size)
        for i in range(num_states):
            board.store_state(i, _make_state(i))

        recent = board.get_recent_states(cache_size)
        assert recent[0].iteration == num_states - 1
        for i in range(len(recent) - 1):
            assert recent[i].iteration - recent[i + 1].iteration == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestAllStatesPersisted:
    @given(
        states=state_sequence_strategy(min_size=1, max_size=30),
        cache_size=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_all_states_persisted_after_finalize(self, states, cache_size):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemBackend(base_dir=tmpdir)
            writer = PersistenceWriter(backend, InMemoryStateStore())
            wrapper = PersistingStoreWrapper(
                InMemoryStateStore(cache_size=cache_size), writer.queue
            )

            for s in states:
                wrapper.store_state(
                    s.iteration, s, timestamp=1710000000.0 + s.iteration
                )
            writer.finalize()

            for s in states:
                persisted = backend.read_state(s.iteration)
                assert persisted is not None
                assert persisted["iteration"] == s.iteration

    @given(num_states=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_state_file_count_matches(self, num_states):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemBackend(base_dir=tmpdir)
            writer = PersistenceWriter(backend, InMemoryStateStore())
            wrapper = PersistingStoreWrapper(
                InMemoryStateStore(cache_size=2), writer.queue
            )

            for i in range(num_states):
                wrapper.store_state(i, _make_state(i), timestamp=1710000000.0 + i)
            writer.finalize()

            # After finalize, L2 compaction merges into convergence.parquet
            from flo_pro_sdk.core.persistence_backend import (
                CONVERGENCE_DIR,
                L2_CONVERGENCE_FILE,
            )
            import pyarrow.parquet as pq

            data_path = backend.run_dir / CONVERGENCE_DIR / L2_CONVERGENCE_FILE
            assert data_path.exists()
            table = pq.read_table(data_path)
            assert table.num_rows == num_states
